"""The things the bot does each cycle. Each task: run(img) -> bool (True = it changed the screen)."""
import json
import logging
import time

from pathlib import Path

import cv2

import ocr
import vision

log = logging.getLogger("frogbot")
ROOT_TEMPLATES = Path(__file__).parent / "templates"


class Task:
    def __init__(self, bot, cfg):
        self.bot, self.cfg = bot, cfg
        self.every = cfg.get("every_seconds", 1.0)
        self.last = 0.0

    def due(self):
        return time.time() - self.last >= self.every

    def run(self, img) -> bool:
        raise NotImplementedError


class UnlockNew(Task):
    """Anything marked NEW -> tap it -> press its green unlock/build button so it starts working."""

    def run(self, img):
        # template cut from the real sticker (reliable) + colour/shape backup
        items = vision.match_templates(img, ROOT_TEMPLATES / "new", self.cfg.get("threshold", 0.75),
                                       self.bot.ignore)
        for d in vision.find_new_items(img, self.bot.ignore):
            if all(abs(d.cx - o.cx) > 40 or abs(d.cy - o.cy) > 40 for o in items):
                items.append(d)
        for it in items[: self.cfg.get("max_per_cycle", 2)]:
            log.info("NEW item at (%d,%d) -> opening", it.cx, it.cy)
            if not self.bot.tap(it.cx, it.cy):
                continue
            self.bot.wait(self.cfg.get("popup_wait", 0.8))
            # NEW items open the same 'Level' popup (or a window with an X)
            ok = lambda im: self.bot.is_item_popup(im) or vision.find_close_button(im) is not None
            bought = self.bot.buy_until_grey(self.cfg.get("max_clicks", 3), screen_ok=ok)
            log.info("  unlocked" if bought else "  nothing to buy (not enough coins yet?)")
            self.bot.close_item_popup(it.cx, it.cy)
            self.bot.close_with_x()
        return bool(items)


class UpgradeItems(Task):
    """Red badge with a white up arrow on a food/station -> tap it -> press green buy button."""

    def run(self, img):
        badges = vision.find_upgrade_badges(img, self.bot.ignore)
        if not badges:
            return False
        ox, oy = self.cfg.get("tap_offset", [1.2, 1.3])
        H = img.shape[0]
        # items at the very top have no room for their popup: wait until scrolled into view
        badges = [b for b in badges if b.cy > self.cfg.get("min_y", 0.12) * H]
        for b in badges[: self.cfg.get("max_per_cycle", 3)]:
            ix, iy = b.cx + ox * b.w, b.cy + oy * b.h
            log.info("upgrade badge at (%d,%d) -> opening item", b.cx, b.cy)
            if not self.bot.tap(ix, iy):
                continue
            self.bot.wait(self.cfg.get("popup_wait", 0.7))
            if not self.bot.is_item_popup(self.bot.screenshot()):
                log.info("  no item popup appeared, skipping")
                continue
            bought = self.bot.buy_until_grey(self.cfg.get("max_clicks_per_item", 10),
                                             screen_ok=self.bot.is_item_popup)
            log.info("  bought %d level(s)", bought)
            self.bot.close_item_popup(ix, iy)
        return True


class ProfitUpgrade(Task):
    """Strategy: spend coins where they earn back fastest (shortest PAYBACK time).

    For every upgradeable item in view it opens the popup and reads (OCR):
        profit per dish P, cooking time T seconds, level L, cost C of the next level.
    Then:
        earning rate        R  = P / T                       (coins per second now)
        gain of one level   dP = measured (P_after - P_before) / levels bought,
                                 or P / L until measured     (linear growth estimate)
        payback (level)     = C / (dP / T)                   seconds to earn the cost back
        payback (star)      = coins needed for the next star / R
                              (only when the star bar shows the star is affordable;
                               assumes a star doubles the item's income)
        payback             = min(level, star) * red_weight if the badge is red
    The lowest payback is bought first, levels in bursts, re-reading after each burst; it
    switches when the item stops being the best. If a clearly better item was seen
    elsewhere recently, coins are saved for it instead of being spent here."""

    def __init__(self, bot, cfg):
        super().__init__(bot, cfg)
        self.dp = {}        # name -> measured profit gain per level
        self.seen = {}      # name -> (payback, time) from recent surveys
        self.saving_since = {}  # name -> when we started saving coins for it

    # --- reading ------------------------------------------------------------
    def _open(self, ix, iy):
        if not self.bot.tap(ix, iy):
            return None
        self.bot.wait(self.cfg.get("popup_wait", 0.5))
        img = self.bot.screenshot()
        t = vision.find_template(img, ROOT_TEMPLATES / "popup")
        return (img, t) if t is not None else None

    def _read(self, img, t):
        info = ocr.read_popup(img, t)
        if info:
            info["bar"] = vision.read_item_popup(img, t)
        return info

    def _balance(self, img):
        H, W = img.shape[:2]
        return ocr.read_amount(img, (0.12, 0.012, 0.31, 0.062))

    # --- the calculation ----------------------------------------------------
    def payback(self, info, balance, red):
        P, T, C, L = info["profit"], info["seconds"], info["cost"], max(info["level"] or 1, 1)
        dP = self.dp.get(info["name"]) or P / L
        pb = C / (dP / T)
        bar = info.get("bar")
        if bar and balance and bar["gain"] > 0 and bar["done"] + bar["gain"] >= 0.97:
            to_star = balance * (1 - bar["done"]) / bar["gain"]
            pb = min(pb, to_star / (P / T))
        if red:
            pb *= self.cfg.get("red_weight", 0.7)
        return pb

    # --- main ----------------------------------------------------------------
    def run(self, img):
        H = img.shape[0]
        badges = [b for b in vision.find_upgrade_badges(img, self.bot.ignore)
                  if b.cy > self.cfg.get("min_y", 0.12) * H]
        if not badges:
            return False
        ox, oy = self.cfg.get("tap_offset", [1.2, 1.3])

        # PRIORITY 3: red arrows are upgraded straight away - no evaluation
        reds = [b for b in badges if vision.badge_color(img, b) == "red"]
        if reds:
            self.last_red = time.time()
            for b in reds[: self.cfg.get("max_red_per_cycle", 4)]:
                ix, iy = b.cx + ox * b.w, b.cy + oy * b.h
                log.info("red arrow at (%d,%d) -> upgrading", b.cx, b.cy)
                if not self.bot.tap(ix, iy):
                    continue
                self.bot.wait(self.cfg.get("popup_wait", 0.5))
                if not self.bot.is_item_popup(self.bot.screenshot()):
                    continue
                n = self.bot.buy_until_grey(self.cfg.get("max_clicks", 45),
                                            screen_ok=self.bot.is_item_popup)
                log.info("  pressed buy ~%d times", n)
                self.bot.close_item_popup(ix, iy)
            return True

        # PRIORITY 4: orange arrows - only once no red has been seen for a whole scroll
        # sweep (reds elsewhere come first), then ordered by the payback evaluation
        if time.time() - getattr(self, "last_red", 0) < self.cfg.get("red_sweep_seconds", 25):
            return False
        balance = self._balance(img)
        cands = []
        for b in badges[: self.cfg.get("max_survey", 6)]:
            red = False
            ix, iy = b.cx + ox * b.w, b.cy + oy * b.h
            opened = self._open(ix, iy)
            if opened is None:
                continue
            info = self._read(*opened)
            self.bot.close_item_popup(ix, iy)
            if not info:
                continue
            pb = self.payback(info, balance, red)
            if pb < self.cfg.get("min_payback", 1.0):
                log.info("  ignoring implausible reading for %s: %s", info["name"], info["text"])
                continue
            cands.append((pb, ix, iy, red, info))
            if info["name"]:
                self.seen[info["name"]] = (pb, time.time())
        if not cands:
            return True
        cands.sort(key=lambda c: c[0])
        log.info("orange payback ranking: " + " | ".join(
            f"{c[4]['name'] or '?'}{'(red)' if c[3] else ''} {c[4]['text']} -> {c[0]:.0f}s"
            for c in cands))

        # save coins if a clearly better item lives in another part of the restaurant
        fresh = {n: pb for n, (pb, ts) in self.seen.items()
                 if time.time() - ts < self.cfg.get("memory_seconds", 120)}
        here = {c[4]["name"] for c in cands}
        if fresh:
            g_name = min(fresh, key=fresh.get)
            if g_name not in here and cands[0][0] > self.cfg.get("save_factor", 2.0) * fresh[g_name]:
                since = self.saving_since.setdefault(g_name, time.time())
                if time.time() - since < self.cfg.get("save_max_seconds", 60):
                    log.info("  saving coins for %s (payback %.0fs elsewhere)", g_name, fresh[g_name])
                    return True
                log.info("  waited long enough for %s, spending here", g_name)
                self.seen.pop(g_name, None)
            self.saving_since.clear()

        for i, (pb, ix, iy, red, info) in enumerate(cands[: self.cfg.get("max_buy_items", 3)]):
            next_pb = cands[i + 1][0] if i + 1 < len(cands) else float("inf")
            if not self._rush(ix, iy, red, next_pb, balance):
                break  # out of coins
        return True

    def _rush(self, ix, iy, red, next_pb, balance):
        """Buy levels while this item still has the best payback. False = out of coins."""
        opened = self._open(ix, iy)
        if opened is None:
            return True
        img, t = opened
        first = cur = self._read(img, t)
        burst, bought, affordable = self.cfg.get("burst", 3), 0, True
        while cur and bought < self.cfg.get("max_clicks", 45):
            if self.bot.guard(img):
                return False
            btns = [b for b in vision.find_green_buttons(img) if b.y > t.y]
            if not btns:
                affordable = False
                break
            if not self.bot.tap(btns[0].cx, btns[0].cy, times=burst):
                break
            bought += burst
            self.bot.wait(self.bot.cfg.get("buy_delay", 0.25))
            img = self.bot.screenshot()
            t = vision.find_template(img, ROOT_TEMPLATES / "popup")
            if t is None:
                break
            new = self._read(img, t)
            if not new:
                break
            # learn the real gain per level (skip if a star multiplied the profit)
            if new["level"] and cur["level"] and new["level"] > cur["level"]:
                ratio = new["profit"] / cur["profit"]
                if ratio < 1.5:
                    self.dp[new["name"]] = (new["profit"] - cur["profit"]) / (new["level"] - cur["level"])
                else:
                    log.info("  * star reached on %s: profit x%.1f", new["name"], ratio)
            cur = new
            if self.payback(cur, balance, red) > next_pb:
                break  # another item is now the better buy
        if first and cur:
            log.info("  bought %s: level %s -> %s, %s", cur["name"], first["level"], cur["level"], cur["text"])
            if cur["name"]:  # remember its new (higher) payback
                self.seen[cur["name"]] = (self.payback(cur, balance, red), time.time())
                self.saving_since.pop(cur["name"], None)
        self.bot.close_item_popup(ix, iy)
        return affordable


class Planner(ProfitUpgrade):
    """Whole-restaurant planner with a dish database (dishes.json, rebuilt every scan).

    SCAN   scroll top -> bottom. On every view: priority 1+2 jobs (Manage Eatery, egg, NEW,
           table coins) first. Then every arrow item is opened and noted in the database, in
           restaurant order: colour, position, profit, time, cost, payback (OCR).
           A RED is bought right there while its popup is open - unless the database already
           knows a red with a better payback (tie-breaker). Unreadable reds are still bought.
    ACT    any red in the database -> go to the best red and buy it. If it is not affordable,
           keep the coins for it (no orange is bought) for up to red_wait_seconds.
           No red at all -> best orange by payback.
    Rescan every rescan_seconds (coins grow, arrows change colour)."""

    INF = float("inf")

    def __init__(self, bot, cfg):
        super().__init__(bot, cfg)
        self.db = {}                 # key -> record
        self.scanned_at = 0.0
        self.waiting_red = None      # (key, since)
        self.db_file = ROOT_TEMPLATES.parent / "dishes.json"

    # --- scrolling ---------------------------------------------------------------
    def _up(self, img):
        return self.bot.drag(img, 0.5, 0.30, 0.75)

    def _down(self, img):
        return self.bot.drag(img, 0.5, 0.72, 0.38)

    def _to_top(self, img):
        for _ in range(10):
            img, moved = self._up(img)
            if not moved:
                break
        return img

    def _go_to(self, k):
        img = self._to_top(self.bot.screenshot())
        for _ in range(k):
            img, moved = self._down(img)
            if not moved:
                break
        return img

    @staticmethod
    def _shift(a, b):
        """Pixels the content moved up between screenshots a and b."""
        sw, sh = 225, 400
        g1 = cv2.cvtColor(cv2.resize(a, (sw, sh)), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.resize(b, (sw, sh)), cv2.COLOR_BGR2GRAY)
        y0, y1, top = int(0.55 * sh), int(0.72 * sh), int(0.1 * sh)
        res = cv2.matchTemplate(g2[top:int(0.9 * sh)], g1[y0:y1, 20:sw - 20], cv2.TM_CCOEFF_NORMED)
        _, _, _, loc = cv2.minMaxLoc(res)
        return (y0 - (loc[1] + top)) * a.shape[0] / sh

    def _quick(self, img):
        for name in self.cfg.get("quick_tasks", ["manage_eatery", "tap_egg", "unlock_new", "collect_coins"]):
            t = self.bot.task_by_name.get(name)
            if t and t.run(img):
                img = self.bot.screenshot()
        return img

    # --- database ------------------------------------------------------------------
    def _save_db(self):
        rows = sorted(self.db.values(), key=lambda r: r["ay"])
        out = [{k: v for k, v in r.items() if k != "info"} | {"text": r["info"]["text"] if r["info"] else None}
               for r in rows]
        try:
            self.db_file.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        except OSError:
            pass

    def _log_db(self):
        rows = sorted(self.db.values(), key=lambda r: r["ay"])
        n = {c: sum(r["colour"] == c for r in rows) for c in ("red", "orange", "none")}
        log.info("DISH DATABASE (top -> bottom): %d dishes - %d red, %d orange, %d no arrow now",
                 len(rows), n["red"], n["orange"], n["none"])
        for i, r in enumerate(rows, 1):
            colour = r["colour"] if r["colour"] != "none" else "-"
            log.info("  %2d. %-6s %-16s %-30s payback %s", i, colour, r["name"] or "?",
                     r["info"]["text"] if r["info"] else "(unreadable)",
                     "?" if r["pb"] == self.INF else f"{r['pb']:.0f}s")

    def _reds(self):
        return sorted((r for r in self.db.values() if r["colour"] == "red"), key=lambda r: r["pb"])

    # --- buying in an open popup ----------------------------------------------------
    def _buy_here(self, img, t, ix, iy, rec, stop_pb):
        """Popup is open. Buy in bursts until grey, or until payback > stop_pb.
        Returns True if it ended because the button was grey (not affordable)."""
        balance = getattr(self, "balance", None)
        cur = first = rec["info"]
        first_level = cur["level"] if cur else None
        bought, grey, bursts = 0, False, 0
        burst = self.cfg.get("burst", 3)
        # OCR is the slow part: reds are bought without reading in between; oranges are
        # re-read only every few bursts to check they are still the best buy
        ocr_every = 0 if stop_pb == self.INF else self.cfg.get("ocr_every_bursts", 3)
        while bought < self.cfg.get("max_clicks", 45):
            if self.bot.guard(img):
                break
            btns = [x for x in vision.find_green_buttons(img) if x.y > t.y]
            if not btns:
                grey = True
                break
            if not self.bot.tap(btns[0].cx, btns[0].cy, times=burst):
                break
            bought += burst
            bursts += 1
            self.bot.wait(self.bot.cfg.get("buy_delay", 0.25))
            img = self.bot.screenshot()
            t = vision.find_template(img, ROOT_TEMPLATES / "popup")
            if t is None:
                break
            if ocr_every and bursts % ocr_every == 0:
                cur = self._read(img, t) or cur
                if cur and self.payback(cur, balance, False) > stop_pb:
                    break
        if t is not None:                           # one final reading for the database
            cur = self._read(img, t) or cur
        if first and cur and first.get("level") and cur.get("level") and cur["level"] > first["level"]:
            ratio = cur["profit"] / first["profit"]
            if ratio < 1.5:
                self.dp[cur["name"]] = (cur["profit"] - first["profit"]) / (cur["level"] - first["level"])
            else:
                log.info("  * star reached on %s: profit x%.1f", cur["name"], ratio)
        if cur:
            rec["info"], rec["pb"] = cur, self.payback(cur, balance, False)
        log.info("  %s %s: pressed buy ~%d (level %s -> %s)%s", rec["colour"], rec["name"] or "?",
                 bought, first_level, cur["level"] if cur else "?", " - not affordable" if grey else "")
        self.bot.close_item_popup(ix, iy)
        return grey

    # --- scan --------------------------------------------------------------------------
    def scan(self):
        H = self.bot.screenshot().shape[0]
        img = self._to_top(self.bot.screenshot())
        old_reds = {r["name"]: r["pb"] for r in self._reds() if r["name"]}
        # the database persists: every known dish stays listed with its last numbers; this
        # scan only refreshes which dishes show an arrow now (red / orange / none)
        self.db = {k: v for k, v in self.db.items() if v["name"]}
        for r in self.db.values():
            r["colour"] = "none"
        seen, offset = [], 0.0
        ox, oy = self.cfg.get("tap_offset", [1.2, 1.3])
        for k in range(self.cfg.get("max_views", 8)):
            img = self._quick(img)
            self.balance = self._balance(img) or getattr(self, "balance", None)
            for b in vision.find_upgrade_badges(img, self.bot.ignore):
                ay = b.cy + offset
                if any(abs(ay - sy) < 45 and abs(b.cx - sx) < 45 for sx, sy in seen):
                    continue                       # same dish seen from the previous view
                if not (0.18 * H < b.cy < 0.86 * H):
                    continue                       # edge: noted from a view where it is central
                seen.append((b.cx, ay))
                colour = vision.badge_color(img, b)
                ix, iy = b.cx + ox * b.w, b.cy + oy * b.h
                opened = self._open(ix, iy)
                if opened is None and colour == "red":   # reds get a second try
                    self.bot.wait(0.4)
                    opened = self._open(ix, iy)
                info = self._read(*opened) if opened else None
                pb = self.payback(info, self.balance, False) if info else self.INF
                if pb < self.cfg.get("min_payback", 1.0):
                    pb = self.INF
                name = info["name"] if info else None
                rec = {"k": k, "x": b.cx, "y": b.cy, "ay": round(ay), "colour": colour,
                       "name": name, "info": info, "pb": pb, "seen": time.strftime("%H:%M:%S")}
                self.db[name or f"pos{round(ay)}_{b.cx}"] = rec
                if opened is None:
                    continue
                if colour == "red":
                    better = [p for n, p in old_reds.items() if n != name and p < pb]
                    if not better:                 # best red we know of: buy it right now
                        log.info("RED %s found -> buying now", name or "?")
                        if self._buy_here(*opened, ix, iy, rec, self.INF):
                            self.waiting_red = (name, time.time())
                        continue
                self.bot.close_item_popup(ix, iy)
            img = self.bot.screenshot()
            new, moved = self._down(img)
            if not moved:
                break
            offset += self._shift(img, new)
            img = new
        self.scanned_at = time.time()
        self._log_db()
        self._save_db()

    # --- act -------------------------------------------------------------------------
    def run(self, img):
        if not self.db or time.time() - self.scanned_at > self.cfg.get("rescan_seconds", 90):
            self.scan()
            return True
        reds = self._reds()
        if reds:
            target, pool = reds[0], reds
        else:
            self.waiting_red = None
            pool = sorted((r for r in self.db.values() if r["colour"] == "orange"),
                          key=lambda r: r["pb"])
            target = pool[0] if pool else None
        if target is None:
            self.scanned_at = 0
            return False
        # a red we cannot afford yet: keep the coins for it (no oranges meanwhile)
        if self.waiting_red and target["colour"] == "red":
            since = self.waiting_red[1]
            if time.time() - since < self.cfg.get("retry_red_seconds", 15):
                return False                       # give coins time to grow
            if time.time() - since > self.cfg.get("red_wait_seconds", 120):
                log.info("waited %ds for red %s - rescanning", time.time() - since, target["name"])
                self.waiting_red = None
                self.scanned_at = 0
                return True
        nxt = pool[1]["pb"] if len(pool) > 1 else self.INF
        log.info("DECIDE: %s %s (payback %s)", target["colour"], target["name"] or "?",
                 "?" if target["pb"] == self.INF else f"{target['pb']:.0f}s")
        img = self._go_to(target["k"])
        img = self._quick(img)
        near = [b for b in vision.find_upgrade_badges(img, self.bot.ignore)
                if abs(b.cx - target["x"]) < 60 and abs(b.cy - target["y"]) < 90]
        key = next(k for k, v in self.db.items() if v is target)
        if not near:
            log.info("  %s has no arrow any more - kept in the database", target["name"] or key)
            target["colour"] = "none"
            self.waiting_red = None
            self._save_db()
            return True
        b = min(near, key=lambda b: abs(b.cx - target["x"]) + abs(b.cy - target["y"]))
        target["colour"] = vision.badge_color(img, b)
        ox, oy = self.cfg.get("tap_offset", [1.2, 1.3])
        ix, iy = b.cx + ox * b.w, b.cy + oy * b.h
        opened = self._open(ix, iy)
        if opened is None:
            self.scanned_at = 0
            return True
        info = self._read(*opened)
        if info and target["name"] and info["name"] and info["name"] != target["name"]:
            log.info("  found %s instead of %s - rescanning", info["name"], target["name"])
            self.bot.close_item_popup(ix, iy)
            self.scanned_at = 0
            return True
        target["info"] = info or target["info"]
        grey = self._buy_here(*opened, ix, iy, target, nxt if target["colour"] != "red" else self.INF)
        if grey and target["colour"] == "red":
            if not self.waiting_red:
                self.waiting_red = (target["name"], time.time())
        else:
            self.waiting_red = None
        if grey and target["colour"] != "red":
            target["pb"] = self.INF               # unaffordable orange: next one
        # arrow still there?
        img = self.bot.screenshot()
        still = [x for x in vision.find_upgrade_badges(img, self.bot.ignore)
                 if abs(x.cx - b.cx) < 60 and abs(x.cy - b.cy) < 60]
        if not still:
            target["colour"] = "none"               # bought out: dish stays listed
            self.waiting_red = None
        else:
            target["colour"] = vision.badge_color(img, still[0])
        self._save_db()
        return True


class ManageEatery(Task):
    """Green up-arrow button (bottom right) -> Manage Eatery list -> buy every green button."""

    def run(self, img):
        H, W = img.shape[:2]
        bx, by = self.cfg.get("button_pos", [0.88, 0.96])
        # only open when the button shows its red '!' (= something is affordable)
        dot = self.cfg.get("badge_region", [0.78, 0.91, 0.85, 0.96])
        if self.cfg.get("only_when_badged", True) and vision.red_fraction(img, dot) < 0.05:
            return False
        log.info("opening Manage Eatery")
        self.bot.tap(bx * W, by * H)
        self.bot.wait(self.cfg.get("popup_wait", 0.8))
        title = self.bot.eatery_title(self.bot.screenshot())
        if title is None:
            log.info("  Manage Eatery did not open, skipping")
            return True
        # only buttons inside the panel (below its title): ignores e.g. 'Froggy Bank: Open'
        ok = lambda im: self.bot.eatery_title(im) is not None
        # just the green buttons that are visible - no list scrolling
        bought = self.bot.buy_until_grey(self.cfg.get("max_buys", 15), min_w=0.12,
                                         screen_ok=ok, below_y=title.y + title.h)
        log.info("  bought %d eatery upgrade(s)", bought)
        self.bot.close_with_x()
        return True


class CollectCoins(Task):
    def run(self, img):
        coins = vision.find_coins(img, self.bot.ignore)
        for c in coins[: self.cfg.get("max_taps", 15)]:
            self.bot.tap(c.cx, c.cy)
        if coins:
            log.info("collected %d coin pile(s)", len(coins))
        return bool(coins)


class TapTemplates(Task):
    """Taps anything matching your own cropped PNGs in a folder (e.g. order bubbles)."""

    def run(self, img):
        hits = vision.match_templates(img, self.cfg.get("dir", "templates/tap"),
                                      self.cfg.get("threshold", 0.85), self.bot.ignore)
        for h in hits:
            self.bot.tap(h.cx, h.cy)
        if hits:
            log.info("tapped %d template match(es): %s", len(hits), sorted({h.kind for h in hits}))
        return bool(hits)


class Scroll(Task):
    """Pans the restaurant one step and reverses at the top/bottom, so every station
    is visited. Keep this last in config so the other tasks run on each view first."""

    def __init__(self, bot, cfg):
        super().__init__(bot, cfg)
        self.direction = 1  # 1 = towards the bottom of the restaurant

    def run(self, img):
        step = self.cfg.get("step", 0.3) / 2
        x, mid = self.cfg.get("x", 0.5), self.cfg.get("y", 0.5)
        # finger moves up to reveal content further down
        _, moved = self.bot.drag(img, x, mid + step * self.direction, mid - step * self.direction)
        if not moved:
            self.direction *= -1
            log.info("reached the %s, scrolling back", "bottom" if self.direction < 0 else "top")
        return moved


TASKS = {
    "unlock_new": UnlockNew,
    "upgrade_items": UpgradeItems,
    "smart_upgrade": ProfitUpgrade,
    "planner": Planner,
    "manage_eatery": ManageEatery,
    "collect_coins": CollectCoins,
    "tap_templates": TapTemplates,
    "tap_egg": TapTemplates,   # same logic, its own folder (templates/egg)
    "scroll": Scroll,
}
