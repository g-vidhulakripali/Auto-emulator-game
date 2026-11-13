"""Frog's Kitchen auto-player.

  python bot.py                 run the bot (Ctrl+C to stop)
  python bot.py --dry-run       detect and log, but never tap
  python bot.py --debug         also save annotated screenshots to debug/
  python bot.py --image X.png   offline: run detectors on a saved screenshot
"""
import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import yaml

import ocr
import vision
from tasks import TASKS

ROOT = Path(__file__).parent
log = logging.getLogger("frogbot")


def _mem_mb():
    """This process's private memory in MB (Windows), to spot memory growth early."""
    try:
        import ctypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + \
                       [(n, ctypes.c_size_t) for n in (
                           "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                           "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                           "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                           "PrivateUsage")]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(pmc)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(k32.GetCurrentProcess()),
                                                 ctypes.byref(pmc), pmc.cb)
        return int(pmc.PrivateUsage / 1048576)
    except Exception:
        return -1


class Bot:
    def __init__(self, cfg, dry_run=False, debug=False):
        from device import Device
        self.cfg = cfg
        self.dry_run, self.debug = dry_run, debug
        self.ignore = [tuple(r) for r in cfg.get("ignore_regions", [])]
        # taps are blocked in the HUD except where the bot needs a specific menu button
        self.ignore_taps = [tuple(r) for r in cfg.get("no_tap_regions", [])]
        self.last_img, self._alerts = None, None
        self.banner_block, self.free_ok = None, False
        self.tap_lock = threading.Lock()
        self.dev = Device(cfg["adb_path"], cfg["serials"])
        self.task_by_name = {name: TASKS[name](self, tcfg) for name, tcfg in cfg["tasks"].items()
                             if tcfg.get("enabled", True)}
        self.tasks = list(self.task_by_name.values())
        self.shape = None

    # --- primitives -------------------------------------------------------
    def screenshot(self):
        img = self.dev.screenshot_raw()
        self.shape = img.shape
        self.last_img, self._alerts = img, None
        return img

    # --- FREE banner watcher (background thread) -----------------------------------
    def start_banner_watch(self):
        """FREE banners last only a moment, so a separate thread watches the banner strip
        ~3x a second regardless of what the main loop is doing."""
        self._free_tpl = cv2.imread(str(ROOT / "templates" / "free" / "free_word.png"))
        threading.Thread(target=self._watch_banners, daemon=True).start()

    def _watch_banners(self):
        H = W = None
        x1, y1, x2, y2 = vision.BANNER_STRIP
        video_noted = False
        while True:
            try:
                img = self.dev.screenshot_raw()
                H, W = img.shape[:2]
                if vision.banner_gold(img) < self.cfg.get("banner_guard_gold", 0.20):
                    self.banner_block, video_noted = None, False
                    time.sleep(self.cfg.get("banner_poll", 0.3))
                    continue
                self.banner_block = vision.BANNER_STRIP      # hands off until proven FREE
                oy, ox = int(y1 * H) - 40, int(x1 * W) - 40
                strip = img[oy:int(y2 * H) + 40, ox:]
                # fast path: picture of the word FREE (~7 ms). Video banners score <= 0.4
                r = cv2.matchTemplate(strip, self._free_tpl, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(r)
                th, tw = self._free_tpl.shape[:2]
                if score >= self.cfg.get("free_match", 0.75):
                    self._claim_free(img, ox + loc[0] + tw // 2, oy + loc[1] + th // 2,
                                     f"template {score:.2f}")
                    continue
                # backup: OCR once per banner (if FREE is drawn slightly differently)
                if not video_noted and vision.banner_gold(img) >= self.cfg.get("banner_gold_min", 0.28):
                    b = ocr.read_banner(img, vision.BANNER_STRIP)
                    if b and b["free"]:
                        self._claim_free(img, *b["tap"], f"ocr {b['texts']}")
                        continue
                    if b:
                        video_noted = True
                        log.info("video-ad banner showing - not touching it  %s", b["texts"])
                        self._save_banner(img, False)
            except Exception as e:  # any error here stops the whole bot (main loop checks)
                log.debug("banner watcher error", exc_info=True)
                self.fatal = e
                return
            time.sleep(self.cfg.get("banner_poll", 0.3))

    def _claim_free(self, img, x, y, why):
        if self.ad_showing():
            return
        log.info("FREE banner -> claiming it (%s)", why)
        self._save_banner(img, True)
        with self.tap_lock:
            self.dev.tap(x, y)
        self.banner_block = None
        time.sleep(1.0)

    def _save_banner(self, img, free):
        H, W = img.shape[:2]
        x1, y1, x2, y2 = vision.BANNER_STRIP
        out = ROOT / "screenshots" / "banners"
        out.mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%H%M%S')}_{'FREE' if free else 'video'}.png"
        cv2.imwrite(str(out / name), img[int(y1 * H) - 30:int(y2 * H) + 30, int(x1 * W) - 40:])

    def safe_to_tap(self, x, y) -> bool:
        """RULE: never tap a red/orange '!' badge or the HUD (ignore_regions)."""
        img = self.last_img
        if img is None:
            return False
        H, W = img.shape[:2]
        # the ONE exception: the Manage Eatery button may be pressed even with its '!' -
        # but ONLY if the real green arrow button is visible there (an ad banner can cover
        # the bottom menu, with a 'Play now' button in the same spot)
        for ax, ay, ar in self.cfg.get("allow_tap_points", []):
            if ((x / W - ax) ** 2 + (y / H - ay) ** 2) ** 0.5 <= ar:
                if self.eatery_button_visible(self.dev.screenshot_raw()):
                    return True
                log.warning("refused tap at (%d,%d): Manage Eatery button not visible "
                            "(ad banner over the menu?)", x, y)
                return False
        # an ad is in front: never touch anything (you close ads yourself)
        if self.ad_showing():
            log.warning("refused tap at (%d,%d): an AD is on screen", x, y)
            return False
        # banner strip (right edge): a FRESH look before every tap there. Anything gold
        # sliding in = refuse, unless FreeBanner has just read the word FREE on it.
        sx1, sy1, sx2, sy2 = vision.BANNER_STRIP
        if sx1 - 0.02 <= x / W <= sx2 and sy1 - 0.02 <= y / H <= sy2 + 0.02:
            if not getattr(self, "free_ok", False):
                fresh = self.dev.screenshot_raw()
                if vision.banner_gold(fresh) >= self.cfg.get("banner_guard_gold", 0.20):
                    log.warning("refused tap at (%d,%d): reward banner area (not verified FREE)", x, y)
                    return False
        if not getattr(self, "x_ok", False) and \
                any(x1 <= x / W <= x2 and y1 <= y / H <= y2 for x1, y1, x2, y2 in self.ignore_taps):
            log.warning("refused tap at (%d,%d): inside a no-touch HUD region", x, y)
            return False
        if self._alerts is None:
            self._alerts = vision.find_alert_badges(img)
        margin = 0.025 * W
        for a in self._alerts:
            if ((x - a.cx) ** 2 + (y - a.cy) ** 2) ** 0.5 < max(a.w, a.h) / 2 + margin:
                log.warning("refused tap at (%d,%d): too close to a '!' badge", x, y)
                return False
        return True

    def ad_showing(self) -> bool:
        """Checks (at most every 2 s) whether Android shows an ad window in front."""
        now = time.time()
        if now - getattr(self, "_focus_t", 0) > 2:
            self._focus_t = now
            self._ad = "AdActivity" in self.dev.focus()
        return self._ad

    def handle_ad(self):
        """Full-screen ad in front: no taps at all; press Back every few seconds until the
        ad lets go (many only allow closing after a countdown)."""
        if getattr(self, "_ad_since", None) is None:
            self._ad_since = time.time()
            log.warning("FULL-SCREEN AD - pressing Back until it closes (no taps on the ad)")
        if not self.cfg.get("close_ads_with_back", True):
            time.sleep(3)
            return
        time.sleep(self.cfg.get("ad_back_every", 4))
        if self.dev.back_out_of_ad():
            log.info("  Back pressed (ad open %ds)", time.time() - self._ad_since)
        time.sleep(1.5)
        self._focus_t = 0                       # re-check the window right away
        if not self.ad_showing():
            log.info("  ad closed after %ds", time.time() - self._ad_since)
            self._ad_since = None

    def tap(self, x, y, times=1):
        if not self.safe_to_tap(x, y):
            return False
        callers, fr = [], sys._getframe(1)
        while fr is not None and len(callers) < 2:
            callers.append(fr.f_code.co_name)
            fr = fr.f_back
        log.debug("tap (%d,%d) x%d  from %s", x, y, times, " <- ".join(callers))
        if self.dry_run:
            log.debug("[dry-run] tap %d,%d x%d", x, y, times)
            return True
        with self.tap_lock:
            if times == 1:
                self.dev.tap(x, y)
            else:
                self.dev.tap_many([(x, y)] * times)
        time.sleep(self.cfg.get("tap_delay", 0.05))
        return True

    @staticmethod
    def wait(s):
        time.sleep(s)

    def drag(self, img, x, y_from, y_to):
        """Slow vertical drag (fractions of the screen). Returns the new screenshot and
        whether the view actually moved (False = already at that end of the scroll)."""
        H, W = img.shape[:2]
        s = self.cfg.get("scroll", {})
        if not self.dry_run:
            self.dev.swipe(x * W, y_from * H, x * W, y_to * H, s.get("swipe_ms", 450))
        self.wait(s.get("settle", 0.6))
        new = self.screenshot()
        return new, not vision.same_view(img, new, s.get("same_threshold", 6.0))

    # --- screen recognition (templates cropped from the live game) ---------
    def is_quit_dialog(self, img):
        return vision.find_template(img, ROOT / "templates" / "guard") is not None

    def is_item_popup(self, img):
        return vision.find_template(img, ROOT / "templates" / "popup") is not None

    def eatery_title(self, img):
        return vision.find_template(img, ROOT / "templates" / "eatery")

    def guard(self, img=None) -> bool:
        """Never quit the game: if the 'Quit Game' dialog is up, close it with its X.
        (The Android Back key opens that dialog, so the bot never presses Back.)"""
        img = self.screenshot() if img is None else img
        if not self.is_quit_dialog(img):
            return False
        log.warning("'Quit Game' dialog detected -> closing it with X")
        x = vision.find_close_button(img)
        if x:
            self.tap(x.cx, x.cy)
            self.wait(0.6)
        return True

    # --- shared popup helpers ---------------------------------------------
    def buy_until_grey(self, max_clicks, min_w=0.15, screen_ok=None, below_y=0):
        """Keep pressing the top-most green coin button until none is left (or limit hit).
        screen_ok(img) must confirm the expected popup is open before every tap."""
        bought = 0
        for _ in range(max_clicks):
            img = self.screenshot()
            if self.guard(img) or (screen_ok and not screen_ok(img)):
                break
            btns = [b for b in vision.find_green_buttons(img, min_w) if b.y > below_y]
            if not btns:
                break
            n = min(self.cfg.get("buy_burst", 3), max_clicks - bought)
            if not self.tap(btns[0].cx, btns[0].cy, times=n):
                break
            bought += n
            self.wait(self.cfg.get("buy_delay", 0.25))
        return bought

    def close_item_popup(self, ix, iy):
        """Item popups close by tapping the same item again. Only tap if still open,
        otherwise we would re-open it."""
        for _ in range(2):
            if not self.is_item_popup(self.screenshot()):
                return
            self.tap(ix, iy)
            self.wait(0.5)

    def close_stale_popup(self, img) -> bool:
        """An item popup left open (it opened late, or its 'Level' title is hidden under
        the coin bar). Recognised by its green COIN button outside Manage Eatery. Closed by
        tapping its item (right below the popup), or empty floor if that spot is off-limits
        - both close item popups."""
        if self.eatery_title(img) is not None:
            return False
        btns = vision.find_green_buttons(img)
        if not btns:
            return False
        H, W = img.shape[:2]
        b = btns[0]
        x, y = b.cx, b.y + b.h + 0.042 * H
        log.info("closing a leftover item popup")
        if not (y < H and self.tap(x, y)):
            fx, fy = self.cfg.get("floor_spot", [0.67, 0.5])
            if abs(fy * H - b.cy) < 0.2 * H:   # keep the floor tap away from the popup
                fy = 0.75 if b.cy < 0.5 * H else 0.3
            self.tap(fx * W, fy * H)
        self.wait(0.5)
        return True

    def eatery_button_visible(self, img) -> bool:
        H, W = img.shape[:2]
        if not hasattr(self, "_eat_tpl"):
            self._eat_tpl = cv2.imread(str(ROOT / "templates" / "eatery_button" / "eatery_button.png"))
        region = img[int(0.87 * H):, int(0.72 * W):]
        return cv2.matchTemplate(region, self._eat_tpl, cv2.TM_CCOEFF_NORMED).max() >= 0.8

    def level_complete(self, img) -> bool:
        """'Well Done!' screen at the end of a location -> press 'To the next location!'.
        Everything else in the bot is generic, so the same strategy continues there; the
        planner forgets the old dishes (new_location flag)."""
        hits = {d.kind: d for d in vision.match_templates(img, ROOT / "templates" / "levelup", 0.8)}
        if not hits:
            # the map button has a moving shine that lowers the template score on most
            # frames: a weaker match is confirmed by reading the text there
            for d in vision.match_templates(img, ROOT / "templates" / "levelup", 0.5):
                if not d.kind.startswith("next_location"):
                    continue
                crop = img[max(0, d.y - 15):d.y + d.h + 15, max(0, d.x - 30):d.x + d.w + 30]
                res, _ = ocr.engine()(crop)
                if any("nextlocation" in r[1].replace(" ", "").lower() for r in res or []):
                    hits[d.kind] = d
                    break
        for kind in ("next_location", "next_location_map"):   # 'Well Done!' and world map
            if kind in hits:
                b = hits[kind]
                log.info("LOCATION COMPLETE -> pressing 'To the next location!' (%s)", kind)
                if self.tap(b.cx, b.cy + (20 if kind == "next_location_map" else 0)):
                    self.new_location = True
                    time.sleep(self.cfg.get("location_load_wait", 5))
                return True
        if "ready_set_go" in hits:           # 'Grand Opening!' of the new location
            b = hits["ready_set_go"]
            log.info("GRAND OPENING -> pressing 'Ready! Set! Go!'")
            if self.tap(b.cx, b.cy):
                self.new_location = True
                time.sleep(3)
            return True
        if "tap_to_skip" in hits:            # reward animation: tap the middle, far from ads
            H, W = img.shape[:2]
            log.info("reward animation -> tap to skip")
            self.tap(W / 2, H / 2)
            time.sleep(2)
            return True
        if "well_done" in hits:              # button still animating in
            time.sleep(1)
            return True
        return False

    def clear_overlays(self, img) -> bool:
        """Leftover windows must be closed before the bot looks at the restaurant, or it
        mistakes their contents for dishes/coins (e.g. a Manage Eatery list left open)."""
        if self.guard(img):
            return True
        if self.level_complete(img):
            return True
        if self.eatery_title(img) is not None:
            log.info("Manage Eatery window is open - closing it")
            self.close_with_x()
            return True
        # any other popup with an orange X (Starter Pack / real-money offers, 'Rate Us!',
        # news ...): close it with the X and never press anything else on it
        x = vision.find_close_button(img)
        if x is not None:
            log.info("popup window -> closing it with its X")
            self.x_ok = True        # a verified X sits ON TOP of the HUD icons: allow it
            try:
                self.tap(x.cx, x.cy)
            finally:
                self.x_ok = False
            time.sleep(0.8)
            return True
        return self.close_stale_popup(img)

    def close_with_x(self):
        x = vision.find_close_button(self.screenshot())
        if x:
            self.x_ok = True
            try:
                self.tap(x.cx, x.cy)
            finally:
                self.x_ok = False
            self.wait(0.5)

    # --- main loop --------------------------------------------------------
    def run(self):
        log.info("connected to %s", self.dev.connect())
        log.info("tasks: %s", [type(t).__name__ for t in self.tasks])
        if not self.dry_run:
            self.start_banner_watch()
            log.info("FREE-banner watcher running in the background")
        interval = self.cfg.get("loop_interval", 0.5)
        n = 0
        start, first_bar, last_report = time.time(), None, 0.0
        while True:
            try:
                if getattr(self, "fatal", None) is not None:   # error in the banner watcher
                    raise self.fatal
                img = self.screenshot()
                if time.time() - last_report >= self.cfg.get("report_seconds", 60):
                    bar = vision.level_bar(img)
                    first_bar = bar if first_bar is None else first_bar
                    mins = (time.time() - start) / 60
                    rate = (bar - first_bar) / mins if mins > 0.5 else 0
                    log.info("LEVEL BAR %.1f%%  (+%.1f%% in %.0f min, %.2f%%/min)  bot memory %d MB",
                             bar * 100, (bar - first_bar) * 100, mins, rate * 100, _mem_mb())
                    last_report = time.time()
                if self.ad_showing():
                    self.handle_ad()
                    continue
                self._ad_since = None
                if self.clear_overlays(img):
                    continue
                if self.debug and n % 10 == 0:
                    self._save_debug(img, n)
                for t in self.tasks:
                    if t.due():
                        t.last = time.time()
                        if t.run(img):
                            img = self.screenshot()
                n += 1
                time.sleep(interval)
            except KeyboardInterrupt:
                log.info("stopped")
                return
            except Exception as e:  # ANY error: stop the bot (no retrying)
                msg = str(e)
                log.debug("error details", exc_info=True)       # full traceback -> frogbot.log
                if isinstance(e, MemoryError) or "allocation" in msg or "paging file" in msg:
                    log.error("STOPPED: the PC ran out of memory (bot used %d MB). "
                              "Close other programs and start bot.py again.", _mem_mb())
                else:
                    last = msg.strip().splitlines()[-1][:200] if msg.strip() else type(e).__name__
                    log.error("STOPPED because of an error: %s (details in frogbot.log)", last)
                raise SystemExit(1)

    def _save_debug(self, img, n):
        dets = detect_all(img, self.ignore)
        out = ROOT / "debug"
        out.mkdir(exist_ok=True)
        cv2.imwrite(str(out / f"frame_{n:05d}.png"), vision.annotate(img, dets, self.ignore))


def detect_all(img, ignore):
    dets = vision.find_new_items(img, ignore) + vision.find_upgrade_badges(img, ignore)
    dets += vision.find_coins(img, ignore)
    dets += vision.find_green_buttons(img, 0.12)
    c = vision.find_close_button(img)
    return dets + ([c] if c else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--image", help="run detectors on a saved screenshot and exit")
    args = ap.parse_args()
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if args.debug else logging.INFO)
    logfile = logging.FileHandler(ROOT / "frogbot.log", encoding="utf-8")  # every tap + reason
    logfile.setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG, handlers=[console, logfile],
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    if args.image:
        img = cv2.imread(args.image)
        ignore = [tuple(r) for r in cfg.get("ignore_regions", [])]
        dets = detect_all(img, ignore)
        for d in dets:
            print(f"{d.kind:8s} at ({d.cx},{d.cy}) size {d.w}x{d.h}")
        out = Path(args.image).with_name(Path(args.image).stem + "_annotated.png")
        cv2.imwrite(str(out), vision.annotate(img, dets, ignore))
        print("annotated ->", out)
        return

    Bot(cfg, args.dry_run, args.debug).run()


if __name__ == "__main__":
    main()
