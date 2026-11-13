"""The things the bot does each cycle. Each task: run(img) -> bool (True = it changed the screen)."""
import logging
import time

import vision

log = logging.getLogger("frogbot")


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
        items = vision.find_new_items(img, self.bot.ignore)
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
        bought = 0
        for _ in range(self.cfg.get("max_pages", 6)):  # the upgrade list scrolls too
            bought += self.bot.buy_until_grey(self.cfg.get("max_buys", 15), min_w=0.12,
                                              screen_ok=ok, below_y=title.y + title.h)
            # drag on the item text (left side) so we never press a buy button by accident
            _, moved = self.bot.drag(self.bot.screenshot(), 0.35, 0.70, 0.45)
            if not moved:
                break
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
    "manage_eatery": ManageEatery,
    "collect_coins": CollectCoins,
    "tap_templates": TapTemplates,
    "tap_egg": TapTemplates,   # same logic, its own folder (templates/egg)
    "scroll": Scroll,
}
