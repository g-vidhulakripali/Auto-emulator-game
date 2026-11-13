"""Frog's Kitchen auto-player.

  python bot.py                 run the bot (Ctrl+C to stop)
  python bot.py --dry-run       detect and log, but never tap
  python bot.py --debug         also save annotated screenshots to debug/
  python bot.py --image X.png   offline: run detectors on a saved screenshot
"""
import argparse
import logging
import time
from pathlib import Path

import cv2
import yaml

import vision
from tasks import TASKS

ROOT = Path(__file__).parent
log = logging.getLogger("frogbot")


class Bot:
    def __init__(self, cfg, dry_run=False, debug=False):
        from device import Device
        self.cfg = cfg
        self.dry_run, self.debug = dry_run, debug
        self.ignore = [tuple(r) for r in cfg.get("ignore_regions", [])]
        # taps are blocked in the HUD except where the bot needs a specific menu button
        self.ignore_taps = [tuple(r) for r in cfg.get("no_tap_regions", [])]
        self.last_img, self._alerts = None, None
        self.dev = Device(cfg["adb_path"], cfg["serials"])
        self.tasks = [TASKS[name](self, tcfg) for name, tcfg in cfg["tasks"].items()
                      if tcfg.get("enabled", True)]
        self.shape = None

    # --- primitives -------------------------------------------------------
    def screenshot(self):
        img = self.dev.screenshot_raw()
        self.shape = img.shape
        self.last_img, self._alerts = img, None
        return img

    def safe_to_tap(self, x, y) -> bool:
        """RULE: never tap a red/orange '!' badge or the HUD (ignore_regions)."""
        img = self.last_img
        if img is None:
            return False
        H, W = img.shape[:2]
        if any(x1 <= x / W <= x2 and y1 <= y / H <= y2 for x1, y1, x2, y2 in self.ignore_taps):
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

    def tap(self, x, y, times=1):
        if not self.safe_to_tap(x, y):
            return False
        if self.dry_run:
            log.debug("[dry-run] tap %d,%d x%d", x, y, times)
            return True
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
        """An item popup left open (e.g. it opened late). Its item sits straight below
        the 'Level' title, so tapping there toggles it closed."""
        t = vision.find_template(img, ROOT / "templates" / "popup")
        if t is None:
            return False
        H = img.shape[0]
        x, y = t.cx + 0.025 * img.shape[1], t.cy + self.cfg.get("popup_item_dy", 0.245) * H
        log.info("closing a leftover item popup")
        if y < H and self.tap(x, y):
            self.wait(0.5)
        return True

    def close_with_x(self):
        x = vision.find_close_button(self.screenshot())
        if x:
            self.tap(x.cx, x.cy)
            self.wait(0.5)

    # --- main loop --------------------------------------------------------
    def run(self):
        log.info("connected to %s", self.dev.connect())
        log.info("tasks: %s", [type(t).__name__ for t in self.tasks])
        interval = self.cfg.get("loop_interval", 0.5)
        n = 0
        while True:
            try:
                img = self.screenshot()
                if self.guard(img) or self.close_stale_popup(img):
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
            except Exception as e:  # emulator hiccup: reconnect and keep going
                log.warning("error: %s -- reconnecting in 3s", e)
                time.sleep(3)
                try:
                    self.dev.connect()
                except Exception as e2:
                    log.warning("reconnect failed: %s", e2)

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
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
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
