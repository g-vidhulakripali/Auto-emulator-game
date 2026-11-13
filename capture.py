"""Helpers for tuning the bot.

  python capture.py shot                         save an emulator screenshot to screenshots/
  python capture.py crop IMG X Y W H NAME        cut a template into templates/tap/NAME.png
"""
import sys
import time
from pathlib import Path

import cv2
import yaml

from device import Device

ROOT = Path(__file__).parent


def shot():
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    dev = Device(cfg["adb_path"], cfg["serials"])
    dev.connect()
    img = dev.screenshot()
    out = ROOT / "screenshots"
    out.mkdir(exist_ok=True)
    path = out / f"shot_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(path), img)
    print(f"saved {path}  ({img.shape[1]}x{img.shape[0]})")


def crop(img_path, x, y, w, h, name):
    img = cv2.imread(img_path)
    out = ROOT / "templates" / "tap"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    cv2.imwrite(str(path), img[int(y):int(y) + int(h), int(x):int(x) + int(w)])
    print("saved", path)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "shot":
        shot()
    elif len(sys.argv) == 8 and sys.argv[1] == "crop":
        crop(*sys.argv[2:])
    else:
        print(__doc__)
