"""Thin ADB wrapper: screenshots, taps and key presses for an Android emulator."""
import subprocess

import cv2
import numpy as np


class Device:
    def __init__(self, adb_path: str, serials: list[str]):
        self.adb = adb_path
        self.candidates = serials
        self.serial = None

    def _adb(self, *args, timeout=20, text=False):
        return subprocess.run([self.adb, *args], capture_output=True, timeout=timeout, text=text)

    def _run(self, *args, timeout=20):
        return self._adb("-s", self.serial, *args, timeout=timeout)

    def connect(self) -> str:
        """Connect to the first reachable serial from the config and return it."""
        for serial in self.candidates:
            if ":" in serial:
                self._adb("connect", serial, text=True)
            out = self._adb("devices", text=True).stdout
            for line in out.splitlines():
                parts = line.split("\t")
                if len(parts) == 2 and parts[0] == serial and parts[1] == "device":
                    self.serial = serial
                    return serial
        raise ConnectionError(
            "No emulator reachable over ADB. In LDPlayer: Settings -> Other settings -> "
            "ADB debugging -> 'Open local connection', then restart the emulator."
        )

    def screenshot(self) -> np.ndarray:
        r = self._run("exec-out", "screencap", "-p")
        img = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"screencap failed: {r.stderr[:200]!r}")
        return img

    def screenshot_raw(self) -> np.ndarray:
        """Faster: raw RGBA frame (no PNG encoding on the device)."""
        data = self._run("exec-out", "screencap").stdout
        if len(data) < 16:
            return self.screenshot()
        w, h = (int(v) for v in np.frombuffer(data[:8], np.uint32))
        hdr = len(data) - w * h * 4
        if hdr not in (12, 16):
            return self.screenshot()
        rgba = np.frombuffer(data[hdr:], np.uint8).reshape(h, w, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    def tap(self, x: float, y: float):
        self._run("shell", "input", "tap", str(int(x)), str(int(y)))

    def tap_many(self, points):
        """Several taps in one ADB round trip."""
        cmd = "; ".join(f"input tap {int(x)} {int(y)}" for x, y in points)
        self._run("shell", cmd)

    def focus(self) -> str:
        """Name of the window in front (e.g. '...UnityPlayerActivity' or '...AdActivity')."""
        r = self._run("shell", "dumpsys window | grep mCurrentFocus")
        return r.stdout.decode(errors="ignore")

    def back_out_of_ad(self) -> bool:
        """Android Back, ONLY while an ad window is in front (checked right here), to close
        the full-screen ads the game shows between locations. Never touches the ad itself.
        In the game itself Back opens 'Quit Game', so it is never sent there."""
        if "AdActivity" not in self.focus():
            return False
        self._run("shell", "input", "keyevent", "4")
        return True

    # NOTE: no plain Back key - in Frog's Kitchen it opens the 'Quit Game' dialog.

    def swipe(self, x1, y1, x2, y2, ms=450):
        self._run("shell", "input", "swipe", *(str(int(v)) for v in (x1, y1, x2, y2, ms)))
