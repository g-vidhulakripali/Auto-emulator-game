"""Screen understanding by colour + shape heuristics (no game-specific templates needed)."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Det:
    kind: str
    x: int  # bbox left
    y: int  # bbox top
    w: int
    h: int

    @property
    def cx(self):
        return self.x + self.w // 2

    @property
    def cy(self):
        return self.y + self.h // 2


def _hsv_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return mask


def _blobs(mask, min_d, max_d, min_circ=0.0, aspect=(0.0, 99.0), min_fill=0.0):
    """External contours filtered by size (bbox width in px), circularity, aspect and fill."""
    out = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (min_d <= w <= max_d):
            continue
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True) or 1
        circ = 4 * np.pi * area / (perim * perim)
        ar = w / max(h, 1)
        fill = area / max(w * h, 1)
        if circ >= min_circ and aspect[0] <= ar <= aspect[1] and fill >= min_fill:
            out.append((x, y, w, h))
    return out


def in_regions(det: Det, shape, regions) -> bool:
    H, W = shape[:2]
    nx, ny = det.cx / W, det.cy / H
    return any(x1 <= nx <= x2 and y1 <= ny <= y2 for x1, y1, x2, y2 in regions)


def find_upgrade_badges(img, ignore_regions=()):
    """Red circle with a white UP ARROW = item can be upgraded.
    Rejects red '!' notification badges by the shape of the white glyph."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # red badges (food items) and orange badges (stations, e.g. drink dispenser)
    red = _hsv_mask(hsv, [((0, 140, 140), (8, 255, 255)), ((170, 140, 140), (180, 255, 255)),
                          ((15, 200, 200), (24, 255, 255))])
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    white = _hsv_mask(hsv, [((0, 0, 200), (180, 70, 255))])
    dets = []
    for x, y, w, h in _blobs(red, 0.02 * W, 0.08 * W, min_circ=0.55, aspect=(0.7, 1.4)):
        d = Det("upgrade", x, y, w, h)
        if in_regions(d, img.shape, ignore_regions):
            continue
        if _is_up_arrow(white[y:y + h, x:x + w]):
            dets.append(d)
    # merge in arrow-first detections (badge colour merged with an orange dish)
    for e in find_upgrade_badges_by_arrow(img, ignore_regions):
        if all(abs(e.cx - d.cx) > d.w or abs(e.cy - d.cy) > d.h for d in dets):
            dets.append(e)
    return dets


def find_upgrade_badges_by_arrow(img, ignore_regions=()):
    """Second detector that starts from the WHITE ARROW and checks for a red/orange ring
    around it. Catches badges whose colour merges with an orange dish behind them."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = _hsv_mask(hsv, [((0, 0, 200), (180, 70, 255))])
    ring_col = _hsv_mask(hsv, [((0, 140, 140), (8, 255, 255)), ((170, 140, 140), (180, 255, 255)),
                               ((12, 170, 190), (24, 255, 255))])
    n, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    dets = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (0.011 * W <= h <= 0.035 * W and 0.5 * h <= w <= 1.3 * h):
            continue
        glyph = (white[y:y + h, x:x + w] > 0)
        widths = glyph.sum(axis=1)
        head = widths[: max(1, h // 2)].max()
        stem = np.median(widths[int(h * 0.65):]) or 1
        # arrow: wide head + solid stem (~40% of head). A sparkle star has a needle-thin tail.
        if not (0.33 * head <= stem <= 0.62 * head) or (widths == 0).any():
            continue
        cx, cy = x + w / 2, y + h / 2

        def ring_share(r):
            pts = [(int(cx + r * np.cos(a)), int(cy + r * np.sin(a))) for a in angles]
            return sum(1 for px, py in pts if 0 <= px < W and 0 <= py < H and ring_col[py, px]) / len(pts)

        if ring_share(0.68 * h) < 0.8:   # badge body all around the arrow
            continue
        if ring_share(1.35 * h) > 0.85:  # but not buried in orange (e.g. coin pile in an ad banner)
            continue
        s = int(2 * h)
        d = Det("upgrade", int(cx - s / 2), int(cy - s / 2), s, s)
        if not in_regions(d, img.shape, ignore_regions):
            dets.append(d)
    return dets


def find_alert_badges(img):
    """Red/orange '!' notification badges anywhere on screen (menus, offers, events).
    The bot must NEVER tap these; Bot.tap() refuses any tap that lands near one."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red = _hsv_mask(hsv, [((0, 140, 140), (8, 255, 255)), ((170, 140, 140), (180, 255, 255)),
                          ((15, 200, 200), (24, 255, 255))])
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    white = _hsv_mask(hsv, [((0, 0, 200), (180, 70, 255))])
    dets = []
    for x, y, w, h in _blobs(red, 0.02 * W, 0.08 * W, min_circ=0.55, aspect=(0.7, 1.4)):
        crop = white[y:y + h, x:x + w]
        inner = np.zeros_like(crop)
        cv2.circle(inner, (w // 2, h // 2), int(min(w, h) * 0.38), 255, -1)
        glyph = cv2.bitwise_and(crop, inner)
        ys, xs = np.nonzero(glyph)
        if len(ys) < 0.03 * w * h or _is_up_arrow(crop):
            continue
        # '!' = tall, narrow white glyph
        if (ys.max() - ys.min() + 1) >= 1.6 * (xs.max() - xs.min() + 1):
            dets.append(Det("alert", x, y, w, h))
    return dets


def _is_up_arrow(white_crop) -> bool:
    h, w = white_crop.shape
    # only look inside the circle, away from any white outline ring
    inner = np.zeros_like(white_crop)
    cv2.circle(inner, (w // 2, h // 2), int(min(w, h) * 0.38), 255, -1)
    glyph = cv2.bitwise_and(white_crop, inner)
    ys, xs = np.nonzero(glyph)
    if len(ys) < 0.04 * w * h:
        return False
    top, bot = ys.min(), ys.max()
    gh = bot - top + 1
    if gh < 0.3 * h:
        return False
    widths = [(glyph[r] > 0).sum() for r in range(top, bot + 1)]
    n = len(widths)
    head = max(widths[: max(1, n // 2)])          # arrow head: wide triangle at the top
    stem = np.median([v for v in widths[n // 2:] if v > 0] or [1])  # narrow stem below
    # '!' has a gap between bar and dot; an arrow is one connected glyph
    gaps = sum(1 for v in widths if v == 0)
    return head >= 1.6 * stem and gaps <= 1


def find_new_items(img, ignore_regions=()):
    """'NEW' sticker on a freshly available item: white disc with teal/green 'NEW'
    letters running across its middle."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = _hsv_mask(hsv, [((0, 0, 225), (180, 50, 255))])
    teal = _hsv_mask(hsv, [((60, 90, 110), (100, 255, 255))])
    # the white disc alone (letters are just holes inside it); never OR in teal, or the
    # disc merges with the teal bottle it is printed on
    disc = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dets = []
    for x, y, w, h in _blobs(disc, 0.035 * W, 0.16 * W, min_circ=0.6, aspect=(0.75, 1.35)):
        d = Det("new", x, y, w, h)
        if in_regions(d, img.shape, ignore_regions):
            continue
        t = teal[y:y + h, x:x + w]
        inner = np.zeros_like(t)
        cv2.circle(inner, (w // 2, h // 2), int(min(w, h) * 0.42), 255, -1)
        t = cv2.bitwise_and(t, inner)
        ys, xs = np.nonzero(t)
        if len(ys) < 0.06 * w * h or len(ys) > 0.45 * w * h:
            continue
        middle = ((ys > 0.3 * h) & (ys < 0.72 * h)).mean()  # letters sit in a middle band
        span = (xs.max() - xs.min() + 1) / w                 # ...and run across the disc
        if middle < 0.7 or span < 0.5:
            continue
        # text check: at least 3 separate letters (N, E, W) side by side on one line
        k, _, st, cen = cv2.connectedComponentsWithStats(t, connectivity=8)
        letters = [cen[j] for j in range(1, k) if st[j, cv2.CC_STAT_AREA] >= max(4, 0.004 * w * h)]
        if len(letters) >= 3 and np.std([c[1] for c in letters]) < 0.12 * h:
            dets.append(d)
    return dets


def find_green_buttons(img, min_w=0.15, need_coin=True):
    """Green 'buy' buttons (item popup or Manage Eatery list): wide, solid, bright green.
    Unaffordable buttons are greyed out, so they are not matched. Sorted top to bottom.

    SAFETY: with need_coin (default) only buttons showing a gold coin icon count, i.e.
    things paid with in-game coins. 'Confirm' (Quit Game), 'Open' (Froggy Bank) and
    gem/real-money offers have no coin icon and are never returned."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = _hsv_mask(hsv, [((38, 130, 150), (75, 255, 255))])
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    gold = _hsv_mask(hsv, [((12, 150, 180), (24, 255, 255))])
    out = []
    for b in _blobs(green, min_w * W, 0.95 * W, aspect=(1.6, 6.0), min_fill=0.7):
        x, y, w, h = b
        if need_coin and (gold[y:y + h, x:x + w] > 0).mean() < 0.02:
            continue
        out.append(Det("buy", *b))
    return sorted(out, key=lambda d: d.y)


def red_fraction(img, region):
    """Share of notification-red pixels inside a normalised region (x1, y1, x2, y2).
    Used to see whether a menu button carries its red '!' badge."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = region
    hsv = cv2.cvtColor(img[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)], cv2.COLOR_BGR2HSV)
    red = _hsv_mask(hsv, [((0, 140, 140), (8, 255, 255)), ((170, 140, 140), (180, 255, 255))])
    return float((red > 0).mean())


def find_template(img, folder, threshold=0.8):
    """First match of any template in folder, or None (used to recognise screens)."""
    hits = match_templates(img, folder, threshold)
    return hits[0] if hits else None


def find_close_button(img):
    """Orange circle with a white X in the top-right of a popup (e.g. Manage Eatery)."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    orange = _hsv_mask(hsv, [((5, 110, 190), (20, 255, 255))])
    orange = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    white = _hsv_mask(hsv, [((0, 0, 210), (180, 60, 255))])
    best = None
    for x, y, w, h in _blobs(orange, 0.04 * W, 0.13 * W, min_circ=0.6, aspect=(0.75, 1.33)):
        d = Det("close", x, y, w, h)
        if d.cx < 0.6 * W or d.cy > 0.7 * H:
            continue
        ratio = (white[y:y + h, x:x + w] > 0).mean()
        if 0.08 <= ratio <= 0.5 and (best is None or d.cy < best.cy):
            best = d
    return best


def find_coins(img, ignore_regions=(), min_d=0.032):
    """Big gold coins lying on tables, waiting to be tapped for money.
    min_d (fraction of width) skips the small coin icons in the '187e' income labels."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gold = _hsv_mask(hsv, [((12, 170, 190), (25, 255, 255))])  # rim ~17, face ~23
    gold = cv2.morphologyEx(gold, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    white = _hsv_mask(hsv, [((0, 0, 200), (180, 70, 255))])
    rim = _hsv_mask(hsv, [((12, 215, 190), (20, 255, 255))])
    dets = []
    for x, y, w, h in _blobs(gold, min_d * W, 0.07 * W, min_circ=0.6, aspect=(0.8, 1.25)):
        d = Det("coin", x, y, w, h)
        if in_regions(d, img.shape, ignore_regions):
            continue
        if (white[y:y + h, x:x + w] > 0).mean() > 0.12:  # white glyph inside = badge, not coin
            continue
        if (rim[y:y + h, x:x + w] > 0).mean() < 0.08:  # coins have a darker orange rim;
            continue                                    # yellow frog bellies do not
        dets.append(d)
    return dets


def match_templates(img, folder, threshold=0.85, ignore_regions=()):
    """Optional: tap anything matching PNG templates you cropped with capture.py."""
    dets = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for p in sorted(Path(folder).glob("*.png")):
        tpl = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if tpl is None or tpl.shape[0] > gray.shape[0] or tpl.shape[1] > gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        th, tw = tpl.shape
        ys, xs = np.where(res >= threshold)
        for x, y in zip(xs, ys):
            d = Det(p.stem, int(x), int(y), tw, th)
            if in_regions(d, img.shape, ignore_regions):
                continue
            if all(abs(d.cx - o.cx) > tw // 2 or abs(d.cy - o.cy) > th // 2 for o in dets):
                dets.append(d)
    return dets


def same_view(a, b, threshold=6.0) -> bool:
    """True if two screenshots show the same scroll position (only frogs walking around).
    Compares the play area (no HUD) downscaled, so small animations barely count."""
    H = a.shape[0]
    band = slice(int(0.12 * H), int(0.85 * H))

    def prep(img):
        g = cv2.cvtColor(img[band], cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (64, 96), interpolation=cv2.INTER_AREA).astype(np.int16)

    if a.shape != b.shape:
        return False
    return float(np.abs(prep(a) - prep(b)).mean()) < threshold


COLORS = {"new": (255, 0, 255), "upgrade": (0, 0, 255), "buy": (0, 200, 0), "coin": (0, 200, 255), "close": (255, 128, 0)}


def annotate(img, dets, ignore_regions=()):
    out = img.copy()
    H, W = out.shape[:2]
    for x1, y1, x2, y2 in ignore_regions:
        cv2.rectangle(out, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)), (128, 128, 128), 1)
    for d in dets:
        c = COLORS.get(d.kind, (255, 0, 255))
        cv2.rectangle(out, (d.x, d.y), (d.x + d.w, d.y + d.h), c, 2)
        cv2.putText(out, d.kind, (d.x, max(10, d.y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    return out
