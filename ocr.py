"""Reads the numbers in an item popup with RapidOCR (offline, CPU).

Popup layout (relative to the 'Level' title):
    Level 162
    Ravioli
    * * *            star bar  +1
    (coin) 152.27e   (clock) 1.9s     <- profit per dish, cooking time
    [ (coin) 20.97e ]                 <- cost of the next level
"""
import gc
import re
import threading

_engine = None
AMOUNT = re.compile(r"^(\d+(?:\.\d+)?)([a-z]{0,2})$")
# in-game money: '12.34e' later on, plain '159' at the start of a new location
MONEY = re.compile(r"^\d{1,4}(?:\.\d{1,2})?[a-z]{0,2}$")
TIME = re.compile(r"^\d{1,3}(?:\.\d)?s$")


_lock = threading.Lock()


REBUILD_EVERY = 300   # OCR reads before the engine is rebuilt (frees memory it piled up)


def _make_engine():
    from rapidocr_onnxruntime import RapidOCR
    from rapidocr_onnxruntime.utils import infer_engine as ie
    if not getattr(ie.OrtInferSession, "_frogbot_patched", False):
        orig = ie.OrtInferSession._init_sess_opts

        def opts(config):
            o = orig(config)
            o.enable_mem_pattern = False      # no per-input-size memory plans (they pile up)
            return o

        ie.OrtInferSession._init_sess_opts = staticmethod(opts)
        ie.OrtInferSession._frogbot_patched = True
    return RapidOCR()


class _Locked:
    """The OCR engine is shared by the main loop and the banner watcher thread.
    It is rebuilt every REBUILD_EVERY reads so its memory cannot pile up."""

    def __init__(self):
        self.eng, self.calls = _make_engine(), 0

    def __call__(self, img):
        with _lock:
            self.calls += 1
            if self.calls >= REBUILD_EVERY:
                self.eng, self.calls = None, 0
                gc.collect()
                self.eng = _make_engine()
            return self.eng(img)                 # errors propagate: the bot stops


def engine():
    global _engine
    with _lock:
        if _engine is None:
            _engine = _Locked()
    return _engine


def parse_amount(text):
    """'152.27e' -> 152.27 * 1000**5. Suffix letters step x1000: a, b, ... z, aa, ab ..."""
    m = AMOUNT.match(text.strip().lower().replace(",", ""))
    if not m:
        return None
    num, suf = float(m.group(1)), m.group(2)
    if len(suf) == 0:
        rank = 0
    elif len(suf) == 1:
        rank = ord(suf) - 96
    else:
        rank = 26 + (ord(suf[0]) - 97) * 26 + (ord(suf[1]) - 96)
    return num * 1000.0 ** rank


def read_amount(img, region):
    """OCR one amount (e.g. the coin balance '409.76e') from a normalised region."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = region
    res, _ = engine()(img[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)])
    for r in res or []:
        v = parse_amount(r[1].replace(" ", ""))
        if v:
            return v
    return None


def read_banner(img, strip):
    """Read the timed reward banner in `strip` (normalised x1, y1, x2, y2).
    Returns None if there is no banner (no '+amount' text), else
    {'free': True only if the word FREE is printed on it, 'tap': (x, y) of its amount}.
    Video-ad banners carry no FREE word -> free=False -> never tapped."""
    H, W = img.shape[:2]
    x1, y1 = int(strip[0] * W), int(strip[1] * H)
    crop = img[y1:int(strip[3] * H), x1:int(strip[2] * W)]
    res, _ = engine()(crop)
    amount, free, texts = None, False, []
    for r in res or []:
        t = r[1].replace(" ", "")
        texts.append(t)
        if "FREE" in t.upper():
            free = True
        if t.startswith("+") and parse_amount(t[1:]):
            cx = (r[0][0][0] + r[0][2][0]) / 2 + x1
            cy = (r[0][0][1] + r[0][2][1]) / 2 + y1
            amount = (cx, cy)
    if amount is None:
        return None
    if not free:  # second look at 2x zoom: the small 'FREE' word is easy to miss
        import cv2
        big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        res2, _ = engine()(big)
        texts += [r[1] for r in res2 or []]
        free = any("FREE" in r[1].upper().replace(" ", "") for r in res2 or [])
    return {"free": free, "tap": amount, "texts": texts}


def read_popup(img, title):
    """OCR the popup above `title` (a vision.Det of the 'Level' template).
    Returns dict(name, level, profit, seconds, cost, text) or None."""
    H, W = img.shape[:2]
    x0, x1 = max(0, title.cx - 180), min(W, title.cx + 210)
    y0, y1 = max(0, title.y - 15), min(H, title.y + 320)
    res, _ = engine()(img[y0:y1, x0:x1])
    if not res:
        return None
    # (text, x centre, y centre relative to the 'Level' title top)
    toks = []
    for r in res:
        t = re.sub(r"^[^0-9A-Za-z]+", "", r[1].strip()).replace(" ", "")
        toks.append((t, (r[0][0][0] + r[0][2][0]) / 2 + x0,
                     (r[0][0][1] + r[0][2][1]) / 2 + y0 - title.y))

    def band(lo, hi):
        return [k for k in toks if lo <= k[2] <= hi]

    level = name = None
    for t, _, _ in band(-15, 35):
        m = re.match(r"^level(\d+)$", t.lower())
        if m:
            level = int(m.group(1))
    for t, _, _ in band(25, 75):
        if re.fullmatch(r"[A-Za-z']{3,}", t):
            name = t
            break

    # profit row: money on the left, 'N.Ns' on the right. Money MUST be '12.34e'-shaped.
    row = band(160, 235)
    money_row = [k for k in row if MONEY.match(k[0].lower())]
    times = [k for k in row if TIME.match(k[0].lower())]
    cost_toks = [k for k in band(245, 325) if MONEY.match(k[0].lower())]
    if not money_row or not times or not cost_toks:
        return None
    profit_t = min(money_row, key=lambda k: k[1])
    time_t = max(times, key=lambda k: k[1])
    if time_t[1] <= profit_t[1]:
        return None
    cost_t = min(cost_toks, key=lambda k: abs(k[1] - (title.cx + 30)))
    profit, cost = parse_amount(profit_t[0]), parse_amount(cost_t[0])
    seconds = float(time_t[0][:-1])
    if not profit or not cost or seconds <= 0:
        return None
    if not (1e-3 <= cost / profit <= 1e6):   # misread (e.g. a dropped suffix): ignore
        return None
    return {"name": name, "level": level, "profit": profit, "seconds": seconds, "cost": cost,
            "text": f"{profit_t[0]}/{time_t[0]} for {cost_t[0]}"}
