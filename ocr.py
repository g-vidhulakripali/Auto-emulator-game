"""Reads the numbers in an item popup with RapidOCR (offline, CPU).

Popup layout (relative to the 'Level' title):
    Level 162
    Ravioli
    * * *            star bar  +1
    (coin) 152.27e   (clock) 1.9s     <- profit per dish, cooking time
    [ (coin) 20.97e ]                 <- cost of the next level
"""
import re

_engine = None
AMOUNT = re.compile(r"^(\d+(?:\.\d+)?)([a-z]{0,2})$")
MONEY = re.compile(r"^\d{1,3}(?:\.\d{1,2})?[a-z]{1,2}$")   # in-game money always has a suffix
TIME = re.compile(r"^\d{1,3}(?:\.\d)?s$")


def engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
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
    return {"name": name, "level": level, "profit": profit, "seconds": seconds, "cost": cost,
            "text": f"{profit_t[0]}/{time_t[0]} for {cost_t[0]}"}
