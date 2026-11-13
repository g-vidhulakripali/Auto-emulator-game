# Auto-emulator-game

A bot that plays **Frog's Kitchen** (`com.outloud.idlefrog`, idle restaurant game) on an Android
emulator by itself. It reads the screen over ADB, finds things with OpenCV colour/shape detection
and a few templates cropped from the live game, and taps.

## What it does

| Task | How |
|---|---|
| **Unlock NEW items** | Taps anything wearing a white "NEW" sticker and presses its green coin button so the new item starts working. |
| **Upgrade items** | Finds red/orange badges with a white up arrow on food and stations, opens the item, and presses the green buy button (in bursts of 3) until it turns grey. Closes the popup by tapping the item again. |
| **Tap the egg** | The white/teal frog egg that wanders around is tapped to burst it for coin-carrying frogs. |
| **Manage Eatery** | When the green up-arrow button (bottom right) shows its red "!", opens the menu, buys every affordable upgrade (top first), scrolls the list for more, and closes it with the orange X. |
| **Collect coins** | Taps the big gold coins lying on the dining tables. |
| **Scroll** | The restaurant is about two screens tall: the bot pans between the dining hall (top) and kitchen (bottom), working each view, and turns around at the ends. |

Finishing a level = these upgrades filling the orange progress bar at the bottom.

## Safety rules (built in)

- **Never taps a red/orange "!" badge.** Every tap is checked; one near a "!" is refused and logged.
- **Never taps the HUD:** coins/gems bar, calendar, settings, starter pack, ad TV, gem frog, event
  icons, discount, rocket, and the bottom menu (camera, shop, x2 ad boost, outfits). Only the
  Manage Eatery button is allowed. See `no_tap_regions` in `config.yaml`.
- **Only presses green buttons with a gold coin icon** (in-game coins). "Open" (Froggy Bank),
  "Confirm" and gem/real-money offers have no coin icon and are never pressed.
- **Never presses Android Back** — in this game it opens "Quit Game". If that dialog ever shows,
  the bot closes it with its X.
- Watch-ad offers sliding in from the side are never tapped.

## Setup (LDPlayer 9 on Windows)

1. In LDPlayer: **Settings → Other settings → ADB debugging → Open local connection → Save settings**,
   then restart the emulator.
2. Install Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Check `adb_path` in `config.yaml` points at your LDPlayer `adb.exe`.
4. Open Frog's Kitchen in the emulator (screen 900x1600; templates are cut at that size).

## Run

```bash
python bot.py
```

| Command | Purpose |
|---|---|
| `python bot.py --dry-run` | Detect and log, never tap (safe test). |
| `python bot.py --debug` | Also save annotated screenshots to `debug/`. |
| `python bot.py --image shot.png` | Run detectors on a saved screenshot, writes `shot_annotated.png`. |
| `python capture.py shot` | Save an emulator screenshot to `screenshots/`. |
| `python capture.py crop IMG X Y W H NAME` | Cut a template into `templates/tap/NAME.png`. |

Stop with **Ctrl+C**.

## Tuning

Everything is in `config.yaml`: how often each task runs, how many levels to buy per item,
burst size, scroll step, and the no-touch regions (fractions of the screen).

## Files

- `bot.py` main loop, tap safety guard, popup helpers
- `tasks.py` the individual jobs
- `vision.py` screen detection (badges, "!" alerts, buy buttons, close X, coins, NEW, templates)
- `device.py` ADB screenshots, taps and swipes
- `capture.py` screenshot and template tools
- `templates/` crops from the live game: `guard/` (Quit Game), `popup/` (item "Level" title),
  `eatery/` (Manage Eatery title), `egg/` (frog egg)

> Automating a game may be against its terms of service. Use at your own risk.
