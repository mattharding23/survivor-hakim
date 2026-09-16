# survivor-hakim

Personal weekly pick recommendation for the scohak.com "Survivor Football
League" -- the real ruleset, not a simplified one: two lives (out on the
2nd loss, or a $5 buy-back, or a drop to the $1 Consolation pool if that
2nd loss lands in Weeks 1-7), a Week-6 "Loser's Week" (pick one of your
Weeks 1-5 winners to *lose*), a Week-15 two-pick week, and a pre-season
Double-Dip team nominated for double use. Point differential is tiebreaker
#6 of 9 only.

Private, personal use only -- no site, no GitHub Pages.

## Local use

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # installs survivor-core (git submodule) editable

python run_hakim.py                              # auto-detects the current week
python run_hakim.py --week 9 --mode consolation
python run_hakim.py --double-dip CAR
python run_hakim.py --recommend-double-dip        # rank eligible Double-Dip nominees
```

The `core/` submodule is [survivor-core](https://github.com/mattharding23/survivor-core)
(shared schedule/odds/EV logic with `survivor-lms`). Clone with
`git clone --recurse-submodules`, or `git submodule update --init` after a
plain clone.

Life tracking auto-detects realized losses in your own logged picks against
actual game results (`data/2026/hakim/my_picks.csv` vs. the real win-prob
matrix) and shrinks your modeled life count accordingly, instead of always
assuming full health -- pass `--lives N` to override (e.g. after a buy-back).

## Data you maintain

- `data/2026/hakim/all_picks.csv` -- the real scohak export of every
  entrant's picks each week. Drop in the latest export; the opponent roster
  AND the real pick % both come straight from it, with 2-life elimination
  tracking built in. Falls back to hand-maintained `participants.csv` only
  if this file is missing.
- `data/2026/hakim/my_picks.csv` -- your own picks (`week,team,role`).
- `data/2026/hakim/double_dip.csv` -- your nominated Double-Dip team.
- `.odds_api_key` (git-ignored, one line) or `ODDS_API_KEY` env var -- The
  Odds API key. In CI this comes from a Doppler-synced Actions secret named
  `ODDS_API_KEY`.

## Weekly Action

`.github/workflows/weekly.yml` runs every Wednesday, gated on the real
America/New_York wall clock (see the workflow's comments), runs the
pipeline, writes the recommendation to the workflow's step summary, and
uploads the full `outputs/` directory (markdown, ranked CSV, chart PNG) as
a 90-day workflow artifact. Trigger a manual run any time with
`gh workflow run weekly.yml [-f week=N] [-f mode=consolation]` (bypasses
the hour gate).
