#!/usr/bin/env python
"""Run the Hakim / scohak league optimiser for one week.

    python run_hakim.py                                # auto-detect the current week
    python run_hakim.py --week 1
    python run_hakim.py --week 9 --mode consolation
    python run_hakim.py --week 1 --double-dip CAR      # nominate your Double-Dip team

Inputs live in  data/2026/hakim/ :
    my_picks.csv       week,team,role   (role: win | loser | w15a | w15b) — hand-maintained
    double_dip.csv     one team abbr (missed the playoffs last year)      — hand-maintained
    all_picks.csv      real scohak export of every entrant's picks (preferred opponent
                       roster source; auto-parsed, including 2-life elimination tracking)
    participants.csv   participant,week,team[,role] — manual fallback, only used
                       when all_picks.csv isn't present
"""
from __future__ import annotations

import argparse
import sys

from survivor_core.schedule import load_season_schedule
from survivor_hakim.config import Config
from survivor_hakim.hakim_pipeline import run


def _auto_week(cfg: Config) -> int:
    """First week with at least one game not yet played."""
    schedule = load_season_schedule(cfg.season)
    for wk in range(1, cfg.n_weeks + 1):
        wk_games = schedule[schedule["week"] == wk]
        if not wk_games.empty and not bool(wk_games["played"].all()):
            return wk
    return cfg.n_weeks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--week", type=int, default=None,
                    help="NFL week (default: auto-detect the current week)")
    ap.add_argument("--mode", choices=["main", "consolation"], default="main")
    ap.add_argument("--lives", type=int, default=None,
                    help="override current life count (auto-detected from realized "
                         "results of your logged picks otherwise; set after a buy-back)")
    ap.add_argument("--double-dip", default=None, help="your Double-Dip team abbr")
    ap.add_argument("--recommend-double-dip", action="store_true",
                    help="rank eligible teams as Double-Dip nominees")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    week = args.week or _auto_week(cfg)
    if not (1 <= week <= cfg.n_weeks):
        ap.error("--week must be 1..18")
    res = run(week, mode=args.mode, double_dip=args.double_dip,
              recommend_dd=args.recommend_double_dip, force_refresh=args.refresh, cfg=cfg,
              lives=args.lives)
    print("\n" + res["report"]["markdown"])
    print(f"\nWritten to: {res['report']['dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
