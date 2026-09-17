#!/usr/bin/env python
"""Generate site/data/*.json + markdown for the static site.

Reuses survivor_hakim.hakim_pipeline.run() exactly as run_hakim.py does --
no new computation, just serializes what it already returns and copies the
chart PNGs it already generates. Single-user (unlike survivor-lms, there's
no member loop): the whole site is "your" Hakim recommendation.

    python export_site_data.py --week 2
    python export_site_data.py                 # auto-detects the current week
    python export_site_data.py --mode consolation
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from survivor_core.schedule import load_season_schedule
from survivor_hakim.config import Config
from survivor_hakim.hakim_pipeline import run

SITE_DATA = Path(__file__).resolve().parent / "site" / "data"

# hakim_pipeline._write() already generates these PNGs into outputs/ on every
# run() call -- this just copies them into site/data/charts/ instead of
# discarding them.
CHART_LABELS = {
    "win_probability": "Win Probability",
    "pick_distribution": "Pick Distribution",
    "best_picks": "Best Picks",
    "ranked_table": "Ranked Table",
    "teams_remaining": "Teams Remaining (whole pool)",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _auto_week(cfg: Config) -> int:
    """First week with at least one game not yet played."""
    schedule = load_season_schedule(cfg.season)
    for wk in range(1, cfg.n_weeks + 1):
        wk_games = schedule[schedule["week"] == wk]
        if not wk_games.empty and not bool(wk_games["played"].all()):
            return wk
    return cfg.n_weeks


def _clean(obj):
    """Recursively swap NaN/NaT for None so json.dumps doesn't choke, and
    stringify dict keys (JSON object keys must be strings; ours are often
    int week numbers)."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, pd.Series):
        return _clean(obj.to_dict())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(data), indent=2, default=str))


def _matrix_to_dict(df: pd.DataFrame) -> dict:
    return {team: {str(wk): _clean(v) for wk, v in df.loc[team].items()} for team in df.index}


def _chart_key(filename: str) -> str | None:
    for key in CHART_LABELS:
        if key in filename:
            return key
    return None


def copy_charts(chart_paths: list[str], dest_dir: Path) -> list[dict]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for p in chart_paths:
        src = Path(p)
        if not src.exists():
            continue
        key = _chart_key(src.name)
        if key is None:
            continue
        shutil.copy2(src, dest_dir / src.name)
        out.append(dict(key=key, label=CHART_LABELS[key], file=src.name))
    order = {k: i for i, k in enumerate(CHART_LABELS)}
    out.sort(key=lambda c: order[c["key"]])
    return out


def export(cfg: Config, week: int, mode: str, result: dict) -> None:
    schedule = load_season_schedule(cfg.season).copy()
    schedule["gameday"] = schedule["gameday"].astype(str)
    _write_json(SITE_DATA / "schedule.json", dict(
        season=cfg.season, week=week, generated_at=_now(),
        games=schedule.to_dict("records"),
    ))

    _write_json(SITE_DATA / "winprob.json", dict(
        week=week, generated_at=_now(),
        teams=_matrix_to_dict(result["winprob_matrix"]),
    ))

    pct = result["public_pct"]
    _write_json(SITE_DATA / "public_pct.json", dict(
        week=week, generated_at=_now(),
        source=result["meta"]["consensus"].get("source"),
        pct={t: _clean(v) for t, v in pct.items()},
    ))

    ranked = result["ranked"]
    _write_json(SITE_DATA / "ranked.json", dict(
        week=week, generated_at=_now(),
        mode=mode, lives=result["lives"], lives_max=(1 if mode == "consolation" else 2),
        double_dip=result["double_dip"],
        realized_losses=result["meta"].get("realized_losses", 0),
        n_participants=result["meta"]["n_participants"], n_alive=result["meta"]["n_alive"],
        roster_source=result["meta"]["roster_source"],
        used_counts=result["used_counts"],
        teams=_clean(ranked.to_dict("records")) if len(ranked) else [],
        recommendation=_clean(result["recommendation"]),
        base_path=result["base_path"], base_exp_weeks=result["base_exp_weeks"],
        base_p_survive=result["base_p_survive"],
    ))

    (SITE_DATA / "recommendation.md").write_text(result["report"]["markdown"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--week", type=int, default=None,
                    help="NFL week (default: auto-detect the current week)")
    ap.add_argument("--mode", choices=["main", "consolation"], default="main")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    week = args.week or _auto_week(cfg)
    if not (1 <= week <= cfg.n_weeks):
        ap.error("--week must be 1..18")

    print(f"[export] Week {week} ({'explicit' if args.week else 'auto-detected'}), mode={args.mode}")

    result = run(week, mode=args.mode, force_refresh=args.refresh, cfg=cfg)
    export(cfg, week, args.mode, result)

    charts = copy_charts(result["report"].get("charts", []), SITE_DATA / "charts")
    _write_json(SITE_DATA / "charts" / "manifest.json", dict(
        week=week, generated_at=_now(), charts=charts,
    ))
    print(f"[export] wrote site data ({len(charts)} charts) -> {SITE_DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
