"""End-to-end run for the Hakim / scohak league (2-life main pool or 1-life
consolation), reusing the base data pipeline + charts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from survivor_core import survivorgrid as sg_mod
from survivor_core import viz
from .config import Config
from survivor_core.consensus import expected_duplication, fetch_public_pick_pct
from survivor_core.elo import EloModel
from .hakim import (HakimConfig, LOSERS_WEEK, TWO_PICK_WEEK, beam_search_path,
                    expected_margin_matrix, load_hakim_inputs, max_uses,
                    pick_pct_from_roster, realized_life_losses, recommend_double_dip,
                    to_arrays, used_counts_from_picks, weeks_1_5_win_picks)
from survivor_core.schedule import load_history, load_season_schedule, playoff_teams
from survivor_core.teams import ABBRS, display
from survivor_core.winprob import build_winprob_matrix


def _fmt_plan(path: dict, w0: int, n: int = 6) -> str:
    out = []
    for k in sorted(path):
        if k < w0:
            continue
        v = path[k]
        if isinstance(v, tuple) and v and v[0] == "LOSE":
            out.append(f"W{k}:LOSE {v[1]}")
        elif isinstance(v, tuple):
            out.append(f"W{k}:{v[0]}+{v[1]}")
        else:
            out.append(f"W{k}:{v}")
        if len(out) >= n:
            break
    return "  ".join(out)


def run(current_week: int, mode: str = "main", double_dip: str | None = None,
        recommend_dd: bool = False, force_refresh: bool = False,
        cfg: Config | None = None, lives: int | None = None) -> dict:
    cfg = cfg or Config()
    cfg.current_week = current_week
    hcfg = HakimConfig(mode=mode, double_dip=double_dip, lives=lives)

    schedule = load_season_schedule(cfg.season, force=force_refresh)
    history = load_history(cfg.season, cfg.elo.lookback_seasons, force=force_refresh)
    elo = EloModel(cfg.elo).fit(history)
    if history[history["season"] == cfg.season].empty:
        elo.preseason_regress()

    sg = sg_mod.load(force=force_refresh)
    if not sg.ok:
        print(f"[survivorgrid] unavailable: {sg.note}")
    wp, src = build_winprob_matrix(schedule, elo, cfg, sg=sg)
    exp_margin = expected_margin_matrix(schedule, wp, cfg, sg=sg)
    arrays = to_arrays(wp, exp_margin, cfg.n_weeks)

    inp = load_hakim_inputs(cfg, hcfg)
    my_picks, dd = inp["my_picks"], inp["double_dip"]
    hcfg.double_dip = dd
    used0 = used_counts_from_picks(my_picks)
    roster = inp["roster"]

    # Realize actual outcomes of already-played picks against the (mostly decided)
    # win-prob matrix, and shrink the life count accordingly -- lives_for_mode()
    # otherwise always assumes you're walking in at full health.
    realized_losses = 0
    if hcfg.lives is None:
        realized_losses = realized_life_losses(my_picks, wp, current_week)
        if realized_losses:
            base_lives = hcfg.lives_for_mode()
            hcfg.lives = max(base_lives - realized_losses, 0)
            print(f"[hakim] {realized_losses} of your logged pick(s) before Week {current_week} "
                 f"lost -- modelling {hcfg.lives} life/lives remaining (was {base_lives}). "
                 f"Pass --lives to override.")
            if hcfg.lives == 0 and mode == "main":
                print(f"[hakim] You're out of the Main Pool as of Week {current_week} unless you "
                     f"bought back or dropped to Consolation -- rerun with --mode consolation "
                     f"(or --lives N after a buy-back).")

    # consensus
    avail_week = pd.Series({t: (current_week in wp.columns and pd.notna(wp.loc[t, current_week])
                                and wp.loc[t, current_week] > 0
                                and used0.get(t, 0) < max_uses(t, dd)) for t in ABBRS})
    real_pct = pick_pct_from_roster(roster, current_week)
    if real_pct is not None:
        public_pct, cmeta = real_pct, {"source": "all_picks.csv", "n_entries": len(roster.participants)}
    else:
        public_pct, cmeta = fetch_public_pick_pct(
            current_week, wp_week=wp[current_week] if current_week in wp.columns else None, sg=sg)
    crowd = expected_duplication(current_week, wp, public_pct, roster, avail_week)

    # Double-Dip nomination advice (pre-season decision)
    dd_reco = None
    if recommend_dd:
        prev_po = playoff_teams(cfg.season - 1, force=force_refresh)
        dd_reco = recommend_double_dip(wp, exp_margin, cfg, hcfg, used0, my_picks,
                                       public_pct=public_pct, prev_playoff_teams=prev_po,
                                       arrays=arrays)
        dd_reco.attrs["prev_playoff_teams"] = sorted(prev_po)

    # unconditional optimal season plan under the Hakim life model
    base = beam_search_path(wp, exp_margin, cfg, hcfg, current_week, used0, my_picks, arrays=arrays)

    L = hcfg.lives_for_mode()
    cbw = hcfg.cand_beam_width

    rows = []
    if current_week == LOSERS_WEEK:
        pool = list(dict.fromkeys(weeks_1_5_win_picks(my_picks)))
        for t in pool:
            pw = float(wp.loc[t, current_week]) if pd.notna(wp.loc[t, current_week]) else np.nan
            r = beam_search_path(wp, exp_margin, cfg, hcfg, current_week, used0, my_picks,
                                 forced_first=("LOSE", t), beam_width=cbw, arrays=arrays)
            rows.append(dict(team=t, team_name=display(t), win_prob=pw,
                             p_lose=1 - pw if pd.notna(pw) else np.nan,
                             pub_pick_pct=0.0, exp_pool_opponents=0.0, exp_pool_share=0.0,
                             future_value=0.0, exp_weeks=r["exp_weeks"], p_survive=r["p_survive"],
                             p_full_lives=r["p_lives"][-1], exp_point_diff=r["exp_point_diff"],
                             implied_path=r["path"]))
    elif current_week == TWO_PICK_WEEK:
        avail = [t for t in ABBRS if pd.notna(wp.loc[t, current_week]) and wp.loc[t, current_week] > 0
                 and used0.get(t, 0) < max_uses(t, dd)]
        top = sorted(avail, key=lambda t: -wp.loc[t, current_week])[:8]
        for i, a in enumerate(top):
            for b in top[i + 1:]:
                r = beam_search_path(wp, exp_margin, cfg, hcfg, current_week, used0, my_picks,
                                     forced_first=(a, b), beam_width=cbw, arrays=arrays)
                joint = float(wp.loc[a, current_week] * wp.loc[b, current_week])
                rows.append(dict(team=f"{a}+{b}", team_name=f"{a} + {b}", win_prob=joint,
                                 pub_pick_pct=0.0, exp_pool_opponents=0.0, exp_pool_share=0.0,
                                 future_value=0.0, exp_weeks=r["exp_weeks"], p_survive=r["p_survive"],
                                 p_full_lives=r["p_lives"][-1], exp_point_diff=r["exp_point_diff"],
                                 implied_path=r["path"]))
    else:
        candidates = [t for t in ABBRS if pd.notna(wp.loc[t, current_week])
                      and wp.loc[t, current_week] > 0 and used0.get(t, 0) < max_uses(t, dd)]
        for t in candidates:
            r = beam_search_path(wp, exp_margin, cfg, hcfg, current_week, used0, my_picks,
                                 forced_first=t, beam_width=cbw, arrays=arrays)
            exp_share = float(crowd.loc[t, "exp_share"]) if t in crowd.index else 0.0
            pub = float(crowd.loc[t, "pub_pct"]) if t in crowd.index else 0.0
            fv = 0.0
            for k in range(current_week + 1, cfg.n_weeks + 1):
                p = wp.loc[t, k] if k in wp.columns else np.nan
                if pd.notna(p) and p > hcfg.engine.fv_threshold:
                    fv += (hcfg.engine.fv_discount ** (k - current_week)) * (p - hcfg.engine.fv_threshold)
            score = (r["exp_weeks"]
                     - hcfg.consensus_weeks_penalty * exp_share
                     + hcfg.pointdiff_weight * np.tanh(r["exp_point_diff"] / 40.0))
            rows.append(dict(
                team=t, team_name=display(t), win_prob=float(wp.loc[t, current_week]),
                pub_pick_pct=pub, exp_pool_opponents=float(crowd.loc[t, "exp_opponents"]) if t in crowd.index else 0.0,
                exp_pool_share=exp_share, future_value=fv,
                exp_weeks=r["exp_weeks"], p_survive=r["p_survive"], p_full_lives=r["p_lives"][-1],
                exp_point_diff=r["exp_point_diff"], hakim_score=score,
                implied_path=r["path"]))

    if not rows:
        msg = (f"Week {current_week} is Loser's Week — it needs your Weeks 1-5 win picks "
               f"logged in data/2026/hakim/my_picks.csv before it can recommend a team to "
               f"lose." if current_week == LOSERS_WEEK else
               f"No available teams to rank for Week {current_week}.")
        print(f"[hakim] {msg}")
        out_dir = cfg.week_out_dir() / "hakim"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"week_{current_week:02d}_hakim_recommendation.md").write_text(
            f"# Hakim League — Week {current_week} ({cfg.season}) — {mode.title()} Pool\n\n{msg}\n")
        return dict(mode=mode, week=current_week, lives=hcfg.lives_for_mode(),
                    ranked=pd.DataFrame(), recommendation=None, note=msg,
                    base_path=base["path"], dd_reco=dd_reco,
                    report=dict(markdown=msg, dir=str(out_dir), charts=[]))

    df = pd.DataFrame(rows)
    sort_col = "hakim_score" if "hakim_score" in df.columns else "exp_weeks"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    lo, hi = df["exp_weeks"].min(), df["exp_weeks"].max()
    df["pick_score"] = ((df["exp_weeks"] - lo) / (hi - lo) * 100).round(1) if hi > lo else 50.0

    result = dict(
        mode=mode, week=current_week, lives=L, double_dip=dd, dd_reco=dd_reco,
        ranked=df, recommendation=(df.iloc[0].to_dict() if len(df) else None),
        base_path=base["path"], base_p_survive=base["p_survive"], base_exp_weeks=base["exp_weeks"],
        base_p_lives=base["p_lives"], base_exp_point_diff=base["exp_point_diff"],
        used_counts=used0, winprob_matrix=wp, public_pct=public_pct, crowd=crowd, roster=roster,
        meta=dict(winprob_sources=", ".join(f"{k}:{v}" for k, v in
                                            (src[current_week].value_counts().to_dict().items()
                                             if current_week in src.columns else [])),
                  consensus=cmeta, n_participants=len(roster.participants),
                  n_alive=roster.n_opponents_alive(), roster_source=inp["roster_source"],
                  realized_losses=realized_losses),
    )
    result["report"] = _write(result, cfg, hcfg)
    return result


def _write(result: dict, cfg: Config, hcfg: HakimConfig) -> dict:
    out_dir = cfg.week_out_dir() / "hakim"
    out_dir.mkdir(parents=True, exist_ok=True)
    w = cfg.current_week
    df = result["ranked"]
    df.to_csv(out_dir / f"week_{w:02d}_hakim_ranked_full.csv", index=False)

    L = result["lives"]
    lines = [f"# Hakim League — Week {w} ({cfg.season}) — {result['mode'].title()} Pool\n"]
    rl = result["meta"].get("realized_losses", 0)
    lives_note = f" — {rl} realized loss(es) already deducted" if rl else ""
    lines.append(f"- Lives model: **{L} remaining**{lives_note} "
                 f"({'out on next loss' if L == 1 else 'two losses and out' if L == 2 else 'single elimination'})")
    lines.append(f"- Double-Dip team: **{result['double_dip'] or 'not set'}** "
                 f"(usable {2 if result['double_dip'] else 1}×; 3× if also the Wk6 loser)")
    lines.append(f"- Win-prob sources: {result['meta']['winprob_sources']}")
    lines.append(f"- Public pick %: {result['meta']['consensus'].get('source')}")
    lines.append(f"- Opponent roster: {result['meta']['roster_source']} "
                 f"({result['meta']['n_participants']} entrants, "
                 f"{result['meta']['n_alive']} alive)")
    lines.append(f"- Teams used: {', '.join(f'{k}×{v}' for k, v in result['used_counts'].items()) or 'none'}\n")

    ddr = result.get("dd_reco")
    if ddr is not None and len(ddr):
        prev_po = ddr.attrs.get("prev_playoff_teams", [])
        lines.append("## 🎯 Double-Dip nomination (pre-season)")
        lines.append(f"Eligible = missed the {cfg.season - 1} playoffs. "
                     f"Ineligible (made it): {', '.join(prev_po)}\n")
        lines.append("Ranked by **survivor-asset value** (a team worth using twice), "
                     "with a nudge toward lower-owned nominees:\n")
        top = ddr.head(8)
        tb = pd.DataFrame({
            "rank": top["rank"], "team": top["team"],
            "avg win%": (top["avg_win_prob"] * 100).round(0),
            "strong wks (>60%)": top["strong_weeks"],
            "elite wks (>70%)": top["elite_weeks"],
            "back-half strong": top["back_half_strong"],
            "Wk15 win%": (top["wk15_win_prob"] * 100).round(0),
            "public %": (top["public_pick_pct"] * 100).round(1),
        })
        try:
            lines.append(tb.to_markdown(index=False))
        except Exception:
            lines.append(tb.to_string(index=False))
        best = ddr.iloc[0]
        lines.append(f"\n**Suggested nominee: `{best['team']}`** — {best['avg_win_prob']*100:.0f}% "
                     f"avg win prob, {best['strong_weeks']} weeks projecting >60% "
                     f"({best['back_half_strong']} in the back half), and only "
                     f"{best['public_pick_pct']*100:.1f}% public on them now — a team you'd "
                     f"be happy to lean on twice that the field won't all grab.\n")

    if result.get("double_dip") and ddr is not None and \
            result["double_dip"] not in set(ddr["team"]):
        lines.append(f"> ⚠️ Your set Double-Dip **{result['double_dip']}** made the "
                     f"{cfg.season - 1} playoffs — it's **not eligible**. Pick from the list above.\n")

    rec = result["recommendation"]
    if rec:
        if w == LOSERS_WEEK:
            lines.append(f"## ✅ Week {w} LOSER pick: **{rec['team']}** "
                         f"(P(they lose) = {rec.get('p_lose', float('nan'))*100:.1f}%)")
        elif w == TWO_PICK_WEEK:
            lines.append(f"## ✅ Week {w} two-pick: **{rec['team']}** "
                         f"(both win = {rec['win_prob']*100:.1f}%)")
        else:
            lines.append(f"## ✅ Week {w} pick: **{rec['team']} ({display(rec['team'])})**")
            lines.append(f"- Market win probability: **{rec['win_prob']*100:.1f}%**  |  "
                         f"public {rec['pub_pick_pct']*100:.1f}%")
        lines.append(f"- **Expected weeks survived (of {cfg.n_weeks}, {L} lives): "
                     f"{rec['exp_weeks']:.1f}**")
        lines.append(f"- P(reach season end still alive) = {rec['p_survive']*100:.1f}%  ·  "
                     f"P(never lose all year) = {rec['p_full_lives']*100:.1f}%")
        lines.append(f"- Expected cumulative point differential: {rec['exp_point_diff']:+.1f} "
                     f"(deep tiebreaker only)")
        lines.append(f"- Implied plan: `{_fmt_plan(rec['implied_path'], w)}`\n")

    lines.append("## Optimal season plan (this pool)")
    lines.append(f"`{_fmt_plan(result['base_path'], w, n=20)}`")
    lines.append(f"- E[weeks survived] = **{result['base_exp_weeks']:.1f}** / {cfg.n_weeks}  ·  "
                 f"P(reach end alive) = {result['base_p_survive']*100:.1f}%  ·  "
                 f"P(never lose) = {result['base_p_lives'][-1]*100:.1f}%  ·  "
                 f"E[pt diff] = {result['base_exp_point_diff']:+.1f}")
    if result["double_dip"]:
        dd_weeks = [k for k, v in result["base_path"].items()
                    if v == result["double_dip"] or (isinstance(v, tuple) and result["double_dip"] in v)]
        lines.append(f"- Double-Dip **{result['double_dip']}** used in weeks: "
                     f"{dd_weeks or '(held in reserve)'}")

    # Loser's Week setup: which of the plan's Weeks 1-5 picks are good "pick to lose" options
    if w <= LOSERS_WEEK <= cfg.n_weeks:
        wpm = result["winprob_matrix"]
        setup = []
        for k in range(max(w, 1), 6):
            t = result["base_path"].get(k)
            if not t or isinstance(t, tuple):
                continue
            p6 = wpm.loc[t, LOSERS_WEEK] if LOSERS_WEEK in wpm.columns else None
            if p6 is not None and pd.notna(p6) and p6 > 0:
                setup.append((k, t, 1 - float(p6)))
        setup.sort(key=lambda x: -x[2])
        if setup:
            best_ldr = setup[0]
            lines.append(f"- **Loser's Week (6) setup:** best team-to-lose from this plan is "
                         f"**{best_ldr[1]}** (Wk {best_ldr[0]} pick, ~{best_ldr[2]*100:.0f}% to lose in Wk 6)"
                         + (f"; backup {setup[1][1]} (~{setup[1][2]*100:.0f}%)" if len(setup) > 1 else "")
                         + ".")
        else:
            lines.append("- ⚠️ **Loser's Week (6):** this plan's Weeks 1-5 picks are all "
                         "favourites in Week 6 — you'd likely eat a loss there. Consider "
                         "banking a Week-6 underdog in one of your first 5 picks.")

    lines.append("\n## Ranked options")
    show = pd.DataFrame({
        "rank": df["rank"], "team": df["team"],
        "win_%": (df["win_prob"] * 100).round(1),
        "public_%": (df.get("pub_pick_pct", pd.Series(0, index=df.index)) * 100).round(1),
        "E[wks]": df["exp_weeks"].round(2),
        "P(end)_%": (df["p_survive"] * 100).round(1),
        "E[pt diff]": df["exp_point_diff"].round(1),
        "plan": [_fmt_plan(p, w, 5) for p in df["implied_path"]],
    })
    try:
        lines.append(show.to_markdown(index=False))
    except Exception:
        lines.append(show.to_string(index=False))

    # charts (reuse base viz where the shapes line up)
    pool_tag = f"Hakim — {result['mode'].title()} Pool"
    charts = []
    try:
        charts.append(str(viz.win_probability(
            result["winprob_matrix"][w], cfg, out_dir / f"week_{w:02d}_hakim_win_probability.png",
            pool_label=pool_tag)))
    except Exception as e:  # noqa: BLE001
        print(f"[viz] hakim win_probability failed: {e}")
    try:
        charts.append(str(viz.pick_distribution(
            result["public_pct"], cfg, out_dir / f"week_{w:02d}_hakim_pick_distribution.png",
            pool_label=f"{pool_tag} · {result['meta']['consensus'].get('source')}")))
    except Exception as e:  # noqa: BLE001
        print(f"[viz] hakim pick_distribution failed: {e}")
    try:
        charts.append(str(viz.best_picks(
            df, cfg, out_dir / f"week_{w:02d}_hakim_best_picks.png", pool_label=pool_tag)))
    except Exception as e:  # noqa: BLE001
        print(f"[viz] hakim best_picks failed: {e}")
    try:
        roster = result.get("roster")
        if roster is not None and len(roster.participants) > 0:
            n_alive = roster.n_opponents_alive()
            used_counts = {}
            for opp in roster.alive():
                for t in roster.used_by(opp, w + 1):
                    used_counts[t] = used_counts.get(t, 0) + 1
            remaining = pd.DataFrame([
                dict(team=t, players_left=n_alive - used_counts.get(t, 0), total_players_left=n_alive)
                for t in ABBRS])
            charts.append(str(viz.people_remaining(
                remaining, cfg, out_dir / f"week_{w:02d}_hakim_teams_remaining.png",
                pool_label=pool_tag)))
    except Exception as e:  # noqa: BLE001
        print(f"[viz] hakim people_remaining failed: {e}")
    try:
        tbl = df.copy()
        tbl["ev_path"] = df["exp_weeks"]
        tbl["exp_pool_opponents"] = df.get("exp_pool_opponents", 0.0)
        tbl["pub_pick_pct"] = df.get("pub_pick_pct", 0.0)
        tbl["future_value"] = df.get("future_value", 0.0)
        charts.append(str(viz.ranked_table(
            tbl, cfg, out_dir / f"week_{w:02d}_hakim_ranked_table.png",
            recommend_team=(rec["team"] if rec else None),
            title=f"Hakim League — Week {w} — {result['mode'].title()} Pool",
            ev_header="E[wks]\nsurv.", ev_fmt=lambda v: f"{v:.2f}",
            pool_label=f"{L}-life model · sorted by expected weeks survived "
                       f"(of {cfg.n_weeks}) · double-dip: {result['double_dip'] or '—'}")))
    except Exception as e:  # noqa: BLE001
        print(f"[viz] hakim table failed: {e}")

    md = "\n".join(lines)
    if charts:
        md += "\n\n## Charts\n" + "\n".join(f"- `{c.split('/')[-1]}`" for c in charts)
    (out_dir / f"week_{w:02d}_hakim_recommendation.md").write_text(md)
    return dict(markdown=md, dir=str(out_dir), table=show, charts=charts)
