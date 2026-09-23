"""Hakim / scohak "Survivor Football League" variant.

Differences from a plain survivor pool (see rules):
  * TWO LIVES in the main pool — you're out on your 2nd loss (buy-back $5, or drop
    to the Consolation Pool if the 2nd loss is in Weeks 1-7).
  * Consolation Pool = single elimination (1 life), used teams carry over, you
    start picking the week you enter.
  * Loser's Week (Week 6): pick one of your Weeks 1-5 WIN picks to LOSE. It does
    NOT consume the team and it isn't bound by the no-reuse rule.
  * Two-Pick Week (Week 15): pick TWO teams; BOTH must win or it's a loss; both
    are consumed.
  * Double-Dip: one nominated team (missed playoffs last year) may be used twice
    (a third time only as the Week 6 loser pick).
  * Point differential is only a deep tiebreaker (#6 of 9) — reported, lightly
    weighted, never the driver.

Objective: maximise P(still alive at season's end) given the life count, via a
beam search over week-by-week team assignments that tracks the full life
distribution.  The plain-pool EV engine is reused for the single-week ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import re

from .config import Config
from survivor_core.config import EngineConfig
from survivor_core.constraints import Roster
from survivor_core.teams import ABBRS, display, to_abbr

_WK_COL_RE = re.compile(r"^\s*Wk\s*(\d+)\s*$", re.IGNORECASE)
_RANK_PREFIX_RE = re.compile(r"^\s*\d+\.\s*")

LOSERS_WEEK = 6
TWO_PICK_WEEK = 15
WP_LO, WP_HI = 0.02, 0.995


# --------------------------------------------------------------------------- #
# config + manual inputs
# --------------------------------------------------------------------------- #
@dataclass
class HakimConfig:
    mode: str = "main"                 # "main" (2 lives) or "consolation" (1 life)
    lives: int | None = None           # override; default 2 main / 1 consolation
    entry_week: int | None = None      # first week you pick in THIS pool (consolation
                                        # entrants: the week after the loss that dropped
                                        # them there -- losses before it already spent
                                        # the life that got them here, and don't also
                                        # cost their fresh life in this pool)
    double_dip: str | None = None      # team abbr usable twice
    engine: EngineConfig = field(default_factory=EngineConfig)
    beam_width: int = 320
    branch: int = 7                    # candidate teams explored per week
    pointdiff_weight: float = 0.02     # tiny tiebreak nudge toward bigger favourites
    cand_beam_width: int = 64          # beam width for per-candidate scoring passes
    consensus_weeks_penalty: float = 1.2  # weeks-equiv penalty when the whole pool is on a team

    def lives_for_mode(self) -> int:
        if self.lives is not None:
            return self.lives
        return 1 if self.mode == "consolation" else 2


def _hakim_dir(cfg: Config) -> Path:
    d = cfg.year_dir / "hakim"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_all_picks_roster(cfg: Config, hcfg: HakimConfig) -> Roster | None:
    """Parse the scohak-exported data/<season>/hakim/all_picks.csv (real picks for
    every entrant, one "Wk N" column per week: '✓TEAM' win, '✗TEAM' loss, '—'
    pending) into a Roster, tracking cumulative losses to mark entrants OUT once
    they exceed this mode's life count.

    Returns None if the file doesn't exist, so callers can fall back to a
    hand-maintained participants.csv.
    """
    f = _hakim_dir(cfg) / "all_picks.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, dtype=str, encoding="utf-8-sig").fillna("")
    wk_cols = [(c, int(_WK_COL_RE.match(c).group(1))) for c in df.columns
              if _WK_COL_RE.match(c)]
    lives = hcfg.lives_for_mode()

    rows = []
    for _, r in df.iterrows():
        entry = _RANK_PREFIX_RE.sub("", str(r.get("Entry", "")).strip()).strip()
        if not entry:
            continue
        losses = 0
        out = False
        for col, wk in sorted(wk_cols, key=lambda x: x[1]):
            cell = str(r[col]).strip()
            if not cell or cell == "—" or out:
                continue
            result = "loss" if cell.startswith("✗") else "win" if cell.startswith("✓") else "win"
            team_raw = cell.lstrip("✓✗").strip()
            if not team_raw or team_raw == "—":
                continue
            try:
                team = to_abbr(team_raw)
            except KeyError:
                continue
            rows.append((entry, wk, team))
            if result == "loss":
                losses += 1
                if losses >= lives:
                    rows.append((entry, 0, "OUT"))
                    out = True
    return Roster(pd.DataFrame(rows, columns=["participant", "week", "team"]))


def pick_pct_from_roster(roster: Roster, week: int) -> pd.Series | None:
    """Real pick % for `week` straight from the parsed all_picks.csv roster --
    the actual field for this pool, not a national-pool proxy. None if the
    roster has no picks logged for that week yet (e.g. participants.csv fallback,
    or a future week nobody has picked)."""
    m = roster.picks["week"].eq(week)
    counts = roster.picks.loc[m, "team"].value_counts()
    counts = counts[counts.index != "OUT"]
    if counts.empty:
        return None
    total = counts.sum()
    full = pd.Series(0.0, index=ABBRS)
    full.update(counts / total)
    return full


def load_hakim_inputs(cfg: Config, hcfg: HakimConfig) -> dict:
    d = _hakim_dir(cfg)
    # my_picks.csv : week,team,role   role in {win, loser, w15a, w15b}
    picks_f = d / "my_picks.csv"
    if not picks_f.exists():
        picks_f.write_text("week,team,role\n"
                           "# week,team,role  — role: win (default) | loser (Wk6) | w15a | w15b\n")
    mp = pd.read_csv(picks_f, dtype=str, comment=None).fillna("")
    mp = mp[mp["week"].astype(str).str.strip().str.match(r"^\d+$")]
    my_picks = []
    for _, r in mp.iterrows():
        role = (r.get("role") or "win").strip().lower() or "win"
        my_picks.append(dict(week=int(r["week"]), team=to_abbr(r["team"]), role=role))

    # double_dip.csv : single line with the team
    dd_f = d / "double_dip.csv"
    if not dd_f.exists():
        dd_f.write_text("team\n# one team abbr that missed the playoffs last year\n")
    dd = hcfg.double_dip
    if dd is None:
        for line in dd_f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "team")):
                dd = to_abbr(line)
                break

    # entry_week.csv : single line, the first week you pick in the CURRENT pool
    # (only meaningful for mode=="consolation" -- the week after the loss that
    # dropped you there).
    ew_f = d / "entry_week.csv"
    if not ew_f.exists():
        ew_f.write_text("week\n# first week you pick in this pool -- consolation "
                        "entrants: the week after the loss that dropped you there\n")
    entry_week = hcfg.entry_week
    if entry_week is None:
        entry_week = 1
        for line in ew_f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "week")):
                entry_week = int(line)
                break

    # participants.csv : participant,week,team[,role] — hand-maintained fallback,
    # only consulted when all_picks.csv (the real scohak export) isn't present.
    part_f = d / "participants.csv"
    if not part_f.exists():
        part_f.write_text("participant,week,team,role\n"
                          "# opponents' picks; role optional (win|loser|w15a|w15b); "
                          "'name,0,OUT' once they're eliminated\n")

    roster = load_all_picks_roster(cfg, hcfg)
    roster_source = "all_picks.csv"
    if roster is None:
        roster = Roster.load(part_f)
        roster_source = "participants.csv (manual)"

    return dict(my_picks=my_picks, double_dip=dd, entry_week=entry_week, participants_file=part_f,
               roster=roster, roster_source=roster_source)


# --------------------------------------------------------------------------- #
# team-usage accounting (with double-dip multiplicity)
# --------------------------------------------------------------------------- #
def max_uses(team: str, double_dip: str | None) -> int:
    return 2 if (double_dip and team == double_dip) else 1


def used_counts_from_picks(my_picks: list[dict]) -> dict[str, int]:
    """Weeks-6 loser picks do NOT count as a use."""
    c: dict[str, int] = {}
    for p in my_picks:
        if p["role"] == "loser":
            continue
        c[p["team"]] = c.get(p["team"], 0) + 1
    return c


def weeks_1_5_win_picks(my_picks: list[dict]) -> list[str]:
    return [p["team"] for p in my_picks if p["role"] == "win" and 1 <= p["week"] <= 5]


def realized_life_losses(my_picks: list[dict], wp: pd.DataFrame, before_week: int,
                         entry_week: int = 1) -> int:
    """How many of your already-played 'win'-role picks (entry_week <= week <
    before_week) actually lost, per the win-prob matrix (which holds 1.0/0.0
    for decided games, sourced from real results). Loser's Week picks that
    *won* also cost a life; role=='loser' + win_prob near 1 counts as a loss
    here too.

    Picks before entry_week are excluded: for a Consolation entrant, those
    are the Main-pool losses that got them here -- already priced into the
    drop, not a further hit against their fresh Consolation life."""
    losses = 0
    for p in my_picks:
        if p["week"] >= before_week or p["week"] < entry_week:
            continue
        if p["team"] not in wp.index or p["week"] not in wp.columns:
            continue
        pw = wp.loc[p["team"], p["week"]]
        if pd.isna(pw):
            continue
        if p["role"] == "loser":
            if pw > 0.5:  # your designated loser pick actually won -> costs a life
                losses += 1
        elif pw < 0.5:
            losses += 1
    return losses


# --------------------------------------------------------------------------- #
# beam search over the remaining season   (vectorised)
# --------------------------------------------------------------------------- #
_TIDX = {a: i for i, a in enumerate(ABBRS)}


def _clip(p): return float(min(max(p, WP_LO), WP_HI))


def to_arrays(wp: pd.DataFrame, exp_margin: pd.DataFrame, n_weeks: int):
    """DataFrames -> (WPA, EMA) float arrays shape (32, n_weeks+1); col w = week w."""
    wpa = np.zeros((32, n_weeks + 1))
    ema = np.zeros((32, n_weeks + 1))
    for a, i in _TIDX.items():
        for w in range(1, n_weeks + 1):
            v = wp.loc[a, w] if w in wp.columns else np.nan
            wpa[i, w] = 0.0 if pd.isna(v) else float(v)
            m = exp_margin.loc[a, w] if w in exp_margin.columns else np.nan
            ema[i, w] = 0.0 if pd.isna(m) else float(m)
    return wpa, ema


_W6_WEIGHT = 0.5   # how much "walking into Loser's Week with a losable team" is worth


def _w6_readiness(path: dict, WPA) -> float:
    """[0..~1.45] — do the Weeks 1-5 win picks include a team (ideally two) that is
    a coin-flip-or-worse to WIN in Week 6, i.e. a good team to pick to LOSE?"""
    pls = []
    for w in range(1, 6):
        v = path.get(w)
        if v is None or isinstance(v, tuple):
            continue
        i = _TIDX.get(v)
        if i is None or WPA[i, LOSERS_WEEK] <= 0:
            continue
        pls.append(min(1.0 - WPA[i, LOSERS_WEEK], 0.55) / 0.55)
    pls.sort(reverse=True)
    r = pls[0] if pls else 0.0
    if len(pls) > 1:
        r += 0.45 * pls[1]
    return r


class _State:
    __slots__ = ("used", "p_lives", "exp_weeks", "exp_pd", "path", "w6ready")

    def __init__(self, used, p_lives, exp_weeks, exp_pd, path, w6ready=0.0):
        self.used = used            # dict team_idx -> count
        self.p_lives = p_lives      # np array len L+1  (idx 0 = dead)
        self.exp_weeks = exp_weeks  # E[weeks with a life at week start]
        self.exp_pd = exp_pd
        self.path = path            # week -> abbr | (abbr,abbr) | ("LOSE",abbr)
        self.w6ready = w6ready

    def alive(self):
        return float(self.p_lives[1:].sum())

    def score(self):
        return (self.exp_weeks + _W6_WEIGHT * self.w6ready,
                self.alive(), self.p_lives[-1], self.exp_pd)


def _adv(p_lives, psucc):
    q = 1.0 - psucc
    out = p_lives * psucc
    out[:-1] += p_lives[1:] * q
    return out


def beam_search_path(wp, exp_margin, cfg: Config, hcfg: HakimConfig, start_week: int,
                     used0: dict, my_picks: list[dict],
                     forced_first=None, beam_width: int | None = None,
                     arrays=None) -> dict:
    L = hcfg.lives_for_mode()
    n_weeks = cfg.n_weeks
    weeks = list(range(start_week, n_weeks + 1))
    bw = beam_width or hcfg.beam_width
    br = hcfg.branch
    WPA, EMA = arrays if arrays is not None else to_arrays(wp, exp_margin, n_weeks)

    dd_idx = _TIDX[hcfg.double_dip] if hcfg.double_dip in _TIDX else -1
    maxu = np.ones(32, dtype=int)
    if dd_idx >= 0:
        maxu[dd_idx] = 2

    used_start = {_TIDX[t]: c for t, c in used0.items() if t in _TIDX}
    seed_path = {p["week"]: p["team"] for p in my_picks if p["role"] == "win"}
    seed_w15 = [_TIDX[p["team"]] for p in my_picks if 1 <= p["week"] <= 5 and p["role"] == "win"]

    init = np.zeros(L + 1)
    init[L] = 1.0
    beam = [_State(dict(used_start), init, 0.0, 0.0, dict(seed_path))]

    for wk in weeks:
        forced = forced_first if wk == start_week else None
        nxt: list[_State] = []
        for st in beam:
            p_alive_before = st.alive()
            if p_alive_before < 1e-4 and not forced:
                nxt.append(st)
                continue
            ew = st.exp_weeks + p_alive_before

            def emit(pick, psucc, dpd, used_delta):
                u = dict(st.used)
                for k in used_delta:
                    u[k] = u.get(k, 0) + 1
                path = dict(st.path); path[wk] = pick
                w6r = _w6_readiness(path, WPA) if wk <= 5 else st.w6ready
                nxt.append(_State(u, _adv(st.p_lives, _clip(psucc)), ew,
                                  st.exp_pd + dpd, path, w6r))

            # ---- forced first pick (used for per-candidate scoring) ----
            if forced is not None:
                if isinstance(forced, tuple) and forced and forced[0] == "LOSE":
                    ti = _TIDX[forced[1]]
                    emit(("LOSE", forced[1]), 1.0 - WPA[ti, wk], EMA[ti, wk], [])
                elif isinstance(forced, tuple):
                    ai, bi = _TIDX[forced[0]], _TIDX[forced[1]]
                    emit((forced[0], forced[1]), WPA[ai, wk] * WPA[bi, wk],
                         EMA[ai, wk] + EMA[bi, wk], [ai, bi])
                else:
                    ti = _TIDX[forced]
                    emit(forced, WPA[ti, wk], EMA[ti, wk], [ti])
                continue

            # ---- Loser's Week ----
            if wk == LOSERS_WEEK:
                pool = list(dict.fromkeys(
                    [_TIDX[st.path[w]] for w in range(1, 6)
                     if w in st.path and not isinstance(st.path[w], tuple)] or seed_w15))
                pool = [i for i in pool if WPA[i, wk] > 0]          # can't "lose" on a bye
                if not pool:
                    # no eligible team to lose with -> you take the loss this week
                    emit(("LOSE", "—none—"), 0.15, 0.0, [])
                    continue
                order = sorted(pool, key=lambda i: WPA[i, wk])[:3]   # most likely to lose
                for ti in order:
                    p_lose = float(np.clip(1.0 - WPA[ti, wk], 0.05, 0.95))
                    emit(("LOSE", ABBRS[ti]), p_lose, EMA[ti, wk], [])
                continue

            # ---- available teams this week ----
            avail = [i for i in range(32)
                     if WPA[i, wk] > 0 and st.used.get(i, 0) < maxu[i]]
            if not avail:
                nxt.append(_State(dict(st.used), st.p_lives.copy(), ew, st.exp_pd,
                                  dict(st.path), st.w6ready))
                continue
            avail.sort(key=lambda i: -WPA[i, wk])

            # ---- Two-Pick Week ----
            if wk == TWO_PICK_WEEK:
                top = avail[:6]
                pairs = sorted(((a, b) for j, a in enumerate(top) for b in top[j + 1:]),
                               key=lambda ab: -(WPA[ab[0], wk] * WPA[ab[1], wk]))[:br]
                for ai, bi in pairs:
                    emit((ABBRS[ai], ABBRS[bi]), WPA[ai, wk] * WPA[bi, wk],
                         EMA[ai, wk] + EMA[bi, wk], [ai, bi])
                continue

            cand = avail[:br]
            # Weeks 1-5: also let the beam bank teams that are dogs in Week 6, so it
            # walks into Loser's Week with a genuine team-to-lose option (or two).
            if start_week <= wk <= 5 and start_week <= LOSERS_WEEK <= cfg.n_weeks:
                losable = sorted((i for i in avail
                                  if 0 < WPA[i, LOSERS_WEEK] < 0.55 and WPA[i, wk] > 0.50),
                                 key=lambda i: WPA[i, LOSERS_WEEK])[:3]
                cand = list(dict.fromkeys(list(cand) + losable))
            for ti in cand:
                emit(ABBRS[ti], WPA[ti, wk], EMA[ti, wk], [ti])

        # prune
        nxt.sort(key=lambda s: s.score(), reverse=True)
        seen, pruned = set(), []
        for s in nxt:
            key = (tuple(sorted(s.used.items())), str(s.path.get(wk)))
            if key in seen:
                continue
            seen.add(key)
            pruned.append(s)
            if len(pruned) >= bw:
                break
        beam = pruned

    best = max(beam, key=lambda s: s.score())
    return dict(
        path=best.path,
        exp_weeks=best.exp_weeks,
        p_survive=best.alive(),
        p_lives=best.p_lives.tolist(),
        exp_point_diff=best.exp_pd,
        lives_start=L,
    )


# --------------------------------------------------------------------------- #
# Double-Dip nomination helper
# --------------------------------------------------------------------------- #
def recommend_double_dip(wp, exp_margin, cfg: Config, hcfg: HakimConfig,
                         used0: dict, my_picks: list[dict],
                         public_pct=None, prev_playoff_teams: set | None = None,
                         arrays=None) -> pd.DataFrame:
    """Rank eligible teams (missed last year's playoffs) as Double-Dip nominees.

    A good Double-Dip is a team you'd genuinely *want* to use twice — a solid
    club with several strong survivor weeks (especially in the back half and the
    Week-15 two-pick), and ideally one the field WON'T all nominate.  Scored by a
    transparent "survivor asset value" heuristic, not the beam (the beam never
    runs low enough on teams for a 2nd use to bind).
    """
    from survivor_core.teams import to_abbr as _abbr
    eligible = [t for t in ABBRS if not prev_playoff_teams or t not in prev_playoff_teams]
    wk0, W = cfg.current_week, cfg.n_weeks
    HALF = (wk0 + W) // 2

    def wpv(t, w):
        v = wp.loc[t, w] if w in wp.columns else np.nan
        return float(v) if pd.notna(v) else np.nan

    rows = []
    for t in eligible:
        vals = [wpv(t, w) for w in range(wk0, W + 1)]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            continue
        strong = sum(1 for v in vals if v > 0.60)
        elite = sum(1 for v in vals if v > 0.70)
        back_strong = sum(1 for w in range(HALF, W + 1)
                          if not np.isnan(wpv(t, w)) and wpv(t, w) > 0.60)
        wk15 = wpv(t, TWO_PICK_WEEK)
        wk15 = 0.0 if np.isnan(wk15) else wk15
        avg = float(np.mean(vals))
        chalk = float(public_pct.get(t, 0.0)) if public_pct is not None else 0.0
        # "how much would a 2nd use of this team be worth"
        asset = (0.5 * strong + 1.0 * elite + 0.8 * back_strong
                 + 3.0 * avg + 2.5 * wk15)
        rows.append(dict(
            team=t, asset_value=asset, avg_win_prob=avg,
            strong_weeks=strong, elite_weeks=elite, back_half_strong=back_strong,
            wk15_win_prob=wk15, public_pick_pct=chalk,
            contrarian_score=asset - 6.0 * chalk))
    df = pd.DataFrame(rows).sort_values("contrarian_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# --------------------------------------------------------------------------- #
# expected margin (for the point-diff tiebreaker metric)
# --------------------------------------------------------------------------- #
def expected_margin_matrix(schedule: pd.DataFrame, wp: pd.DataFrame, cfg: Config,
                           sg=None) -> pd.DataFrame:
    """team x week expected margin (points).  + = expected to win by that many."""
    em = pd.DataFrame(index=ABBRS, columns=list(range(1, cfg.n_weeks + 1)), dtype=float)
    sd = cfg.elo.margin_sd
    sg_ok = sg is not None and getattr(sg, "ok", False)
    for _, g in schedule.iterrows():
        wk, home, away = int(g["week"]), g["home"], g["away"]
        m = np.nan
        if sg_ok and wk in sg.spread_grid.columns and pd.notna(sg.spread_grid.loc[home, wk]):
            m = -float(sg.spread_grid.loc[home, wk])          # grid: neg = favoured
        elif pd.notna(g.get("spread_line")):
            m = float(g["spread_line"])
        else:
            p = wp.loc[home, wk]
            if pd.notna(p):
                from scipy.stats import norm
                m = float(norm.ppf(min(max(p, 1e-4, ), 1 - 1e-4)) * sd)
        if pd.notna(m):
            em.loc[home, wk] = m
            em.loc[away, wk] = -m
    return em
