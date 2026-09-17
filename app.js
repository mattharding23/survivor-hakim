// Survivor Hakim static site. No build step, no backend -- fetches the
// JSON/markdown the weekly Action writes to data/ (relative paths only:
// this is served from a GitHub Pages *project* site, i.e. a subpath, so
// absolute paths like "/data/..." would 404). Single-user: unlike
// survivor-lms there's no member selector, the whole page is "your" Hakim
// recommendation.

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

async function getText(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.text();
}

// ---------------------------------------------------------------------
// Branding -- same palette/sources as the pipeline's chart PNGs
// (survivor_core/branding.py's fallback colors + ESPN logo CDN), so the
// site reads as one system with the exported images.
// ---------------------------------------------------------------------
const TEAM_COLORS = {
  ARI: "#97233F", ATL: "#A71930", BAL: "#241773", BUF: "#00338D",
  CAR: "#0085CA", CHI: "#0B162A", CIN: "#FB4F14", CLE: "#311D00",
  DAL: "#003594", DEN: "#FB4F14", DET: "#0076B6", GB: "#203731",
  HOU: "#03202F", IND: "#002C5F", JAX: "#006778", KC: "#E31837",
  LA: "#003594", LAC: "#0080C6", LV: "#000000", MIA: "#008E97",
  MIN: "#4F2683", NE: "#002244", NO: "#D3BC8D", NYG: "#0B2265",
  NYJ: "#125740", PHI: "#004C54", PIT: "#FFB612", SEA: "#002244",
  SF: "#AA0000", TB: "#D50A0A", TEN: "#0C2340", WAS: "#5A1414",
};
const teamColor = (abbr) => TEAM_COLORS[abbr] || "#444444";
const teamLogo = (abbr) => `https://a.espncdn.com/i/teamlogos/nfl/500/${abbr.toLowerCase()}.png`;
const logoImg = (abbr) => `<img class="logo" src="${teamLogo(abbr)}" alt="" loading="lazy" onerror="this.style.visibility='hidden'">`;

// Same 3-stop red -> amber -> green scale as the ranked_table chart's win%
// column (viz.py's _RGGRAD), clamped over the same [0.30, 0.88] win-prob
// range so a given win% reads as the same color on the site as in the PNG.
function winProbColor(p) {
  const t = Math.max(0, Math.min(1, (p - 0.30) / (0.88 - 0.30)));
  const stops = [[0.75, 0.06, 0.16], [0.85, 0.64, 0.25], [0.25, 0.56, 0.16]]; // red, amber, green
  const seg = t < 0.5 ? [stops[0], stops[1], t * 2] : [stops[1], stops[2], (t - 0.5) * 2];
  const [a, b, f] = seg;
  const mix = (i) => Math.round((a[i] + (b[i] - a[i]) * f) * 255);
  return `rgb(${mix(0)}, ${mix(1)}, ${mix(2)})`;
}
function textOn(rgbStr) {
  const [r, g, b] = rgbStr.match(/\d+/g).map(Number);
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return lum < 0.55 ? "#fff" : "#1a1a1a";
}

// Single-hue white->color ramps approximating the matplotlib sequential
// colormaps (Reds/Blues/Greens) the ranked_table chart uses for its
// public%/E[wks]/score columns.
function seqColor(hueRgb, lo, hi, v) {
  const t = Math.max(0, Math.min(1, ((v ?? lo) - lo) / ((hi - lo) || 1)));
  const f = 0.15 + 0.7 * t;
  const mix = (i) => Math.round(255 + (hueRgb[i] - 255) * f);
  return `rgb(${mix(0)}, ${mix(1)}, ${mix(2)})`;
}
const REDS = [165, 15, 21], BLUES = [8, 81, 156], GREENS = [0, 109, 44], ORANGES = [166, 54, 3];

// ---------------------------------------------------------------------
// Tiny markdown renderer. The only input this ever sees is our own
// hakim_pipeline.py's output -- a fixed, predictable dialect (# / ##
// headers, **bold**, `code`, "- " bullets, "> " blockquotes, and pandas
// .to_markdown() GFM pipe tables) -- so a hand-rolled line-based pass
// covers it without pulling in a library.
// ---------------------------------------------------------------------
function renderMarkdown(md) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const inline = (s) => esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let i = 0;
  let inList = false;

  const closeList = () => { if (inList) { out.push("</ul>"); inList = false; } };

  while (i < lines.length) {
    const line = lines[i];

    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length &&
        /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      closeList();
      const splitRow = (r) => r.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim());
      const header = splitRow(line);
      let j = i + 2;
      const rows = [];
      while (j < lines.length && /^\s*\|.*\|\s*$/.test(lines[j])) {
        rows.push(splitRow(lines[j]));
        j++;
      }
      out.push("<table><thead><tr>" + header.map(h => `<th>${inline(h)}</th>`).join("") + "</tr></thead><tbody>");
      for (const row of rows) {
        out.push("<tr>" + row.map(c => `<td>${inline(c)}</td>`).join("") + "</tr>");
      }
      out.push("</tbody></table>");
      i = j;
      continue;
    }

    if (/^#{1,3}\s+/.test(line)) {
      closeList();
      const m = line.match(/^(#{1,3})\s+(.*)$/);
      const level = Math.min(m[1].length + 1, 4);
      out.push(`<h${level}>${inline(m[2])}</h${level}>`);
    } else if (/^>\s?/.test(line)) {
      closeList();
      out.push(`<blockquote>${inline(line.replace(/^>\s?/, ""))}</blockquote>`);
    } else if (/^-\s+/.test(line)) {
      if (!inList) { out.push("<ul>"); inList = true; }
      out.push(`<li>${inline(line.replace(/^-\s+/, ""))}</li>`);
    } else if (line.trim() === "") {
      closeList();
    } else {
      closeList();
      out.push(`<p>${inline(line)}</p>`);
    }
    i++;
  }
  closeList();
  return out.join("\n");
}

// ---------------------------------------------------------------------
// Public pick % bars (real, from the scohak all_picks.csv field)
// ---------------------------------------------------------------------
function renderPctBars(pct, source) {
  const el = document.getElementById("pct-bars");
  const src = document.getElementById("pct-source");
  src.textContent = source ? `(${source})` : "";
  const entries = Object.entries(pct)
    .filter(([, v]) => v != null && v > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12);
  const max = entries.length ? entries[0][1] : 1;
  el.innerHTML = entries.map(([team, v]) => `
    <div class="bar-row">
      ${logoImg(team)}
      <span class="abbr">${team}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${(v / max) * 100}%;background:${teamColor(team)}"></span></span>
      <span class="pct">${(v * 100).toFixed(1)}%</span>
    </div>`).join("");
}

// ---------------------------------------------------------------------
// Native recreation of viz.py's ranked_table() gt-style chart, using
// Hakim's own metrics (expected weeks survived under the life model,
// P(survive), the composite hakim_score-derived pick_score) rather than
// LMS's EV(path)/future-value columns.
// ---------------------------------------------------------------------
const LOSERS_WEEK = 6; // Week 6: pick one of your Weeks 1-5 winners to LOSE -- doesn't consume a team

function fmtPlanEntry(wk, team) {
  // implied_path values are either a plain team abbr, a ["LOSE", team] pair
  // (Loser's Week -- shown as "LOSE X", never "LOSE+X"), or a [teamA, teamB]
  // pair (Week 15's two-pick week -- shown as "A+B"). Mirrors hakim_pipeline
  // .py's own _fmt_plan() formatting exactly.
  if (Array.isArray(team)) {
    return team[0] === "LOSE" ? `W${wk}:LOSE ${team[1]}` : `W${wk}:${team.join("+")}`;
  }
  return `W${wk}:${team}`;
}

function renderHakimTable(teams, currentWeek) {
  const rows = [...teams].sort((a, b) => a.rank - b.rank);
  const cell = (color, text) => `<td class="scale-cell" style="background:${color};color:${textOn(color)}">${text}</td>`;
  // Always show through Loser's Week (6) while it's still ahead -- every
  // candidate's plan resolves a real week-6 decision under the hood, but a
  // fixed "next 4 weeks" window can cut off before reaching it this early
  // in the season, which hid that decision even though it's already made.
  const planEndWeek = currentWeek <= LOSERS_WEEK ? LOSERS_WEEK : currentWeek + 3;
  const body = rows.map(t => {
    const plan = Object.entries(t.implied_path || {})
      .filter(([wk]) => Number(wk) >= currentWeek && Number(wk) <= planEndWeek)
      .sort((a, b) => Number(a[0]) - Number(b[0]))
      .map(([wk, team]) => fmtPlanEntry(wk, team)).join("  ");
    return `<tr class="${t.rank === 1 ? "is-rec" : ""}">
      <td>${t.rank}</td>
      <td class="team-cell">${logoImg(t.team)}<span>${t.team}</span></td>
      ${cell(winProbColor(t.win_prob), (t.win_prob * 100).toFixed(1) + "%")}
      ${cell(seqColor(REDS, 0, 0.35, t.pub_pick_pct), (t.pub_pick_pct * 100).toFixed(1) + "%")}
      ${cell(seqColor(BLUES, 0, 18, t.exp_weeks), t.exp_weeks.toFixed(2))}
      ${cell(seqColor(ORANGES, 0, 18 - currentWeek, t.weeks_left ?? 0), t.weeks_left ?? 0)}
      <td>${(t.p_survive * 100).toFixed(1)}%</td>
      <td>${t.exp_point_diff >= 0 ? "+" : ""}${t.exp_point_diff.toFixed(1)}</td>
      ${cell(seqColor(GREENS, 0, 100, t.pick_score), t.pick_score.toFixed(0))}
      <td class="plan-cell">${plan}</td>
    </tr>`;
  }).join("");
  return `<h3>Ranked Available Teams</h3>
    <div class="tablewrap"><table>
      <thead><tr>
        <th>#</th><th>Team</th><th>Win %</th><th>Public %</th><th>E[wks]</th><th>Wks Left</th>
        <th>P(survive)</th><th>E[pt diff]</th><th>Score</th><th>Plan (thru Wk ${planEndWeek})</th>
      </tr></thead>
      <tbody>${body}</tbody>
    </table></div>
    <p class="muted" style="margin-top:10px">
      <b>Wks Left</b> = how many of the remaining weeks this season this team is still projected
      to be a "strong" pick (win prob &gt; 62%) -- a rough read on how much longer you can afford
      to save it instead of using it now.
    </p>`;
}

// ---------------------------------------------------------------------
// Life tracker
// ---------------------------------------------------------------------
function renderLives(lives, livesMax, realizedLosses) {
  const pips = document.getElementById("lives-pips");
  const text = document.getElementById("lives-text");
  const lost = livesMax - lives;
  pips.innerHTML = Array.from({ length: livesMax }, (_, i) =>
    `<span class="pip ${i < lost ? "lost" : "alive"}"></span>`).join("");
  const noun = lives === 1 ? "life" : "lives";
  text.textContent = `${lives} of ${livesMax} ${noun} remaining` +
    (realizedLosses ? ` (${realizedLosses} realized loss${realizedLosses > 1 ? "es" : ""})` : "");
}

// ---------------------------------------------------------------------
// Chart viewer -- a dropdown + <img> pair, fed by the actual branded PNGs
// hakim_pipeline.py already generates (win probability, pick distribution,
// best picks, ranked table, teams remaining pool-wide) -- teams remaining
// first/default since it's the one metric with no other view on the page.
// ---------------------------------------------------------------------
function wireChartViewer(selectEl, imgEl, basePath, charts) {
  selectEl.innerHTML = "";
  if (!charts || !charts.length) {
    selectEl.disabled = true;
    imgEl.removeAttribute("src");
    imgEl.alt = "No charts available yet";
    return;
  }
  selectEl.disabled = false;
  charts.forEach(c => selectEl.add(new Option(c.label, c.file)));
  const show = (file) => {
    const c = charts.find(c => c.file === file) || charts[0];
    imgEl.src = basePath + c.file;
    imgEl.alt = c.label;
  };
  selectEl.onchange = () => show(selectEl.value);
  show(charts[0].file);
}

async function loadCharts() {
  const manifest = await getJSON("data/charts/manifest.json").catch(() => ({ charts: [] }));
  const order = ["teams_remaining", "win_probability", "pick_distribution", "best_picks", "ranked_table"];
  const charts = order
    .map(key => (manifest.charts || []).find(c => c.key === key))
    .filter(Boolean);
  wireChartViewer(
    document.getElementById("chart-select"),
    document.getElementById("chart-img"),
    "data/charts/",
    charts,
  );
}

// ---------------------------------------------------------------------
// Main load
// ---------------------------------------------------------------------
async function loadAll() {
  const [pct, ranked, mdResult] = await Promise.all([
    getJSON("data/public_pct.json"),
    getJSON("data/ranked.json"),
    getText("data/recommendation.md"),
  ]);

  document.getElementById("week-line").textContent =
    `Week ${ranked.week} — generated ${new Date(ranked.generated_at).toLocaleString()}`;
  document.getElementById("pool-sub").textContent =
    `scohak.com "Survivor Football League" — ${ranked.mode === "consolation" ? "Consolation" : "Main"} Pool ` +
    `· ${ranked.n_participants.toLocaleString()} entries tracked (${ranked.roster_source})`;
  document.getElementById("dd-line").textContent =
    `Double-Dip: ${ranked.double_dip || "not set"}`;
  renderLives(ranked.lives, ranked.lives_max, ranked.realized_losses);

  renderPctBars(pct.pct, pct.source);

  const tableWrap = document.getElementById("ranked-table-wrap");
  tableWrap.innerHTML = ranked.teams && ranked.teams.length
    ? renderHakimTable(ranked.teams, ranked.week)
    : "";

  const tmp = document.createElement("div");
  tmp.innerHTML = renderMarkdown(mdResult);
  // The markdown's own "## Ranked options" pipe table and "## Charts"
  // filename list are superseded by the native table above and the chart
  // viewer -- drop both so nothing's shown three times over.
  const dropSectionAfter = (re) => {
    const h = Array.from(tmp.querySelectorAll("h1,h2,h3")).find(el => re.test(el.textContent.trim()));
    if (!h) return;
    let sib = h.nextElementSibling;
    h.remove();
    while (sib) {
      const next = sib.nextElementSibling;
      const isTable = sib.tagName === "TABLE" || !!sib.querySelector?.("table");
      sib.remove();
      sib = next;
      if (isTable) break;
    }
  };
  dropSectionAfter(/ranked options/i);
  dropSectionAfter(/^charts$/i);
  document.getElementById("recommendation-md").innerHTML = tmp.innerHTML;
}

loadAll().catch(e => {
  document.getElementById("week-line").textContent = `Couldn't load site data (${e.message}).`;
});
loadCharts().catch(() => {});
