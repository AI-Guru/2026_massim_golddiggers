// Gold Diggers frontend: polls /api/state, draws the world on a canvas, shows stats.
// Sprites: /static/assets/<name>.png if present (Retro Diffusion), else built-in pixel art.

const SPRITE_NAMES = ["grass", "tree", "gold", "depot", "mark", "miner0", "miner1"];
const PX = 16;                         // sprite resolution

// --- built-in 16x16 pixel art ----------------------------------------------------------
const PALETTE = {
  ".": null, g: "#2f4a26", G: "#36552c", h: "#3d6131",
  t: "#1f5b2a", T: "#2e7d36", L: "#49a04a", b: "#5a3a1e", B: "#3e2814",
  y: "#ffd23f", Y: "#f2a900", o: "#a86b00", w: "#fff6c8",
  r: "#8a5a2b", R: "#6b4220", k: "#2a1a0c", s: "#9aa0a6", S: "#c9ced3",
  f: "#f4f0e6", p: "#d33", c: "#f0c27b", C: "#c98f4a", e: "#222", m: "#777",
  X: "TEAM", x: "TEAMDARK",
};
const ART = {
  grass: [
    "gggggggggggggggg", "ggGgggggggggGggg", "gggggghggggggggg", "gggggggggGgggggg",
    "gGgggggggggggggg", "gggggggggggghggg", "gggghgggGggggggg", "gggggggggggggggg",
    "ggggggggggggGggg", "ggGggggghggggggg", "gggggggggggggggg", "gggggGgggggggghg",
    "ghgggggggggggggg", "gggggggggGgggggg", "ggggGggggggggggg", "gggggggggggggGgg"],
  tree: [
    "......TTTT......", "....TTLLTTT.....", "...TTLLLTTTT....", "..TTLLTTTTTtT...",
    "..TLLTTTTTTtt...", ".TTLTTTTTTTttT..", ".TTTTTTTTTtttT..", ".tTTTTTTTttttt..",
    "..tTTTTTtttttt..", "..ttTTtttttttt..", "...tttttttttt...", ".....ttbBtt.....",
    "......bbB.......", "......bbB.......", ".....bbbBB......", "....kkkkkkk....."],
  gold: [
    "................", "................", "................", "................",
    "................", "......yy........", ".....ywyy.......", "....yyyyYY..yy..",
    "....yyYYYo.ywyY.", ".....YYoo..yYYo.", "......oo....oo..", "..yy............",
    ".ywyY...........", ".yYYo...........", "..oo............", "................"],
  depot: [
    "................", "...RRRRRRRRRR...", "..RrrrrrrrrrrR..", ".RrrrrrrrrrrrrR.",
    "RRRRRRRRRRRRRRRR", "kSSSSSSSSSSSSSSk", "kryyyyrrrryyyyrk", "kryYYyrSSryYYyrk",
    "krrrrrrSSrrrrrrk", "krrrrrrSSrrrrrrk", "krrrrrrrrrrrrrrk", "krrrrrrrrrrrrrrk",
    "kRRRRRRRRRRRRRRk", "kkkkkkkkkkkkkkkk", "................", "................"],
  mark: [
    "................", "................", "....mppppp......", "....mpppppp.....",
    "....mppppp......", "....m...........", "....m...........", "....m...........",
    "....m...........", "....m...........", "...mmm..........", "................",
    "................", "................", "................", "................"],
  miner: [
    "................", ".....SSSSS......", "....SSyySSS.....", "...xxxxxxxxx....",
    ".....cccc.......", ".....cecce......", ".....cccc.......", "....XXXXXX......",
    "...XXXXXXXXs....", "...cXXXXXXcs....", "....XXXXXX.s....", "....BBBBBB.s....",
    "....BB..BB.S....", "....BB..BB......", "...kkk..kkk.....", "................"],
};

function shade(hex, f) {
  const n = parseInt(hex.slice(1), 16);
  const c = [n >> 16, (n >> 8) & 255, n & 255].map(v => Math.max(0, Math.min(255, Math.round(v * f))));
  return "#" + c.map(v => v.toString(16).padStart(2, "0")).join("");
}

function drawArt(rows, teamColor) {
  const cv = document.createElement("canvas");
  cv.width = cv.height = PX;
  const ctx = cv.getContext("2d");
  rows.forEach((row, y) => [...row].forEach((ch, x) => {
    let col = PALETTE[ch];
    if (col === "TEAM") col = teamColor;
    if (col === "TEAMDARK") col = shade(teamColor, 0.7);
    if (col) { ctx.fillStyle = col; ctx.fillRect(x, y, 1, 1); }
  }));
  return cv;
}

const sprites = {};
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function loadSprites() {
  const team = [cssVar("--team0"), cssVar("--team1")];
  const builtin = {
    grass: drawArt(ART.grass), tree: drawArt(ART.tree), gold: drawArt(ART.gold),
    depot: drawArt(ART.depot), mark: drawArt(ART.mark),
    miner0: drawArt(ART.miner, team[0]), miner1: drawArt(ART.miner, team[1]),
  };
  Object.assign(sprites, builtin);
  return Promise.all(SPRITE_NAMES.map(name => new Promise(resolve => {
    const img = new Image();
    img.onload = () => { sprites[name] = img; resolve(); };
    img.onerror = () => resolve();                    // keep built-in art
    img.src = `/static/assets/${name}.png`;
  }))).then(() => {
    document.querySelectorAll(".sw[data-sprite]").forEach(el => {
      const s = sprites[el.dataset.sprite];
      el.style.backgroundImage = `url(${s.toDataURL ? s.toDataURL() : s.src})`;
    });
  });
}

// --- state & rendering -----------------------------------------------------------------
let state = null;
let cellSize = 16;
const BORDER = 1;                      // cells of decorative trees around the map
const world = document.getElementById("world");
const wctx = world.getContext("2d");
const $ = id => document.getElementById(id);

function teamIndex(team) { return state.config.teams.indexOf(team); }
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// display name: "greedy (red)" once a team has announced itself via /api/setteamname
function teamLabel(team) {
  const name = (state.team_names || {})[team] || team;
  return name === team ? team : `${name} (${team})`;
}

function renderWorld() {
  if (!state || !state.running) return;
  const { width: W, height: H } = state.config;
  const frameWidth = world.parentElement.clientWidth - 16;
  cellSize = Math.max(4, Math.floor(Math.min(frameWidth, Math.max(320, window.innerHeight - 120)) / (Math.max(W, H) + 2 * BORDER)));
  const pw = (W + 2 * BORDER) * cellSize, ph = (H + 2 * BORDER) * cellSize;
  if (world.width !== pw || world.height !== ph) { world.width = pw; world.height = ph; }
  world.style.width = pw + "px";
  wctx.imageSmoothingEnabled = false;
  const cs = cellSize;
  // ground: the grass tile faded over a flat colour, so trees and gold stand out against it
  wctx.fillStyle = "#24331f";
  wctx.fillRect(0, 0, pw, ph);
  wctx.globalAlpha = 0.35;
  for (let x = 0; x < W + 2 * BORDER; x++) for (let y = 0; y < H + 2 * BORDER; y++) wctx.drawImage(sprites.grass, x * cs, y * cs, cs, cs);
  wctx.globalAlpha = 1;
  // decorative tree border outside the grid (not part of the game map)
  for (let x = 0; x < W + 2 * BORDER; x++) for (let y = 0; y < H + 2 * BORDER; y++)
    if (x < BORDER || y < BORDER || x >= W + BORDER || y >= H + BORDER) wctx.drawImage(sprites.tree, x * cs, y * cs, cs, cs);
  wctx.save();
  wctx.translate(BORDER * cs, BORDER * cs);
  for (const m of state.marks) wctx.drawImage(sprites.mark, m.pos[0] * cs, m.pos[1] * cs, cs, cs);
  for (const [x, y] of state.obstacles) wctx.drawImage(sprites.tree, x * cs, y * cs, cs, cs);
  for (const [x, y] of state.gold) wctx.drawImage(sprites.gold, x * cs, y * cs, cs, cs);
  drawFog(W, H, cs);
  wctx.drawImage(sprites.depot, state.depot[0] * cs, state.depot[1] * cs, cs, cs);   // never fogged
  for (const a of state.agents) {
    const [x, y] = a.pos;
    wctx.drawImage(sprites["miner" + teamIndex(a.team)], x * cs, y * cs, cs, cs);
    // carried gold: one dot per item above the head
    const d = Math.max(2, Math.floor(cs / 6));
    for (let i = 0; i < a.items; i++) {
      wctx.fillStyle = "#111";
      wctx.fillRect(x * cs + i * (d + 1), y * cs, d + 1, d + 1);
      wctx.fillStyle = "#ffd23f";
      wctx.fillRect(x * cs + i * (d + 1), y * cs, d, d);
    }
    if (a.last_result === "teleported" || a.last_result === "pushed") {
      wctx.strokeStyle = "#ffd23f"; wctx.lineWidth = 2;
      wctx.strokeRect(x * cs + 1, y * cs + 1, cs - 2, cs - 2);
    }
  }
  wctx.restore();
}

// Fog of war: every cell outside the 3x3 view of the selected agents is drawn darker.
function drawFog(W, H, cs) {
  const mode = $("fog").value;
  if (mode === "off") return;
  const visible = new Set();
  for (const a of state.agents) {
    if (mode !== "all" && a.team !== mode) continue;
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) visible.add((a.pos[0] + dx) + "," + (a.pos[1] + dy));
  }
  wctx.fillStyle = "rgba(0, 0, 0, 0.38)";
  for (let x = 0; x < W; x++) for (let y = 0; y < H; y++)
    if (!visible.has(x + "," + y)) wctx.fillRect(x * cs, y * cs, cs, cs);
}

function renderPanel() {
  if (!state.running) { $("status").textContent = "no game running (games are started via POST /api/startgame)"; return; }
  const [t0, t1] = state.config.teams;
  $("t0name").textContent = teamLabel(t0); $("t1name").textContent = teamLabel(t1);
  document.querySelectorAll(".legend-row .team0").forEach(el => el.textContent = teamLabel(t0));
  document.querySelectorAll(".legend-row .team1").forEach(el => el.textContent = teamLabel(t1));
  const fog = $("fog");
  const fogKey = [t0, t1, teamLabel(t0), teamLabel(t1)].join(",");
  if (fog.dataset.teams !== fogKey) {          // teams and their names can change
    const keep = fog.value;
    fog.innerHTML = `<option value="all">both teams</option><option value="${esc(t0)}">${esc(teamLabel(t0))} only</option>` +
                    `<option value="${esc(t1)}">${esc(teamLabel(t1))} only</option><option value="off">off</option>`;
    fog.value = [...fog.options].some(o => o.value === keep) ? keep : "all";
    fog.dataset.teams = fogKey;
  }
  $("t0score").textContent = state.score[t0]; $("t1score").textContent = state.score[t1];
  $("progress-bar").style.width = (100 * state.step / state.config.steps) + "%";
  const queued = state.agents.filter(a => a.queued).length;
  $("status").textContent = state.finished
    ? `finished after ${state.step} steps: ${state.winner ? teamLabel(state.winner) + " wins" : "draw"}`
    : `step ${state.step} / ${state.config.steps} · ${state.config.map_type} ${state.config.width}×${state.config.height} · ${queued}/${state.agents.length} actions queued`;

  const carried = state.agents.reduce((s, a) => s + a.items, 0);
  const st = state.stats;
  const failRate = st.actions ? (100 * st.failed / st.actions).toFixed(1) + "%" : "–";
  const tiles = [
    ["gold on map", state.gold.length], ["gold carried", carried], ["gold delivered", st.deliveries],
    ["gold generated", st.generated_gold], ["failed actions", failRate], ["impossible", st.impossible],
    ["pushes", st.pushes], ["teleports", st.teleports], ["picks", st.picks],
  ];
  $("tiles").innerHTML = tiles.map(([k, v]) => `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");

  const tbody = document.querySelector("#agents tbody");
  tbody.innerHTML = state.agents.map(a =>
    `<tr><td><span class="dot" style="background:var(--team${teamIndex(a.team)})"></span>${a.name}</td>` +
    `<td>${a.pos.join(",")}</td><td>${"●".repeat(a.items) || "–"}</td><td>${a.last_action}</td>` +
    `<td class="r-${a.last_result}">${a.last_result}</td><td>${a.queued || ""}</td></tr>`).join("");
  renderChart();
}

// --- score chart (SVG, one axis, crosshair tooltip) ------------------------------------
const chart = $("chart");
const CH = { w: 400, h: 180, l: 34, r: 30, t: 8, b: 20 };

function renderChart() {
  const hist = state.history, teams = state.config.teams;
  const maxStep = Math.max(state.config.steps, 1);
  const maxScore = Math.max(4, ...hist.map(h => Math.max(...teams.map(t => h.score[t]))));
  const mag = 10 ** Math.floor(Math.log10(maxScore / 5));
  const tick = [1, 2, 5, 10].map(f => f * mag).find(t => t * 5 >= maxScore);
  const yTop = Math.ceil(maxScore / tick) * tick;
  const sx = s => CH.l + (CH.w - CH.l - CH.r) * s / maxStep;
  const sy = v => CH.h - CH.b - (CH.h - CH.t - CH.b) * v / yTop;
  let svg = "";
  for (let v = 0; v <= yTop; v += tick) {
    const y = sy(v);
    svg += `<line x1="${CH.l}" x2="${CH.w - CH.r}" y1="${y}" y2="${y}" stroke="var(--grid)" stroke-width="1" vector-effect="non-scaling-stroke"/>`;
    svg += `<text x="${CH.l - 6}" y="${y + 4}" text-anchor="end" font-size="10" fill="var(--text-muted)">${Math.round(v)}</text>`;
  }
  svg += `<text x="${CH.l}" y="${CH.h - 4}" font-size="10" fill="var(--text-muted)">0</text>`;
  svg += `<text x="${CH.w - CH.r}" y="${CH.h - 4}" font-size="10" text-anchor="end" fill="var(--text-muted)">step ${maxStep}</text>`;
  teams.forEach((t, i) => {
    const pts = hist.map(h => `${sx(h.step).toFixed(1)},${sy(h.score[t]).toFixed(1)}`).join(" ");
    svg += `<polyline points="${pts}" fill="none" stroke="var(--team${i})" stroke-width="2" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>`;
    const last = hist[hist.length - 1];
    svg += `<text x="${sx(last.step) + 4}" y="${sy(last.score[t]) + 4 + (i ? 6 : -2)}" font-size="10" fill="var(--text-secondary)">${last.score[t]}</text>`;
  });
  svg += `<line id="xhair" y1="${CH.t}" y2="${CH.h - CH.b}" stroke="var(--text-muted)" stroke-width="1" vector-effect="non-scaling-stroke" visibility="hidden"/>`;
  chart.innerHTML = svg;
  chart._scale = { sx, maxStep };
  $("chart-legend").innerHTML = teams.map((t, i) => `<span><i style="background:var(--team${i})"></i>${esc(teamLabel(t))}</span>`).join("");
}

chart.addEventListener("mousemove", ev => {
  if (!state || !state.running || !chart._scale) return;
  const rect = chart.getBoundingClientRect();
  const xView = (ev.clientX - rect.left) / rect.width * CH.w;
  const step = Math.round((xView - CH.l) / (CH.w - CH.l - CH.r) * chart._scale.maxStep);
  const h = state.history[Math.max(0, Math.min(state.history.length - 1, step))];
  if (!h) return;
  const xh = chart.querySelector("#xhair");
  xh.setAttribute("x1", chart._scale.sx(h.step)); xh.setAttribute("x2", chart._scale.sx(h.step));
  xh.setAttribute("visibility", "visible");
  const tip = $("chart-tip");
  tip.innerHTML = `step ${h.step}<br>` + state.config.teams.map((t, i) =>
    `<span class="dot" style="background:var(--team${i})"></span>${esc(teamLabel(t))}: ${h.score[t]} delivered, ${h.carried[t]} carried`).join("<br>") +
    `<br>gold on map: ${h.gold_on_map}`;
  tip.hidden = false;
  const left = ev.clientX - rect.left;
  tip.style.left = Math.min(left + 12, rect.width - tip.offsetWidth) + "px";
  tip.style.top = "4px";
});
chart.addEventListener("mouseleave", () => {
  $("chart-tip").hidden = true;
  const xh = chart.querySelector("#xhair"); if (xh) xh.setAttribute("visibility", "hidden");
});

// --- cell hover on the world -----------------------------------------------------------
world.addEventListener("mousemove", ev => {
  if (!state || !state.running) return;
  const rect = world.getBoundingClientRect();
  const x = Math.floor((ev.clientX - rect.left) / rect.width * (state.config.width + 2 * BORDER)) - BORDER;
  const y = Math.floor((ev.clientY - rect.top) / rect.height * (state.config.height + 2 * BORDER)) - BORDER;
  if (x < 0 || y < 0 || x >= state.config.width || y >= state.config.height) { $("cell-tip").hidden = true; return; }
  const has = list => list.some(p => p[0] === x && p[1] === y);
  const parts = [];
  for (const a of state.agents) if (a.pos[0] === x && a.pos[1] === y)
    parts.push(`<b>${a.name}</b> carries ${a.items} · last: ${a.last_action} (${a.last_result})`);
  if (state.depot[0] === x && state.depot[1] === y) parts.push("depot");
  if (has(state.gold)) parts.push("gold");
  if (has(state.obstacles)) parts.push("obstacle");
  const m = state.marks.find(m => m.pos[0] === x && m.pos[1] === y);
  if (m) parts.push(`mark "${m.value}"`);
  const tip = $("cell-tip");
  tip.innerHTML = `(${x},${y}) ` + (parts.join("<br>") || "empty");
  tip.hidden = false;
  const fr = world.parentElement.getBoundingClientRect();
  tip.style.left = Math.min(ev.clientX - fr.left + 14, fr.width - tip.offsetWidth - 4) + "px";
  tip.style.top = (ev.clientY - fr.top + 14) + "px";
});
world.addEventListener("mouseleave", () => { $("cell-tip").hidden = true; });

// --- controls ---------------------------------------------------------------------------
async function api(method, path, body) {
  const r = await fetch(path, { method, headers: { "Content-Type": "application/json" },
                                body: body === undefined ? undefined : JSON.stringify(body) });
  const data = await r.json();
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  return data;
}
function showError(e) { const el = $("error"); el.textContent = e ? String(e.message || e) : ""; el.hidden = !e; }

$("step").onclick = async () => { try { await api("POST", "/api/stepgame"); showError(null); await poll(); } catch (e) { showError(e); } };

let autoTimer = null;
$("auto").onclick = () => {
  if (autoTimer) { clearTimeout(autoTimer); autoTimer = null; $("auto").classList.remove("on"); $("auto").textContent = "auto-step ▶"; return; }
  $("auto").classList.add("on"); $("auto").textContent = "auto-step ■";
  const tick = async () => {
    try {
      await api("POST", "/api/stepgame");
      showError(null);
    } catch (e) { showError(e); $("auto").click(); return; }
    autoTimer = setTimeout(tick, Math.max(20, Number($("interval").value) || 200));
  };
  tick();
};

async function poll() {
  try {
    state = await api("GET", "/api/state");
    renderWorld(); renderPanel();
  } catch (e) { $("status").textContent = "server unreachable"; }
}

(async () => {
  $("fog").onchange = renderWorld;
  await loadSprites();
  await poll();
  setInterval(poll, 250);
  window.addEventListener("resize", renderWorld);
})();
