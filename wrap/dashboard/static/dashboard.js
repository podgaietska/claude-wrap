// wrap dashboard: renders /api/stats and /api/requests. No build step;
// Chart.js is vendored and loaded as a global by index.html.

const Chart = window.Chart;
const $ = (id) => document.getElementById(id);

const RANGES = { "1h": 3600, "24h": 86400, "7d": 7 * 86400 };
const TIERS = ["small", "large", "unrouted"];
const TIER_LABELS = { small: "Small tier", large: "Large tier", unrouted: "Unrouted" };
const KIND_LABELS = { new_message: "message", tool_call: "tool call", side: "side" };

const state = { session: "last", range: "", paused: false };
const charts = {};
let timer = null;
let lastUpdated = null;
let lastStats = null;
let lastRequests = null;
// Set while the status line shows a message instead of the refresh time.
let statusNote = null;

// ---------- formatting (matches `wrap stats`) ----------

function money(v) {
  if (v == null) return "—";
  return Math.abs(v) < 10 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`;
}

// Axis ticks: as few decimals as the step needs.
function axisMoney(v) {
  if (v === 0) return "$0";
  return Math.abs(v) < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`;
}

function signedMoney(v) {
  if (v == null) return "—";
  return `${v >= 0 ? "+" : "−"}${money(Math.abs(v))}`;
}

function tokens(n) {
  if (n == null) return "—";
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

function pct(v, digits = 0) {
  return v == null ? "—" : `${(v * 100).toFixed(digits)}%`;
}

function ms(v) {
  if (v == null) return "—";
  return v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(1)} s`;
}

function shortModel(model) {
  if (!model) return "—";
  return model.replace(/^claude-/, "").replace(/-\d{8}$/, "");
}

function localTime(iso, withDate) {
  const d = new Date(iso);
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  return withDate ? `${d.toLocaleDateString([], { month: "short", day: "numeric" })} ${time}` : time;
}

function bucketLabel(iso, bucketSeconds, spansDays) {
  const d = new Date(iso);
  if (bucketSeconds >= 86400) return d.toLocaleDateString([], { month: "short", day: "numeric" });
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return spansDays ? `${d.toLocaleDateString([], { month: "short", day: "numeric" })} ${time}` : time;
}

// ---------- theme ----------

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function colors() {
  return {
    small: css("--tier-small"),
    large: css("--tier-large"),
    unrouted: css("--tier-unrouted"),
    ink: css("--ink"),
    ink2: css("--ink-2"),
    muted: css("--muted"),
    grid: css("--grid"),
    axis: css("--axis"),
    surface: css("--surface"),
    good: css("--good"),
    critical: css("--critical"),
    neutral: css("--neutral-bar"),
  };
}

function applyChartDefaults() {
  const c = colors();
  Chart.defaults.font.family = css("--font");
  Chart.defaults.font.size = 12;
  Chart.defaults.color = c.muted;
  Chart.defaults.borderColor = c.grid;
  Chart.defaults.animation = false;
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.plugins.legend.display = false;
  Object.assign(Chart.defaults.plugins.tooltip, {
    backgroundColor: c.surface,
    titleColor: c.ink,
    bodyColor: c.ink2,
    borderColor: c.axis,
    borderWidth: 1,
    padding: 10,
    boxPadding: 4,
    usePointStyle: true,
  });
}

// Bars: <= 24px, 4px rounded data end, square at the baseline, and a 2px
// surface-coloured gap between stacked segments.
function barStyle(color, surface) {
  return {
    backgroundColor: color,
    borderColor: surface,
    borderWidth: { top: 2 },
    borderSkipped: "start",
    borderRadius: 4,
    maxBarThickness: 24,
  };
}

function axes(c, yFormat) {
  return {
    x: { grid: { display: false }, border: { color: c.axis }, ticks: { color: c.muted, maxRotation: 0, autoSkipPadding: 12 } },
    y: {
      beginAtZero: true,
      grid: { color: c.grid },
      border: { display: false },
      ticks: { color: c.muted, callback: yFormat, maxTicksLimit: 6 },
    },
  };
}

// ---------- state and data ----------

function readState() {
  const params = new URLSearchParams(location.search);
  state.session = params.get("session") || "last";
  state.range = RANGES[params.get("range")] ? params.get("range") : "";
}

function writeState() {
  const params = new URLSearchParams();
  if (state.session !== "last") params.set("session", state.session);
  if (state.range) params.set("range", state.range);
  const query = params.toString();
  history.replaceState(null, "", query ? `?${query}` : location.pathname);
}

function query() {
  const params = new URLSearchParams({ session: state.session });
  if (state.range) params.set("since", new Date(Date.now() - RANGES[state.range] * 1000).toISOString());
  return params.toString();
}

async function getJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw Object.assign(new Error(body.detail || response.statusText), { status: response.status });
  return body;
}

async function load() {
  clearTimeout(timer);
  const q = query();
  let refreshSeconds = 5;
  try {
    // Sessions first, so the picker is filled even if the selected one is unknown.
    renderSessions(await getJson("/api/sessions"));
    const [stats, requests] = await Promise.all([
      getJson(`/api/stats?${q}`),
      getJson(`/api/requests?${q}&limit=200`),
    ]);
    if (stats.empty) {
      showNotice(`No requests logged yet. Run <code>wrap claude</code> and this page fills in as you work.`);
      setStatus(false, "Waiting for a session");
    } else {
      refreshSeconds = stats.scope.refresh_seconds;
      lastStats = stats;
      lastRequests = requests;
      render(stats, requests);
      lastUpdated = Date.now();
      setStatus(stats.summary.live);
    }
    if (!state.paused && !document.hidden && (stats.empty || stats.summary.live)) {
      timer = setTimeout(load, refreshSeconds * 1000);
    }
  } catch (err) {
    if (err.status === 404) {
      showNotice(`Session <code>${escapeHtml(state.session)}</code> has no requests. Pick another session above.`);
      setStatus(false, "Unknown session");
    } else {
      setStatus(false, "Can't reach the dashboard server");
      timer = setTimeout(load, refreshSeconds * 1000);
    }
  }
}

function setStatus(live, text) {
  statusNote = text || null;
  $("status").classList.toggle("live", live && !state.paused);
  if (text) {
    $("status-text").textContent = text;
    return;
  }
  const age = lastUpdated ? Math.round((Date.now() - lastUpdated) / 1000) : null;
  const updated = age == null ? "" : age < 2 ? "updated just now" : `updated ${age} s ago`;
  $("status-text").textContent = live ? (state.paused ? `Paused · ${updated}` : `Live · ${updated}`) : updated;
}

function showNotice(html) {
  $("notice").innerHTML = html;
  $("notice").hidden = false;
  $("content").hidden = true;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ---------- rendering ----------

function render(stats, requests) {
  $("notice").hidden = true;
  $("content").hidden = false;
  const c = colors();
  renderTiles(stats);
  renderWaterfall(stats, c);
  renderTimeseries(stats, c);
  renderHistogram(stats, c);
  renderModels(stats);
  renderCache(stats);
  renderRequests(requests.rows || [], stats.scope.all_sessions);
}

function renderSessions(data) {
  const select = $("session");
  const options = [["last", "Last session"], ["all", "All sessions"]];
  for (const s of data.sessions || []) {
    if (s.id == null) continue;
    options.push([s.id, `${localTime(s.started, true)} · ${s.id} · ${money(s.cost_usd)}`]);
  }
  if (!options.some(([value]) => value === state.session)) options.push([state.session, state.session]);
  const html = options.map(([value, label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`).join("");
  if (select.innerHTML !== html) select.innerHTML = html;
  select.value = state.session;
}

function tile(label, value, detail, cls = "", valueCls = "") {
  return `<div class="tile ${cls}"><div class="label">${label}</div><div class="value ${valueCls}">${value}</div>` +
    `<div class="detail">${detail || "&nbsp;"}</div></div>`;
}

function renderTiles(stats) {
  const s = stats.summary;
  const e = stats.economics;
  const overall = stats.latency[0];
  const share = e.counterfactual_cost ? e.net_savings / e.counterfactual_cost : null;
  const failedDetail = s.interrupted ? `${s.interrupted} stream${s.interrupted === 1 ? "" : "s"} cut short` : "";
  $("tiles").innerHTML = [
    tile("Total cost", money(s.cost_usd), s.unpriced ? `${s.unpriced} unpriced` : "", "hero"),
    tile("Net savings", signedMoney(e.net_savings), share == null ? "" : `${pct(share)} of ${money(e.counterfactual_cost)}`,
      "hero", e.net_savings >= 0 ? "up" : "down"),
    tile("Requests", String(s.requests), `${s.new_messages} messages · ${s.tool_calls} tool calls · ${s.side} side`),
    tile("Small-tier share", pct(stats.routing.small_share), `of ${stats.routing.routed} routed messages`),
    tile("Prompt-cache reads", pct(stats.prompt_cache.read_share), `${tokens(stats.prompt_cache.read_tokens)} of ${tokens(stats.prompt_cache.prompt_tokens)} prompt tokens`),
    tile("p95 time to first byte", ms(overall.ttfb_p95), `p50 ${ms(overall.ttfb_p50)}`),
    tile("Failed", String(s.failed), failedDetail, "", s.failed ? "down" : ""),
  ].join("");
}

// Labels each bar with its value above its top edge.
const valueLabels = {
  id: "valueLabels",
  afterDatasetsDraw(chart, _args, options) {
    const { ctx } = chart;
    const meta = chart.getDatasetMeta(0);
    ctx.save();
    ctx.font = `600 12px ${css("--font")}`;
    ctx.fillStyle = options.color;
    ctx.textAlign = "center";
    meta.data.forEach((bar, i) => {
      const { y, base } = bar.getProps(["y", "base"], true);
      ctx.fillText(options.labels[i], bar.x, Math.min(y, base) - 6);
    });
    ctx.restore();
  },
};

function renderWaterfall(stats, c) {
  const e = stats.economics;
  const cf = e.counterfactual_cost;
  const afterModels = cf - e.saved_by_model;
  const bars = [
    { label: "Would have cost", range: [0, cf], color: c.neutral, value: money(cf) },
    { label: "Cheaper models", range: [afterModels, cf], color: e.saved_by_model >= 0 ? c.good : c.critical, value: signedMoney(-e.saved_by_model) },
    { label: "Cache misses", range: [afterModels, afterModels + e.cache_penalty], color: e.cache_penalty > 0 ? c.critical : c.good, value: signedMoney(e.cache_penalty) },
    { label: "Actually cost", range: [0, e.actual_cost], color: c.neutral, value: money(e.actual_cost) },
  ];
  const baseline = e.baseline ? `always ${shortModel(e.baseline)}` : "always the requested model";
  $("econ-sub").textContent = `Actual cost against ${baseline}, split into what cheaper models saved and what cache misses cost.`;
  const check = e.check_rate == null ? "no requests to check the estimate against yet" :
    `estimate matches on ${pct(e.check_rate)} of ${e.check_eligible} requests without a switch`;
  $("econ-foot").textContent = `${e.switches} model switch${e.switches === 1 ? "" : "es"}, ${tokens(e.recached_tokens)} tokens re-cached; ${check}. ` +
    "Assumes the requested model would give the same output in the same number of requests.";
  const share = cf ? ` (${pct(e.net_savings / cf)})` : "";
  $("waterfall").setAttribute("aria-label", `Net savings ${signedMoney(e.net_savings)}${share}`);

  const data = {
    labels: bars.map((b) => b.label),
    datasets: [{ data: bars.map((b) => b.range), ...barStyle(null, c.surface), maxBarThickness: 56, backgroundColor: bars.map((b) => b.color) }],
  };
  const options = {
    layout: { padding: { top: 22 } },
    scales: { ...axes(c, axisMoney), x: { ...axes(c).x, ticks: { ...axes(c).x.ticks, autoSkip: false } } },
    plugins: {
      valueLabels: { color: c.ink, labels: bars.map((b) => b.value) },
      tooltip: { callbacks: { label: (item) => bars[item.dataIndex].value } },
    },
  };
  upsert("waterfall", "bar", data, options, [valueLabels]);
}

function renderTimeseries(stats, c) {
  const ts = stats.timeseries;
  const points = ts.points;
  const spansDays = points.length > 1 && new Date(points.at(-1).t) - new Date(points[0].t) >= 86400 * 1000;
  const minutes = ts.bucket_seconds / 60;
  const bucket = minutes < 60 ? `${minutes}-minute` : minutes < 1440 ? `${minutes / 60}-hour` : `${minutes / 1440}-day`;
  $("time-sub").textContent = `Actual cost per tier in ${bucket} buckets, against what the same requests would have cost on the requested model.`;
  const usedTiers = TIERS.filter((t) => points.some((p) => p[`cost_${t}`] > 0));
  const tiers = usedTiers.length ? usedTiers : ["small", "large"];
  $("time-legend").innerHTML = tiers.map((t) => `<li><span class="swatch" style="background:${c[t]}"></span>${TIER_LABELS[t]}</li>`).join("") +
    `<li><span class="swatch line" style="background:${c.ink2}"></span>Would have cost</li>`;
  const total = points.reduce((sum, p) => sum + p.cost_small + p.cost_large + p.cost_unrouted, 0);
  $("timeseries").setAttribute("aria-label", `Cost over time, ${money(total)} in total`);

  const data = {
    labels: points.map((p) => bucketLabel(p.t, ts.bucket_seconds, spansDays)),
    datasets: [
      ...tiers.map((t) => ({
        type: "bar",
        label: TIER_LABELS[t],
        data: points.map((p) => p[`cost_${t}`]),
        stack: "cost",
        order: 2,
        ...barStyle(c[t], c.surface),
      })),
      {
        type: "line",
        label: "Would have cost",
        data: points.map((p) => p.counterfactual),
        stack: "counterfactual",
        order: 1,
        borderColor: c.ink2,
        backgroundColor: c.ink2,
        borderWidth: 2,
        pointRadius: points.length > 40 ? 0 : 3,
        pointHoverRadius: 5,
        tension: 0,
      },
    ],
  };
  const options = {
    interaction: { mode: "index", intersect: false },
    scales: { x: { ...axes(c).x, stacked: true }, y: { ...axes(c, axisMoney).y, stacked: true } },
    plugins: {
      tooltip: {
        callbacks: {
          label: (item) => `${item.dataset.label}: ${money(item.raw)}`,
          footer: (items) => {
            const p = points[items[0].dataIndex];
            return `${p.requests} request${p.requests === 1 ? "" : "s"}${p.switches ? `, ${p.switches} switch${p.switches === 1 ? "" : "es"}` : ""}`;
          },
        },
      },
    },
  };
  upsert("timeseries", "bar", data, options);
}

// Draws the routing threshold as a vertical line on the linear x axis.
const thresholdLine = {
  id: "thresholdLine",
  afterDatasetsDraw(chart, _args, options) {
    const { ctx, chartArea, scales } = chart;
    const x = scales.x.getPixelForValue(options.value);
    ctx.save();
    ctx.strokeStyle = options.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, chartArea.top);
    ctx.lineTo(x, chartArea.bottom);
    ctx.stroke();
    ctx.fillStyle = options.color;
    ctx.font = `12px ${css("--font")}`;
    ctx.textAlign = "left";
    ctx.fillText(`threshold ${options.value}`, x + 6, chartArea.top + 12);
    ctx.restore();
  },
};

function renderHistogram(stats, c) {
  const r = stats.routing;
  $("routing-sub").textContent = r.routed
    ? `Complexity scores of ${r.routed} new messages. At or above the threshold routes to the large tier.`
    : "No routed messages in this selection yet.";
  $("routing-legend").innerHTML = ["small", "large"]
    .map((t) => `<li><span class="swatch" style="background:${c[t]}"></span>${TIER_LABELS[t]}</li>`).join("");
  $("histogram").setAttribute("aria-label", `${pct(r.small_share)} of ${r.routed} messages routed to the small tier`);

  const centre = (b) => (b.lo + b.hi) / 2;
  const data = {
    datasets: ["small", "large"].map((t) => ({
      label: TIER_LABELS[t],
      data: r.histogram.map((b) => ({ x: centre(b), y: b[t] })),
      ...barStyle(c[t], c.surface),
      barPercentage: 1,
      categoryPercentage: 0.9,
    })),
  };
  const options = {
    scales: {
      x: { type: "linear", min: 0, max: 1, stacked: true, grid: { display: false }, border: { color: c.axis },
        ticks: { stepSize: 0.1, color: c.muted, callback: (v) => v.toFixed(1) }, title: { display: true, text: "complexity score", color: c.muted } },
      y: { ...axes(c, (v) => (Number.isInteger(v) ? v : null)).y, stacked: true },
    },
    plugins: {
      thresholdLine: { value: r.threshold, color: c.ink2 },
      tooltip: {
        callbacks: {
          title: (items) => {
            const b = r.histogram[items[0].dataIndex];
            return `score ${b.lo.toFixed(2)}–${b.hi.toFixed(2)}`;
          },
          label: (item) => `${item.dataset.label}: ${item.raw.y}`,
        },
      },
    },
  };
  upsert("histogram", "bar", data, options, [thresholdLine]);
}

function upsert(id, type, data, options, plugins = []) {
  const existing = charts[id];
  if (existing && existing.config.type === type) {
    existing.data = data;
    existing.options = options;
    existing.update("none");
    return;
  }
  existing?.destroy();
  charts[id] = new Chart($(id), { type, data, options, plugins });
}

function cell(text, cls = "") {
  const td = document.createElement("td");
  if (cls) td.className = cls;
  if (text instanceof Node) td.append(text);
  else td.textContent = text;
  return td;
}

function header(table, columns) {
  const thead = document.createElement("thead");
  const tr = document.createElement("tr");
  for (const [label, numeric] of columns) {
    const th = document.createElement("th");
    th.textContent = label;
    if (numeric) th.className = "num";
    tr.append(th);
  }
  thead.append(tr);
  table.replaceChildren(thead);
}

// A tier with its colour swatch, and optionally the model it served, shortened.
function tierCell(tier, model) {
  const span = document.createElement("span");
  span.className = "tier";
  const known = TIERS.includes(tier) ? tier : "unrouted";
  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.background = css(`--tier-${known}`);
  span.append(swatch, tier || "—");
  if (model !== undefined) {
    span.append(` · ${shortModel(model)}`);
    span.title = model || "";
  }
  return span;
}

function renderModels(stats) {
  const table = $("models");
  header(table, [["Tier · model"], ["Requests", 1], ["In / out", 1], ["Cache r / w", 1], ["Cache reads", 1],
    ["Cost", 1], ["TTFB p50 / p95", 1], ["Total p50 / p95", 1]]);
  const latency = Object.fromEntries(stats.latency.slice(1).map((l) => [l.model, l]));
  const tbody = document.createElement("tbody");
  for (const m of stats.models) {
    const l = latency[m.served_model];
    const tr = document.createElement("tr");
    tr.append(
      cell(tierCell(m.tier, m.served_model)),
      cell(String(m.requests), "num"),
      cell(`${tokens(m.input_tokens)} / ${tokens(m.output_tokens)}`, "num"),
      cell(`${tokens(m.cache_read_tokens)} / ${tokens(m.cache_creation_tokens)}`, "num"),
      cell(pct(m.cache_read_share), "num"),
      cell(money(m.cost_usd), "num"),
      cell(l ? `${ms(l.ttfb_p50)} / ${ms(l.ttfb_p95)}` : "—", "num"),
      cell(l ? `${ms(l.latency_p50)} / ${ms(l.latency_p95)}` : "—", "num"),
    );
    tbody.append(tr);
  }
  table.append(tbody);
}

function renderCache(stats) {
  $("cache-card").hidden = !stats.cache;
  if (stats.cache) {
    $("cache-sub").textContent = `${stats.cache.hits} responses served from the cache (${pct(stats.cache.hit_rate)} of new messages).`;
  }
}

function renderRequests(rows, allSessions) {
  const table = $("requests");
  header(table, [["Time"], ["Kind"], ["Tier"], ["Score", 1], ["Model"], ["Input", 1], ["Output", 1], ["Cache r / w", 1],
    ["Cost", 1], ["Net", 1], ["TTFB", 1], ["Latency", 1], ["Status", 1]]);
  const tbody = document.createElement("tbody");
  for (const r of rows) {
    const failed = r.status_code == null || r.status_code >= 400;
    const tr = document.createElement("tr");
    if (failed) tr.classList.add("failed");
    const kind = document.createElement("span");
    kind.className = "badge";
    kind.textContent = KIND_LABELS[r.kind] || r.kind;
    const model = document.createElement("span");
    model.textContent = shortModel(r.served_model);
    if (r.requested_model && r.requested_model !== r.served_model) {
      model.title = `requested ${r.requested_model}`;
      const asked = document.createElement("span");
      asked.className = "dim";
      asked.textContent = ` (asked ${shortModel(r.requested_model)})`;
      model.append(asked);
    }
    if (r.switched) {
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = "switch";
      badge.title = "Served by a different model than the previous request in this conversation";
      model.append(" ", badge);
    }
    const net = cell(r.net_savings == null ? "—" : signedMoney(r.net_savings), "num");
    if (r.net_savings != null && Math.abs(r.net_savings) >= 0.00005) net.classList.add(r.net_savings > 0 ? "pos" : "neg");
    tr.append(
      cell(localTime(r.timestamp, allSessions)),
      cell(kind),
      cell(tierCell(r.tier)),
      cell(r.score == null ? "—" : r.score.toFixed(2), "num"),
      cell(model),
      cell(tokens(r.input_tokens), "num"),
      cell(tokens(r.output_tokens), "num"),
      cell(`${tokens(r.cache_read_tokens)} / ${tokens(r.cache_creation_tokens)}`, "num"),
      cell(money(r.cost_usd), "num"),
      net,
      cell(ms(r.ttfb_ms), "num"),
      cell(ms(r.latency_ms), "num"),
      cell(r.status_code == null ? "—" : String(r.status_code) + (r.error && !failed ? " (cut short)" : ""), "num"),
    );
    tbody.append(tr);
    if (r.error) {
      tr.classList.add("clickable");
      tr.title = "Click to show the error";
      tr.addEventListener("click", () => {
        const next = tr.nextElementSibling;
        if (next?.classList.contains("error-detail")) {
          next.remove();
          return;
        }
        const detail = document.createElement("tr");
        detail.className = "error-detail";
        const td = cell(r.error);
        td.colSpan = 13;
        detail.append(td);
        tr.after(detail);
      });
    }
  }
  table.append(tbody);
}

// ---------- wiring ----------

function init() {
  readState();
  applyChartDefaults();
  $("range").value = state.range;

  $("session").addEventListener("change", (e) => {
    state.session = e.target.value;
    writeState();
    load();
  });
  $("range").addEventListener("change", (e) => {
    state.range = e.target.value;
    writeState();
    load();
  });
  $("pause").addEventListener("click", () => {
    state.paused = !state.paused;
    $("pause").textContent = state.paused ? "Resume" : "Pause";
    if (state.paused) clearTimeout(timer);
    else load();
    setStatus(lastStats?.summary?.live ?? false);
  });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && !state.paused) load();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    applyChartDefaults();
    for (const id of Object.keys(charts)) {
      charts[id].destroy();
      delete charts[id];
    }
    if (lastStats) render(lastStats, lastRequests);
  });
  setInterval(() => {
    if (lastStats && !statusNote) setStatus(lastStats.summary.live);
  }, 1000);

  load();
}

init();
