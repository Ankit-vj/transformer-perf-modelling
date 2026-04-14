// Save as: C:\Users\ankit\transformer-perf-model\web\static\js\main.js

"use strict";

/* ══════════════════════════════════════════════════════
   STATE
══════════════════════════════════════════════════════ */
const state = {
  selectedModel:    "bert-base",
  selectedHardware: "mid-range",
  lastResults:      null,
  reportText:       "",
};

/* ══════════════════════════════════════════════════════
   INIT — fetch available options from API
══════════════════════════════════════════════════════ */
document.addEventListener("DOMContentLoaded", async () => {
  await loadModels();
  await loadHardware();
});

async function loadModels() {
  try {
    const res  = await fetch("/api/models");
    const data = await res.json();
    renderModelGrid(data.models);
  } catch (e) {
    console.error("Failed to load models:", e);
  }
}

async function loadHardware() {
  try {
    const res  = await fetch("/api/hardware");
    const data = await res.json();
    renderHwGrid(data.hardware);
  } catch (e) {
    console.error("Failed to load hardware:", e);
  }
}

/* ══════════════════════════════════════════════════════
   RENDER: Model Grid
══════════════════════════════════════════════════════ */
function renderModelGrid(models) {
  const grid = document.getElementById("model-grid");
  grid.innerHTML = "";

  Object.entries(models).forEach(([key, m]) => {
    const card = document.createElement("div");
    card.className = "model-card" + (key === state.selectedModel ? " selected" : "");
    card.dataset.key = key;
    card.innerHTML = `
      <div class="mc-icon">${m.icon}</div>
      <div class="mc-name">${m.name}</div>
      <div class="mc-cat">${m.category}</div>
      <div class="mc-params">${m.params_m}M params</div>
    `;
    card.onclick = () => selectModel(key);
    grid.appendChild(card);
  });
}

function selectModel(key) {
  state.selectedModel = key;
  document.querySelectorAll(".model-card").forEach(c => {
    c.classList.toggle("selected", c.dataset.key === key);
  });
}

/* ══════════════════════════════════════════════════════
   RENDER: Hardware Grid
══════════════════════════════════════════════════════ */
function renderHwGrid(hardware) {
  const grid = document.getElementById("hw-grid");
  grid.innerHTML = "";

  Object.entries(hardware).forEach(([key, h]) => {
    const card = document.createElement("div");
    card.className = "hw-card" + (key === state.selectedHardware ? " selected" : "");
    card.dataset.key = key;
    card.innerHTML = `
      <span class="hw-icon">${h.icon}</span>
      <div>
        <div class="hw-name">${h.name}</div>
        <div class="hw-desc">${h.description}</div>
      </div>
    `;
    card.onclick = () => selectHardware(key);
    grid.appendChild(card);
  });
}

function selectHardware(key) {
  state.selectedHardware = key;
  document.querySelectorAll(".hw-card").forEach(c => {
    c.classList.toggle("selected", c.dataset.key === key);
  });
}

/* ══════════════════════════════════════════════════════
   RUN ANALYSIS
══════════════════════════════════════════════════════ */
async function runAnalysis() {
  const btn = document.getElementById("run-btn");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner" style="width:20px;height:20px;border-width:3px"></span> Running...`;

  // Gather quant targets
  const quantTargets = [];
  if (document.getElementById("qt-fp16").checked) quantTargets.push("fp16");
  if (document.getElementById("qt-int8").checked) quantTargets.push("int8");
  if (document.getElementById("qt-int4").checked) quantTargets.push("int4");

  // Build request payload
  const payload = {
    model:            state.selectedModel,
    hardware:         state.selectedHardware,
    batch_size:       parseInt(document.getElementById("batch-size").value),
    seq_len:          parseInt(document.getElementById("seq-len").value),
    num_cores:        parseInt(document.getElementById("num-cores").value),
    dtype:            document.getElementById("dtype").value,
    run_quantization: document.getElementById("opt-quant").checked,
    run_pruning:      document.getElementById("opt-prune").checked,
    run_fusion:       document.getElementById("opt-fusion").checked,
    run_sweep:        document.getElementById("opt-sweep").checked,
    quant_targets:    quantTargets,
    sparsity_values:  [0.3, 0.5, 0.7, 0.9],
  };

  // Show loading
  showLoading();

  // Animate progress steps
  const steps = ["step-parse","step-hw","step-lat","step-pow","step-opt","step-viz"];
  let si = 0;
  const interval = setInterval(() => {
    if (si > 0) {
      document.getElementById(steps[si-1]).className = "step done";
    }
    if (si < steps.length) {
      document.getElementById(steps[si]).className = "step active";
      si++;
    } else {
      clearInterval(interval);
    }
  }, 1200);

  try {
    const res  = await fetch("/api/analyze", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    clearInterval(interval);

    const data = await res.json();

    if (data.status === "ok") {
      state.lastResults = data;
      state.reportText  = data.report_text || "";
      renderResults(data);
    } else {
      showError(data.message, data.traceback);
    }
  } catch (e) {
    clearInterval(interval);
    showError(e.message, "");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span class="run-icon">▶</span> Generate Results`;
  }
}

/* ══════════════════════════════════════════════════════
   SHOW LOADING / ERROR
══════════════════════════════════════════════════════ */
function showLoading() {
  document.getElementById("placeholder").style.display      = "none";
  document.getElementById("results-content").style.display  = "none";
  document.getElementById("loading").style.display          = "flex";

  // Reset steps
  ["step-parse","step-hw","step-lat","step-pow","step-opt","step-viz"].forEach(id => {
    document.getElementById(id).className = "step";
  });
}

function showError(msg, tb) {
  document.getElementById("loading").style.display = "none";
  document.getElementById("placeholder").innerHTML = `
    <div class="placeholder-icon">❌</div>
    <h2>Analysis Failed</h2>
    <p style="color:#ef4444">${msg}</p>
    <pre style="background:#1a1d27;padding:1rem;border-radius:8px;font-size:.7rem;
                color:#f87171;text-align:left;max-width:600px;overflow:auto">${tb}</pre>
  `;
  document.getElementById("placeholder").style.display = "flex";
}

/* ══════════════════════════════════════════════════════
   RENDER ALL RESULTS
══════════════════════════════════════════════════════ */
function renderResults(data) {
  document.getElementById("loading").style.display        = "none";
  document.getElementById("placeholder").style.display    = "none";
  document.getElementById("results-content").style.display = "block";

  // Summary bar
  document.getElementById("sum-model").textContent      = data.model_info.name;
  document.getElementById("sum-hw").textContent         = data.hardware_info.name;
  document.getElementById("sum-latency").textContent    = data.latency.total_ms.toFixed(2);
  document.getElementById("sum-throughput").textContent = fmt(data.latency.tokens_per_sec);
  document.getElementById("sum-power").textContent      = data.energy.average_power_w.toFixed(3);
  document.getElementById("sum-util").textContent       = data.latency.hw_utilization_pct.toFixed(1);

  // Tab data
  renderOverview(data);
  renderLatency(data);
  renderEnergy(data);
  renderRoofline(data);
  renderOptimizations(data);
  renderCharts(data);
  renderReport(data);

  // Activate first tab
  showTab("tab-overview");
}

/* ── Overview Tab ─────────────────────────────────────── */
function renderOverview(data) {
  // Model info table
  tableFromObj("model-info-table", {
    "Model Name":     data.model_info.name,
    "Layers (nodes)": data.model_info.num_nodes,
    "Total FLOPs":    `${data.model_info.total_flops_g} GFLOPs`,
    "Weight Memory":  `${data.model_info.weight_mb} MB`,
    "Total Memory":   `${data.model_info.memory_mb} MB`,
    "Data Type":      data.config.dtype.toUpperCase(),
    "Batch Size":     data.config.batch_size,
    "Seq Length":     data.config.seq_len,
  });

  // Hardware info table
  tableFromObj("hw-info-table", {
    "SoC Name":       data.hardware_info.name,
    "Frequency":      `${data.hardware_info.frequency_ghz} GHz`,
    "Cores":          data.hardware_info.num_cores,
    "VLEN":           `${data.hardware_info.vlen} bits`,
    "Peak GFLOP/s":   `${data.hardware_info.peak_gflops} GFLOP/s`,
    "DRAM BW":        `${data.hardware_info.dram_bw_gbps} GB/s`,
    "Ridge Point":    `${data.hardware_info.ridge_point} FLOP/byte`,
  });

  // Performance metrics
  tableFromObj("perf-table", {
    "Total Latency":     `${data.latency.total_ms} ms`,
    "Total Cycles":      data.latency.total_cycles.toLocaleString(),
    "Throughput":        `${data.latency.tokens_per_sec} tokens/s`,
    "Achieved GFLOP/s":  `${data.latency.achieved_gflops}`,
    "Peak GFLOP/s":      `${data.latency.peak_gflops}`,
    "HW Utilization":    `${data.latency.hw_utilization_pct}%`,
    "Total Energy":      `${data.energy.total_mj} mJ`,
    "Average Power":     `${data.energy.average_power_w} W`,
    "Efficiency":        `${data.energy.gflops_per_watt} GFLOP/s/W`,
    "Energy/Token":      `${data.energy.per_token_uj} µJ`,
  });

  // SoC comparison table
  const tbody = document.getElementById("soc-tbody");
  tbody.innerHTML = "";
  data.soc_comparison.forEach(s => {
    const highlight = s.preset === data.config.hardware ? "style='background:rgba(59,130,246,.1)'" : "";
    tbody.innerHTML += `
      <tr ${highlight}>
        <td><strong>${s.soc_name}</strong></td>
        <td>${s.latency_ms}</td>
        <td>${s.gflops}</td>
        <td>${s.power_w}</td>
        <td>${s.efficiency}</td>
      </tr>`;
  });
}

/* ── Latency Tab ──────────────────────────────────────── */
function renderLatency(data) {
  const lat = data.latency;

  document.getElementById("lat-total").textContent  = lat.total_ms;
  document.getElementById("lat-tps").textContent    = fmt(lat.tokens_per_sec);
  document.getElementById("lat-gflops").textContent = lat.achieved_gflops;
  document.getElementById("lat-util").textContent   = lat.hw_utilization_pct;

  // Progress bar
  const pct = Math.min(lat.hw_utilization_pct, 100);
  document.getElementById("util-bar").style.width    = pct + "%";
  document.getElementById("util-bar-label").textContent = pct + "%";

  // Layer table
  const tbody = document.getElementById("layer-tbody");
  tbody.innerHTML = "";
  data.layer_breakdown.forEach((l, i) => {
    const tag = l.bound === "Compute"
      ? `<span class="tag-compute">Compute</span>`
      : `<span class="tag-memory">Memory</span>`;
    tbody.innerHTML += `
      <tr>
        <td>${i + 1}</td>
        <td style="font-size:.7rem;font-family:monospace">${l.name}</td>
        <td>${l.type}</td>
        <td>${l.cycles.toLocaleString()}</td>
        <td>${l.latency_ms}</td>
        <td>${(l.flops / 1e6).toFixed(1)}M</td>
        <td>${tag}</td>
        <td>${l.gflops}</td>
      </tr>`;
  });

  // Sweep table
  const stbody = document.getElementById("sweep-tbody");
  stbody.innerHTML = "";
  data.throughput_sweep.forEach(r => {
    stbody.innerHTML += `
      <tr>
        <td><strong>${r.batch_size}</strong></td>
        <td>${r.latency_ms}</td>
        <td>${fmt(r.tokens_per_second)}</td>
        <td>${r.samples_per_second}</td>
        <td>${r.achieved_gflops}</td>
        <td>${r.hw_utilization}%</td>
        <td>${r.memory_usage_mb}</td>
      </tr>`;
  });
}

/* ── Energy Tab ───────────────────────────────────────── */
function renderEnergy(data) {
  const e = data.energy;

  document.getElementById("eng-total").textContent = e.total_mj;
  document.getElementById("eng-power").textContent = e.average_power_w;
  document.getElementById("eng-eff").textContent   = e.gflops_per_watt;
  document.getElementById("eng-token").textContent = e.per_token_uj;

  // Power breakdown table
  const tbody = document.getElementById("power-tbody");
  tbody.innerHTML = "";
  const pb = e.power_breakdown;
  const total = pb.total_power_w;

  const components = [
    ["Core Dynamic",  pb.core_dynamic_w],
    ["Core Leakage",  pb.core_leakage_w],
    ["Vector Unit",   pb.vector_unit_w],
    ["Cache",         pb.cache_power_w],
    ["DRAM",          pb.dram_power_w],
    ["NoC",           pb.noc_power_w],
  ];

  components.forEach(([name, w]) => {
    const pct = total > 0 ? ((w / total) * 100).toFixed(1) : "0.0";
    tbody.innerHTML += `
      <tr>
        <td>${name}</td>
        <td>${w.toFixed(3)}</td>
        <td>${(w * 1000).toFixed(1)}</td>
        <td>
          <div style="display:flex;align-items:center;gap:.4rem">
            <div style="flex:1;background:var(--surface);border-radius:4px;height:6px;overflow:hidden">
              <div style="width:${pct}%;background:var(--blue);height:100%;border-radius:4px"></div>
            </div>
            <span>${pct}%</span>
          </div>
        </td>
      </tr>`;
  });

  // Total row
  tbody.innerHTML += `
    <tr style="font-weight:700;border-top:1px solid var(--border)">
      <td>TOTAL</td>
      <td>${total.toFixed(3)}</td>
      <td>${(total * 1000).toFixed(1)}</td>
      <td>100%</td>
    </tr>`;
}

/* ── Roofline Tab ─────────────────────────────────────── */
function renderRoofline(data) {
  const r = data.roofline;

  document.getElementById("roof-peak").textContent  = r.peak_gflops;
  document.getElementById("roof-bw").textContent    = r.peak_bw_gbps;
  document.getElementById("roof-ridge").textContent = r.ridge_point;
  document.getElementById("roof-comp").textContent  = r.compute_bound;
  document.getElementById("roof-mem").textContent   = r.memory_bound;

  // Validation table
  const tbody = document.getElementById("val-tbody");
  tbody.innerHTML = "";
  data.validation.forEach(v => {
    const tag = v.within_tolerance
      ? `<span class="tag-pass">✓ OK</span>`
      : `<span class="tag-fail">✗ FAIL</span>`;
    tbody.innerHTML += `
      <tr>
        <td>${v.metric}</td>
        <td>${v.estimated}</td>
        <td>${v.reference}</td>
        <td>${v.error_pct}%</td>
        <td>${tag}</td>
        <td>${v.confidence}</td>
      </tr>`;
  });
}

/* ── Optimizations Tab ────────────────────────────────── */
function renderOptimizations(data) {
  // Quantization
  const qbody = document.getElementById("quant-tbody");
  qbody.innerHTML = "";
  data.quantization.forEach(q => {
    qbody.innerHTML += `
      <tr>
        <td><strong>${q.target_dtype.toUpperCase()}</strong></td>
        <td>${q.compression_ratio}x</td>
        <td>${q.memory_savings_pct}%</td>
        <td>${q.speedup}x</td>
        <td>${q.accuracy_drop_pct}%</td>
        <td>${q.original_mb} MB</td>
        <td>${q.quantized_mb} MB</td>
      </tr>`;
  });

  // Pruning
  const pbody = document.getElementById("prune-tbody");
  pbody.innerHTML = "";
  data.pruning.forEach(p => {
    pbody.innerHTML += `
      <tr>
        <td>${p.sparsity}%</td>
        <td>${p.speedup}x</td>
        <td>${p.memory_savings_pct}%</td>
        <td>${p.flop_reduction_pct}%</td>
        <td>${p.accuracy_drop_pct}%</td>
      </tr>`;
  });

  // Fusion
  if (data.fusion) {
    const f = data.fusion;
    document.getElementById("fusion-metrics").innerHTML = `
      <div class="metric-card blue">
        <div class="metric-icon">🔗</div>
        <div class="metric-label">Opportunities</div>
        <div class="metric-value">${f.opportunities}</div>
      </div>
      <div class="metric-card green">
        <div class="metric-icon">⚡</div>
        <div class="metric-label">Overall Speedup</div>
        <div class="metric-value">${f.overall_speedup}x</div>
      </div>
      <div class="metric-card orange">
        <div class="metric-icon">💾</div>
        <div class="metric-label">Mem Savings</div>
        <div class="metric-value">${f.memory_savings_mb}</div>
        <div class="metric-unit">MB</div>
      </div>
    `;

    const fbody = document.getElementById("fusion-tbody");
    fbody.innerHTML = "";
    f.fusions.forEach(fu => {
      fbody.innerHTML += `
        <tr>
          <td style="font-size:.7rem;font-family:monospace">${fu.name}</td>
          <td>${fu.type}</td>
          <td>${fu.speedup}x</td>
        </tr>`;
    });
  }
}

/* ── Charts Tab ───────────────────────────────────────── */
function renderCharts(data) {
  const grid = document.getElementById("charts-grid");
  grid.innerHTML = "";

  const chartDefs = [
    { key: "latency_breakdown",  title: "Latency Breakdown" },
    { key: "roofline",           title: "Roofline Model" },
    { key: "layer_timeline",     title: "Layer Timeline" },
    { key: "power_breakdown",    title: "Power Breakdown" },
    { key: "throughput_scaling", title: "Throughput Scaling" },
    { key: "soc_comparison",     title: "SoC Comparison" },
  ];

  chartDefs.forEach(({ key, title }) => {
    const b64 = data.images[key];
    if (!b64) return;

    const card = document.createElement("div");
    card.className = "chart-card";
    card.innerHTML = `
      <h4>${title}</h4>
      <img src="data:image/png;base64,${b64}" alt="${title}" loading="lazy" />
    `;
    grid.appendChild(card);
  });

  if (grid.children.length === 0) {
    grid.innerHTML = `
      <p style="color:var(--text-dim);padding:2rem;grid-column:1/-1">
        No charts were generated.
      </p>`;
  }
}

/* ── Report Tab ───────────────────────────────────────── */
function renderReport(data) {
  document.getElementById("report-text").textContent = data.report_text || "No report generated.";
}

/* ══════════════════════════════════════════════════════
   TAB SWITCHING
══════════════════════════════════════════════════════ */
function showTab(tabId) {
  // Deactivate all tabs and content
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

  // Activate selected
  const idx = ["tab-overview","tab-latency","tab-energy",
                "tab-roofline","tab-optim","tab-charts","tab-report"]
    .indexOf(tabId);
  const tabs = document.querySelectorAll(".tab");
  if (idx >= 0 && tabs[idx]) tabs[idx].classList.add("active");

  const content = document.getElementById(tabId);
  if (content) content.classList.add("active");
}

/* ══════════════════════════════════════════════════════
   UTILITIES
══════════════════════════════════════════════════════ */
function tableFromObj(tableId, obj) {
  const tbl = document.getElementById(tableId);
  if (!tbl) return;
  tbl.innerHTML = Object.entries(obj)
    .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`)
    .join("");
}

function fmt(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return n.toString();
}

function downloadReport() {
  if (!state.reportText) return;
  const blob = new Blob([state.reportText], { type: "text/plain" });
  const a    = document.createElement("a");
  a.href     = URL.createObjectURL(blob);
  a.download = "transformer_perf_report.txt";
  a.click();
}

function copyReport() {
  if (!state.reportText) return;
  navigator.clipboard.writeText(state.reportText)
    .then(() => alert("Report copied to clipboard!"))
    .catch(() => alert("Copy failed — please select and copy manually."));
}