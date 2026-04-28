# Save as: C:\Users\ankit\transformer-perf-model\web\app.py
# COMPLETE FILE - copy everything

"""
Flask Web Server for Transformer Performance Modeling Framework.
"""

import sys
import os
import base64
import traceback
import uuid
from datetime import datetime

# ── Path Setup (works both locally and on Railway) ─────────
# Get the directory where app.py lives (web/)
WEB_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the project root (one level up from web/)
PROJECT_ROOT = os.path.dirname(WEB_DIR)

# Add project root to Python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Results directory
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"[STARTUP] Project root : {PROJECT_ROOT}")
print(f"[STARTUP] Results dir  : {RESULTS_DIR}")
print(f"[STARTUP] Python path  : {sys.path[:3]}")

# ── Flask imports ──────────────────────────────────────────
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Framework imports (with error handling) ────────────────
FRAMEWORK_OK = True
framework_error = ""

try:
    from transformer_perf.models.parser import TransformerGraphParser
    from transformer_perf.hardware.soc import RISCVSoC
    from transformer_perf.estimators.latency import LatencyEstimator
    from transformer_perf.estimators.throughput import ThroughputEstimator
    from transformer_perf.estimators.power import PowerEstimator
    from transformer_perf.estimators.roofline import RooflineModel
    from transformer_perf.optimizations.quantization import QuantizationAnalyzer
    from transformer_perf.optimizations.pruning import PruningAnalyzer
    from transformer_perf.optimizations.fusion import FusionAnalyzer
    from transformer_perf.validation.calibration import CalibrationEngine
    from transformer_perf.visualization.dashboard import PerformanceDashboard
    print("[STARTUP] Framework imports: OK")
except Exception as e:
    FRAMEWORK_OK = False
    framework_error = str(e)
    print(f"[STARTUP] Framework import ERROR: {e}")
    traceback.print_exc()

# ── Flask App ──────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(WEB_DIR, "templates"),
    static_folder=os.path.join(WEB_DIR, "static"),
)
CORS(app)

# ── Model Options ──────────────────────────────────────────
MODEL_OPTIONS = {
    "bert-base": {
        "name": "BERT Base",
        "description": "12-layer, 768-hidden, 12-heads, 110M params",
        "category": "Encoder",
        "params_m": 110,
        "icon": "🤖",
    },
    "bert-large": {
        "name": "BERT Large",
        "description": "24-layer, 1024-hidden, 16-heads, 340M params",
        "category": "Encoder",
        "params_m": 340,
        "icon": "🤖",
    },
    "gpt2-small": {
        "name": "GPT-2 Small",
        "description": "12-layer, 768-hidden, 12-heads, 117M params",
        "category": "Decoder",
        "params_m": 117,
        "icon": "📝",
    },
    "gpt2-medium": {
        "name": "GPT-2 Medium",
        "description": "24-layer, 1024-hidden, 16-heads, 345M params",
        "category": "Decoder",
        "params_m": 345,
        "icon": "📝",
    },
    "llama-7b": {
        "name": "LLaMA 7B",
        "description": "32-layer, 4096-hidden, 32-heads, 7B params",
        "category": "Decoder",
        "params_m": 7000,
        "icon": "🦙",
    },
    "vit-base": {
        "name": "ViT Base",
        "description": "12-layer, 768-hidden, 12-heads, Vision Transformer",
        "category": "Vision",
        "params_m": 86,
        "icon": "👁️",
    },
}

# ── Hardware Options ───────────────────────────────────────
SOC_OPTIONS = {
    "minimal": {
        "name": "Minimal RISC-V",
        "description": "1 core, 1.0GHz, VLEN=128, DDR4 2GB",
        "cores": 1, "frequency_ghz": 1.0,
        "vlen": 128, "dram": "DDR4 2GB",
        "icon": "🔹",
    },
    "mid-range": {
        "name": "Mid-Range SoC",
        "description": "4 cores, 2.0GHz, VLEN=256, DDR4 8GB",
        "cores": 4, "frequency_ghz": 2.0,
        "vlen": 256, "dram": "DDR4 8GB",
        "icon": "🔷",
    },
    "high-perf": {
        "name": "High Performance SoC",
        "description": "8 cores, 3.0GHz, VLEN=512, DDR5 32GB",
        "cores": 8, "frequency_ghz": 3.0,
        "vlen": 512, "dram": "DDR5 32GB",
        "icon": "💎",
    },
}

# ══════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════

@app.route("/health")
def health():
    """Health check - Railway uses this to verify app is running"""
    return jsonify({
        "status": "ok",
        "framework_loaded": FRAMEWORK_OK,
        "framework_error": framework_error if not FRAMEWORK_OK else None,
    }), 200


@app.route("/")
def index():
    """Serve main UI"""
    return render_template("index.html")


@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify({"status": "ok", "models": MODEL_OPTIONS})


@app.route("/api/hardware", methods=["GET"])
def get_hardware():
    return jsonify({"status": "ok", "hardware": SOC_OPTIONS})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Main analysis endpoint"""

    # Check framework is loaded
    if not FRAMEWORK_OK:
        return jsonify({
            "status": "error",
            "message": f"Framework not loaded: {framework_error}",
        }), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data"}), 400

        # Parse request
        model_preset  = data.get("model", "bert-base")
        soc_preset    = data.get("hardware", "mid-range")
        batch_size    = int(data.get("batch_size", 1))
        seq_len       = int(data.get("seq_len", 128))
        dtype         = data.get("dtype", "fp32")
        num_cores     = int(data.get("num_cores", 1))
        run_quant     = data.get("run_quantization", True)
        run_prune     = data.get("run_pruning", True)
        run_fusion    = data.get("run_fusion", True)
        run_sweep     = data.get("run_sweep", True)
        quant_targets = data.get("quant_targets", ["fp16", "int8"])
        sparsity_vals = data.get("sparsity_values", [0.3, 0.5, 0.7])

        # Session directory for results
        session_id  = str(uuid.uuid4())[:8]
        session_dir = os.path.join(RESULTS_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        print(f"[ANALYZE] model={model_preset} soc={soc_preset} "
              f"batch={batch_size} seq={seq_len} dtype={dtype}")

        # Step 1: Parse model
        parser = TransformerGraphParser(dtype=dtype)
        graph  = parser.parse_from_preset(
            model_preset, batch_size=batch_size, seq_len=seq_len
        )
        graph_summary = graph.summary()

        # Step 2: Build SoC — apply user's num_cores override
	soc = RISCVSoC(preset=soc_preset)
	if num_cores != soc.num_cores:
    		from transformer_perf.hardware.core import RISCVCore
    		freq_ghz = soc.frequency / 1e9
    		core_cfg = {"name": f"rv64gcv-{soc_preset}", "frequency_ghz": freq_ghz}
    		soc.num_cores = num_cores
    		soc.cores = [RISCVCore(config=core_cfg) for _ in range(num_cores)]
	peak = soc.get_peak_performance(dtype)

        # Step 3: Latency
        lat_est    = LatencyEstimator(soc)
        lat_result = lat_est.estimate_model(
            graph, dtype=dtype,
            batch_size=batch_size, seq_len=seq_len,
            num_cores=num_cores,
        )

        # Step 4: Power
        power_est = PowerEstimator(soc, process_node_nm=28)
        energy    = power_est.estimate_energy(lat_result, seq_len=seq_len)

        # Step 5: Roofline
        roofline_model  = RooflineModel(soc)
        roofline_result = roofline_model.analyze_model(graph, dtype=dtype)

        # Step 6: Throughput sweep
        sweep_results = []
        batch_sweep   = []
        if run_sweep:
            tp_est = ThroughputEstimator(soc)
            batch_sweep = tp_est.sweep_batch_sizes(
                parser, model_preset,
                batch_sizes=[1, 2, 4, 8],
                seq_len=seq_len, dtype=dtype,
            )
            for r in batch_sweep:
                sweep_results.append({
                    "batch_size":         r.batch_size,
                    "latency_ms":         round(r.latency_ms, 4),
                    "tokens_per_second":  round(r.tokens_per_second, 1),
                    "samples_per_second": round(r.samples_per_second, 2),
                    "achieved_gflops":    round(r.achieved_gflops, 2),
                    "hw_utilization":     round(r.hw_utilization * 100, 1),
                    "memory_usage_mb":    round(r.memory_usage_mb, 2),
                })

        # Step 7: Quantization
        quant_results = []
        if run_quant:
            q_analyzer = QuantizationAnalyzer(soc)
            for qt in quant_targets:
                qr = q_analyzer.analyze(
                    graph, source_dtype=dtype,
                    target_dtype=qt, latency_estimator=lat_est,
                )
                quant_results.append({
                    "target_dtype":       qt,
                    "compression_ratio":  round(qr.compression_ratio, 1),
                    "memory_savings_pct": round(qr.memory_savings_pct, 1),
                    "speedup":            round(qr.speedup, 2),
                    "accuracy_drop_pct":  round(qr.estimated_accuracy_drop_pct, 2),
                    "original_mb":        round(qr.original_weight_mb, 2),
                    "quantized_mb":       round(qr.quantized_weight_mb, 2),
                })

        # Step 8: Pruning
        prune_results = []
        if run_prune:
            p_analyzer = PruningAnalyzer()
            for sp in sparsity_vals:
                pr = p_analyzer.analyze(
                    graph, sparsity=sp, pruning_type="structured"
                )
                prune_results.append({
                    "sparsity":           round(sp * 100, 0),
                    "speedup":            round(pr.estimated_speedup, 2),
                    "memory_savings_pct": round(pr.memory_savings_pct, 1),
                    "flop_reduction_pct": round(pr.flop_reduction_pct, 1),
                    "accuracy_drop_pct":  round(pr.estimated_accuracy_drop, 2),
                })

        # Step 9: Fusion
        fusion_result = None
        if run_fusion:
            f_analyzer = FusionAnalyzer()
            fr = f_analyzer.analyze(graph)
            fusion_result = {
                "opportunities":        len(fr.opportunities),
                "overall_speedup":      round(fr.overall_speedup, 2),
                "memory_savings_mb":    round(fr.total_memory_savings_mb, 2),
                "fused_layer_count":    fr.fused_layer_count,
                "original_layer_count": fr.original_layer_count,
                "fusions": [
                    {
                        "name":    opp.name,
                        "type":    opp.fusion_type,
                        "speedup": round(opp.speedup, 2),
                    }
                    for opp in fr.opportunities[:10]
                ],
            }

        # Step 10: SoC comparison
        soc_comparison = []
        for preset in ["minimal", "mid-range", "high-perf"]:
            t_soc    = RISCVSoC(preset=preset)
            t_lat    = LatencyEstimator(t_soc)
            t_result = t_lat.estimate_model(graph, dtype=dtype, seq_len=seq_len)
            t_power  = PowerEstimator(t_soc)
            t_energy = t_power.estimate_energy(t_result, seq_len)
            soc_comparison.append({
                "soc_name":    SOC_OPTIONS[preset]["name"],
                "preset":      preset,
                "latency_ms":  round(t_result.total_latency_ms, 4),
                "gflops":      round(t_result.achieved_gflops, 2),
                "power_w":     round(t_energy.average_power_w, 3),
                "efficiency":  round(t_energy.gflops_per_watt, 2),
                "utilization": round(t_result.hardware_utilization * 100, 1),
            })

        # Step 11: Validation
        calibrator      = CalibrationEngine()
        val_results_raw = calibrator.validate_estimate(soc, lat_result)
        validation = [
            {
                "metric":           r.metric,
                "estimated":        round(r.estimated_value, 4),
                "reference":        round(r.reference_value, 4),
                "error_pct":        round(r.error_pct, 1),
                "within_tolerance": r.within_tolerance,
                "confidence":       r.confidence,
            }
            for r in val_results_raw
        ]

        # Step 12: Generate plots
        dashboard = PerformanceDashboard(output_dir=session_dir)
        dashboard.generate_full_dashboard(
            model_latency=lat_result,
            energy_result=energy,
            roofline_result=roofline_result,
            throughput_sweep=batch_sweep if run_sweep else None,
            soc_comparison=soc_comparison,
            prefix="",
        )

        # Read images as base64
        images = {}
        image_files = {
            "latency_breakdown":  "latency_breakdown.png",
            "layer_timeline":     "layer_timeline.png",
            "roofline":           "roofline.png",
            "power_breakdown":    "power_breakdown.png",
            "throughput_scaling": "throughput_scaling.png",
            "soc_comparison":     "soc_comparison.png",
        }
        for key, fname in image_files.items():
            fpath = os.path.join(session_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, "rb") as img_f:
                    images[key] = base64.b64encode(
                        img_f.read()
                    ).decode("utf-8")

        # Read text report
        report_text = ""
        report_path = os.path.join(session_dir, "report.txt")
        if os.path.exists(report_path):
            with open(report_path, "r") as rf:
                report_text = rf.read()

        # Layer table
        sorted_layers = sorted(
            lat_result.layer_breakdown,
            key=lambda x: x.total_cycles, reverse=True
        )
        layer_table = [
            {
                "name":       l.layer_name,
                "type":       l.layer_type,
                "cycles":     l.total_cycles,
                "latency_ms": round(l.latency_ms, 4),
                "flops":      l.flops,
                "bound":      "Compute" if l.is_compute_bound else "Memory",
                "gflops":     round(l.achieved_gflops, 2),
            }
            for l in sorted_layers[:20]
        ]

        # Power breakdown
        pb = energy.power_breakdown
        power_breakdown = {
            "core_dynamic_w": round(pb.core_dynamic_w, 3),
            "core_leakage_w": round(pb.core_leakage_w, 3),
            "vector_unit_w":  round(pb.vector_unit_w, 3),
            "cache_power_w":  round(pb.cache_power_w, 3),
            "dram_power_w":   round(pb.dram_power_w, 3),
            "noc_power_w":    round(pb.noc_power_w, 3),
            "total_power_w":  round(pb.total_power_w, 3),
        }

        print(f"[ANALYZE] Complete! Latency={lat_result.total_latency_ms:.2f}ms")

        return jsonify({
            "status":     "ok",
            "session_id": session_id,
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "model": model_preset, "hardware": soc_preset,
                "batch_size": batch_size, "seq_len": seq_len,
                "dtype": dtype, "num_cores": num_cores,
            },
            "model_info": {
                "name":          MODEL_OPTIONS[model_preset]["name"],
                "num_nodes":     graph_summary["num_nodes"],
                "total_flops":   graph_summary["total_flops"],
                "total_flops_g": round(graph_summary["total_flops"] / 1e9, 2),
                "weight_mb":     round(graph_summary["total_weight_bytes"] / 1e6, 2),
                "memory_mb":     round(graph_summary["total_memory_bytes"] / 1e6, 2),
            },
            "hardware_info": {
                "name":          SOC_OPTIONS[soc_preset]["name"],
                "peak_gflops":   round(peak["system_gflops"], 2),
                "dram_bw_gbps":  round(peak["dram_bandwidth_gbps"], 1),
                "ridge_point":   round(peak["ridge_point_flops_per_byte"], 2),
                "frequency_ghz": round(peak["frequency_ghz"], 1),
                "vlen":          peak["vlen"],
                "num_cores":     peak["num_cores"],
            },
            "latency": {
                "total_ms":           round(lat_result.total_latency_ms, 4),
                "total_cycles":       lat_result.total_cycles,
                "tokens_per_sec":     round(lat_result.throughput_tokens_per_sec, 1),
                "samples_per_sec":    round(lat_result.throughput_samples_per_sec, 2),
                "achieved_gflops":    round(lat_result.achieved_gflops, 2),
                "peak_gflops":        round(lat_result.peak_gflops, 2),
                "hw_utilization_pct": round(lat_result.hardware_utilization * 100, 1),
            },
            "energy": {
                "total_mj":         round(energy.total_energy_mj, 4),
                "per_token_uj":     round(energy.energy_per_token_uj, 2),
                "per_flop_pj":      round(energy.energy_per_flop_pj, 4),
                "average_power_w":  round(energy.average_power_w, 3),
                "gflops_per_watt":  round(energy.gflops_per_watt, 2),
                "tokens_per_joule": round(energy.tokens_per_joule, 0),
                "power_breakdown":  power_breakdown,
            },
            "roofline": {
                "peak_gflops":   round(roofline_result.peak_compute_gflops, 2),
                "peak_bw_gbps":  round(roofline_result.peak_bandwidth_gbps, 1),
                "ridge_point":   round(roofline_result.ridge_point, 2),
                "compute_bound": sum(
                    1 for p in roofline_result.points
                    if p.bound_type == "compute"
                ),
                "memory_bound": sum(
                    1 for p in roofline_result.points
                    if p.bound_type == "memory"
                ),
            },
            "layer_breakdown":  layer_table,
            "throughput_sweep": sweep_results,
            "quantization":     quant_results,
            "pruning":          prune_results,
            "fusion":           fusion_result,
            "soc_comparison":   soc_comparison,
            "validation":       validation,
            "images":           images,
            "report_text":      report_text,
        })

    except Exception as e:
        print(f"[ANALYZE] ERROR: {e}")
        traceback.print_exc()
        return jsonify({
            "status":    "error",
            "message":   str(e),
            "traceback": traceback.format_exc(),
        }), 500


@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(RESULTS_DIR, filename)


# ══════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Railway REQUIRES the app to bind to 0.0.0.0:$PORT
    port = int(os.environ.get("PORT", 8080))

    print(f"[START] PORT from environment: {os.environ.get('PORT', 'NOT SET')}")
    print(f"[START] Using port: {port}")
    print(f"[START] Starting Flask on 0.0.0.0:{port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
    )