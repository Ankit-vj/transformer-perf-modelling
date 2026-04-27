# Save as: C:\Users\ankit\transformer-perf-model\examples\run_full_analysis.py

"""
============================================================
FULL END-TO-END PERFORMANCE ANALYSIS
============================================================
Runs complete analysis pipeline:
1. Parse transformer model
2. Model RISC-V SoC hardware
3. Estimate latency, throughput, power
4. Analyze optimizations (quantization, pruning, fusion)
5. Validate results
6. Generate dashboard visualizations
============================================================
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from transformer_perf.models.parser import TransformerGraphParser, ModelConfig
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


def main():
    print("\n" + "=" * 70)
    print("  TRANSFORMER INFERENCE PERFORMANCE MODELING FRAMEWORK")
    print("  End-to-End Analysis on RISC-V SoC")
    print("=" * 70)

    # ============================================================
    # STEP 1: Configure Analysis
    # ============================================================
    model_preset = "gpt2-small"
    batch_size = 1
    seq_len = 1
    dtype = "fp32"
    soc_preset = "mid-range"

    print(f"\n[CONFIG]")
    print(f"  Model     : {model_preset}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Seq Length: {seq_len}")
    print(f"  Data Type : {dtype}")
    print(f"  SoC       : {soc_preset}")

    # ============================================================
    # STEP 2: Parse Model
    # ============================================================
    print(f"\n{'─'*70}")
    print("[STEP 1] Parsing Transformer Model...")

    parser = TransformerGraphParser(dtype=dtype)
    graph = parser.parse_from_preset(
        model_preset, batch_size=batch_size, seq_len=seq_len
    )
    graph.print_summary()

    # ============================================================
    # STEP 3: Model Hardware
    # ============================================================
    print(f"\n{'─'*70}")
    print("[STEP 2] Modeling RISC-V SoC Hardware...")

    soc = RISCVSoC(preset=soc_preset)
    soc.print_summary()

    # ============================================================
    # STEP 4: Estimate Latency
    # ============================================================
    print(f"\n{'─'*70}")
    print("[STEP 3] Estimating Inference Latency...")

    lat_estimator = LatencyEstimator(soc)
    latency_result = lat_estimator.estimate_model(
        graph, dtype=dtype, batch_size=batch_size,
        seq_len=seq_len, num_cores=1
    )
    latency_result.print_report()

    # ============================================================
    # STEP 5: Throughput Sweep
    # ============================================================
    print(f"\n{'─'*70}")
    print("[STEP 4] Throughput Scaling Analysis...")

    tp_estimator = ThroughputEstimator(soc)

    # Batch size sweep
    batch_results = tp_estimator.sweep_batch_sizes(
        parser, model_preset,
        batch_sizes=[1, 2, 4, 8, 16],
        seq_len=seq_len, dtype=dtype, num_cores=4

    )
    ThroughputEstimator.print_sweep_results(batch_results, "batch_size")

    # Sequence length sweep
    seq_results = tp_estimator.sweep_seq_lengths(
        parser, model_preset,
        seq_lengths=[32, 64, 128, 256, 512],
        batch_size=batch_size, dtype=dtype
    )
    ThroughputEstimator.print_sweep_results(seq_results, "seq_len")

    # ============================================================
    # STEP 6: Power & Energy Estimation
    # ============================================================
    print(f"\n{'─'*70}")
    print("[STEP 5] Power & Energy Estimation...")

    power_estimator = PowerEstimator(soc, process_node_nm=28)
    energy_result = power_estimator.estimate_energy(
        latency_result, seq_len=seq_len
    )
    energy_result.print_report()

    # ============================================================
    # STEP 7: Roofline Analysis
    # ============================================================
    print(f"\n{'─'*70}")
    print("[STEP 6] Roofline Analysis...")

    roofline = RooflineModel(soc)
    roofline_result = roofline.analyze_model(graph, dtype=dtype)
    roofline_result.print_report()

    # ============================================================
    # STEP 8: Optimization Analysis
    # ============================================================
    print(f"\n{'─'*70}")
    print("[STEP 7] Optimization Analysis...")

    # Quantization
    print("\n  --- Quantization ---")
    quant_analyzer = QuantizationAnalyzer(soc)
    quant_results = quant_analyzer.compare_quantization_options(
        graph, target_dtypes=["fp16", "int8", "int4"],
        latency_estimator=lat_estimator
    )
    for qr in quant_results:
        qr.print_report()

    # Pruning
    print("  --- Pruning ---")
    prune_analyzer = PruningAnalyzer()
    for sparsity in [0.3, 0.5, 0.7, 0.9]:
        prune_result = prune_analyzer.analyze(
            graph, sparsity=sparsity, pruning_type="structured"
        )
        s = prune_result.summary()
        print(f"  Sparsity {sparsity:.0%}: "
              f"speedup={s['speedup']}, "
              f"mem_save={s['memory_savings']}, "
              f"acc_drop={s['accuracy_drop']}")

    # Fusion
    print("\n  --- Operator Fusion ---")
    fusion_analyzer = FusionAnalyzer()
    fusion_result = fusion_analyzer.analyze(graph)
    fusion_result.print_report()

    # ============================================================
    # STEP 9: SoC Comparison
    # ============================================================
    print(f"\n{'─'*70}")
    print("[STEP 8] SoC Configuration Comparison...")

    soc_comparison = []
    for preset in ["minimal", "mid-range", "high-perf"]:
        test_soc = RISCVSoC(preset=preset)
        test_lat = LatencyEstimator(test_soc)
        test_result = test_lat.estimate_model(
            graph, dtype=dtype, seq_len=seq_len
        )
        test_power = PowerEstimator(test_soc)
        test_energy = test_power.estimate_energy(test_result, seq_len)

        soc_comparison.append({
            'soc_name': preset,
            'latency_ms': test_result.total_latency_ms,
            'gflops': test_result.achieved_gflops,
            'power_w': test_energy.average_power_w,
            'efficiency': test_energy.gflops_per_watt,
        })

    print(f"\n  {'SoC':<15} {'Latency(ms)':>12} {'GFLOP/s':>10} "
          f"{'Power(W)':>10} {'GFLOP/s/W':>10}")
    print(f"  {'─'*15} {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
    for d in soc_comparison:
        print(f"  {d['soc_name']:<15} {d['latency_ms']:>12.4f} "
              f"{d['gflops']:>10.2f} {d['power_w']:>10.3f} "
              f"{d['efficiency']:>10.2f}")

    # ============================================================
    # STEP 10: Validation
    # ============================================================
    print(f"\n{'─'*70}")
    print("[STEP 9] Validation...")

    calibrator = CalibrationEngine()
    validation_results = calibrator.validate_estimate(
        soc, latency_result
    )
    calibrator.print_validation_report(validation_results)

    # ============================================================
    # STEP 11: Generate Dashboard
    # ============================================================
    print(f"\n{'─'*70}")
    print("[STEP 10] Generating Visualization Dashboard...")

    output_dir = os.path.join(project_root, "results")
    dashboard = PerformanceDashboard(output_dir=output_dir)

    dashboard.generate_full_dashboard(
        model_latency=latency_result,
        energy_result=energy_result,
        roofline_result=roofline_result,
        throughput_sweep=batch_results,
        soc_comparison=soc_comparison,
        prefix=f"{model_preset}_"
    )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("  ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\n  Key Results for {model_preset} on {soc_preset} SoC:")
    print(f"  {'─'*50}")
    print(f"  Latency        : {latency_result.total_latency_ms:.4f} ms")
    print(f"  Throughput     : {latency_result.throughput_tokens_per_sec:.0f} tokens/s")
    print(f"  Performance    : {latency_result.achieved_gflops:.2f} GFLOP/s")
    print(f"  HW Utilization : {latency_result.hardware_utilization:.1%}")
    print(f"  Power          : {energy_result.average_power_w:.3f} W")
    print(f"  Energy/Token   : {energy_result.energy_per_token_uj:.2f} µJ")
    print(f"  Efficiency     : {energy_result.gflops_per_watt:.2f} GFLOP/s/W")
    print(f"\n  Visualizations saved in: {output_dir}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()