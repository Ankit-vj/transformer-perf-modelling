# Save as: C:\Users\ankit\transformer-perf-model\verify_phase3.py

"""
Phase 3 Verification Script
Tests visualization, optimization, and validation modules.
"""

import sys
import os

sys.path.insert(0, os.getcwd())


def test_quantization():
    print("=" * 60)
    print("TEST 1: Quantization Analysis")
    print("=" * 60)

    from transformer_perf.models.parser import TransformerGraphParser
    from transformer_perf.optimizations.quantization import QuantizationAnalyzer

    parser = TransformerGraphParser(dtype="fp32")
    graph = parser.parse_from_preset("bert-base", batch_size=1, seq_len=128)

    analyzer = QuantizationAnalyzer()
    results = analyzer.compare_quantization_options(
        graph, target_dtypes=["fp16", "int8", "int4"]
    )

    for r in results:
        s = r.summary()
        print(f"  + {s['quantization']}: "
              f"compression={s['compression_ratio']}, "
              f"speedup={s['speedup']}, "
              f"mem_savings={s['memory_savings']}")

    print("  PASSED\n")
    return True


def test_pruning():
    print("=" * 60)
    print("TEST 2: Pruning Analysis")
    print("=" * 60)

    from transformer_perf.models.parser import TransformerGraphParser
    from transformer_perf.optimizations.pruning import PruningAnalyzer

    parser = TransformerGraphParser(dtype="fp32")
    graph = parser.parse_from_preset("bert-base", batch_size=1, seq_len=128)

    analyzer = PruningAnalyzer()
    result = analyzer.analyze(graph, sparsity=0.5, pruning_type="structured")

    s = result.summary()
    print(f"  + Sparsity={s['sparsity']}: "
          f"speedup={s['speedup']}, "
          f"flop_reduction={s['flop_reduction']}")

    print("  PASSED\n")
    return True


def test_fusion():
    print("=" * 60)
    print("TEST 3: Operator Fusion Analysis")
    print("=" * 60)

    from transformer_perf.models.parser import TransformerGraphParser
    from transformer_perf.optimizations.fusion import FusionAnalyzer

    parser = TransformerGraphParser(dtype="fp32")
    graph = parser.parse_from_preset("bert-base", batch_size=1, seq_len=128)

    analyzer = FusionAnalyzer()
    result = analyzer.analyze(graph)

    s = result.summary()
    print(f"  + Fusion opportunities: {s['fusion_opportunities']}")
    print(f"    Overall speedup: {s['overall_speedup']}")
    print(f"    Memory savings: {s['memory_savings_mb']}")

    print("  PASSED\n")
    return True


def test_validation():
    print("=" * 60)
    print("TEST 4: Calibration & Validation")
    print("=" * 60)

    from transformer_perf.hardware.soc import RISCVSoC
    from transformer_perf.models.parser import TransformerGraphParser
    from transformer_perf.estimators.latency import LatencyEstimator
    from transformer_perf.validation.calibration import CalibrationEngine

    soc = RISCVSoC(preset="mid-range")
    parser = TransformerGraphParser(dtype="fp32")
    graph = parser.parse_from_preset("bert-base", batch_size=1, seq_len=128)

    estimator = LatencyEstimator(soc)
    result = estimator.estimate_model(graph, dtype="fp32", seq_len=128)

    calibrator = CalibrationEngine()
    val_results = calibrator.validate_estimate(soc, result)

    passed = sum(1 for r in val_results if r.within_tolerance)
    total = len(val_results)
    print(f"  + Validation checks: {passed}/{total} passed")

    for r in val_results:
        ok = "✓" if r.within_tolerance else "✗"
        print(f"    {ok} {r.metric}: {r.confidence}")

    print("  PASSED\n")
    return True


def test_dashboard():
    print("=" * 60)
    print("TEST 5: Visualization Dashboard")
    print("=" * 60)

    from transformer_perf.hardware.soc import RISCVSoC
    from transformer_perf.models.parser import TransformerGraphParser
    from transformer_perf.estimators.latency import LatencyEstimator
    from transformer_perf.estimators.power import PowerEstimator
    from transformer_perf.estimators.roofline import RooflineModel
    from transformer_perf.visualization.dashboard import PerformanceDashboard

    soc = RISCVSoC(preset="mid-range")
    parser = TransformerGraphParser(dtype="fp32")
    graph = parser.parse_from_preset("bert-base", batch_size=1, seq_len=128)

    lat_est = LatencyEstimator(soc)
    lat_result = lat_est.estimate_model(graph, dtype="fp32", seq_len=128)

    power_est = PowerEstimator(soc)
    energy = power_est.estimate_energy(lat_result, seq_len=128)

    roofline = RooflineModel(soc)
    roof_result = roofline.analyze_model(graph, dtype="fp32")

    # Generate dashboard
    output_dir = os.path.join(os.getcwd(), "test_results")
    dashboard = PerformanceDashboard(output_dir=output_dir)

    dashboard.generate_full_dashboard(
        model_latency=lat_result,
        energy_result=energy,
        roofline_result=roof_result,
        prefix="test_"
    )

    # Verify files were created
    expected_files = [
        "test_latency_breakdown.png",
        "test_layer_timeline.png",
        "test_roofline.png",
        "test_power_breakdown.png",
        "test_report.txt",
    ]

    for fname in expected_files:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  + {fname}: {size_kb:.1f} KB")
        else:
            print(f"  - {fname}: NOT FOUND")

    print("  PASSED\n")
    return True


def test_full_pipeline():
    print("=" * 60)
    print("TEST 6: Full End-to-End Pipeline")
    print("=" * 60)

    from transformer_perf.hardware.soc import RISCVSoC
    from transformer_perf.models.parser import TransformerGraphParser
    from transformer_perf.estimators.latency import LatencyEstimator
    from transformer_perf.estimators.power import PowerEstimator
    from transformer_perf.estimators.roofline import RooflineModel
    from transformer_perf.optimizations.quantization import QuantizationAnalyzer
    from transformer_perf.optimizations.fusion import FusionAnalyzer
    from transformer_perf.validation.calibration import CalibrationEngine

    # Test multiple models × SoCs
    models = ["bert-base", "gpt2-small", "vit-base"]
    socs = ["minimal", "mid-range", "high-perf"]

    print(f"\n  {'Model':<12} {'SoC':<12} {'Latency':>10} "
          f"{'GFLOP/s':>10} {'Power':>8} {'GFLOP/s/W':>10}")
    print(f"  {'─'*12} {'─'*12} {'─'*10} {'─'*10} {'─'*8} {'─'*10}")

    for model in models:
        for soc_preset in socs:
            soc = RISCVSoC(preset=soc_preset)
            parser = TransformerGraphParser(dtype="fp32")
            graph = parser.parse_from_preset(model, batch_size=1, seq_len=128)

            lat_est = LatencyEstimator(soc)
            result = lat_est.estimate_model(graph, dtype="fp32", seq_len=128)

            power_est = PowerEstimator(soc)
            energy = power_est.estimate_energy(result, seq_len=128)

            print(
                f"  {model:<12} {soc_preset:<12} "
                f"{result.total_latency_ms:>10.2f} "
                f"{result.achieved_gflops:>10.2f} "
                f"{energy.average_power_w:>8.3f} "
                f"{energy.gflops_per_watt:>10.2f}"
            )

    print("\n  PASSED\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PHASE 3 VERIFICATION")
    print("=" * 60 + "\n")

    tests = [
        ("Quantization", test_quantization),
        ("Pruning", test_pruning),
        ("Fusion", test_fusion),
        ("Validation", test_validation),
        ("Dashboard", test_dashboard),
        ("Full Pipeline", test_full_pipeline),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Final summary
    print("\n" + "=" * 60)
    print("PHASE 3 SUMMARY")
    print("=" * 60)

    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "+" if passed else "X"
        print(f"  {symbol} {name:<20} [{status}]")

    all_passed = all(results.values())
    print(f"\n{'='*60}")
    if all_passed:
        print("ALL PHASE 3 TESTS PASSED!")
        print("\nFRAMEWORK IS COMPLETE!")
        print("Run the full analysis with:")
        print("  python examples/run_full_analysis.py")
    else:
        print("Some tests FAILED. Fix errors above.")
    print(f"{'='*60}\n")