# Save as: C:\Users\ankit\transformer-perf-model\verify_phase2.py

"""
Phase 2 Verification Script
Tests all mapping and estimator modules.
"""

import sys
import os

sys.path.insert(0, os.getcwd())


def test_tiling():
    """Test tiling strategy"""
    print("=" * 60)
    print("TEST 1: Tiling Strategy")
    print("=" * 60)

    from transformer_perf.mapping.tiling import TilingStrategy

    tiler = TilingStrategy()

    # Test GEMM tiling
    result = tiler.compute_gemm_tiling(
        M=128, N=3072, K=768,
        bytes_per_element=4,
        target_cache="L1D"
    )
    print(f"  + GEMM Tiling: {result.tile_config.summary()}")
    print(f"    Total tiles: {result.total_tiles}")
    print(f"    Cache util: {result.cache_utilization:.1%}")

    # Test attention tiling
    attn = tiler.compute_attention_tiling(
        batch_size=1, num_heads=12,
        seq_len=128, head_dim=64
    )
    print(f"  + Attention tiling: {len(attn)} sub-operations")

    print("  PASSED\n")
    return True


def test_dataflow():
    """Test dataflow analyzer"""
    print("=" * 60)
    print("TEST 2: Dataflow Analyzer")
    print("=" * 60)

    from transformer_perf.mapping.dataflow import DataflowAnalyzer

    analyzer = DataflowAnalyzer()

    result = analyzer.analyze_linear_layer(
        name="test_linear",
        batch_size=1, seq_len=128,
        in_features=768, out_features=3072
    )

    print(f"  + Linear layer dataflow:")
    print(f"    Total movement: {result.total_data_movement_bytes/1024:.1f} KB")
    print(f"    Arithmetic intensity: {result.arithmetic_intensity:.2f} FLOP/byte")
    print(f"    Memory bound: {result.memory_bound}")

    print("  PASSED\n")
    return True


def test_scheduler():
    """Test instruction scheduler"""
    print("=" * 60)
    print("TEST 3: Instruction Scheduler")
    print("=" * 60)

    from transformer_perf.mapping.scheduler import InstructionScheduler

    scheduler = InstructionScheduler()

    result = scheduler.schedule_matmul(
        "test_matmul", M=128, N=3072, K=768
    )

    print(f"  + MatMul schedule:")
    print(f"    Total cycles: {result.total_cycles:,}")
    print(f"    Compute cycles: {result.compute_cycles:,}")
    print(f"    Memory cycles: {result.memory_cycles:,}")
    print(f"    Pipeline util: {result.pipeline_utilization:.1%}")

    print("  PASSED\n")
    return True


def test_latency():
    """Test latency estimator"""
    print("=" * 60)
    print("TEST 4: Latency Estimator")
    print("=" * 60)

    from transformer_perf.hardware.soc import RISCVSoC
    from transformer_perf.models.parser import TransformerGraphParser
    from transformer_perf.estimators.latency import LatencyEstimator

    soc = RISCVSoC(preset="mid-range")
    parser = TransformerGraphParser(dtype="fp32")
    graph = parser.parse_from_preset("bert-base", batch_size=1, seq_len=128)

    estimator = LatencyEstimator(soc)
    result = estimator.estimate_model(graph, dtype="fp32", seq_len=128)

    print(f"  + BERT-base latency on mid-range SoC:")
    print(f"    Total latency: {result.total_latency_ms:.4f} ms")
    print(f"    Throughput: {result.throughput_tokens_per_sec:.1f} tokens/s")
    print(f"    Achieved: {result.achieved_gflops:.2f} GFLOP/s")
    print(f"    HW Utilization: {result.hardware_utilization:.1%}")

    print("  PASSED\n")
    return True


def test_throughput():
    """Test throughput estimator"""
    print("=" * 60)
    print("TEST 5: Throughput Estimator")
    print("=" * 60)

    from transformer_perf.hardware.soc import RISCVSoC
    from transformer_perf.models.parser import TransformerGraphParser
    from transformer_perf.estimators.throughput import ThroughputEstimator

    soc = RISCVSoC(preset="mid-range")
    parser = TransformerGraphParser(dtype="fp32")

    estimator = ThroughputEstimator(soc)
    results = estimator.sweep_batch_sizes(
        parser, "bert-base",
        batch_sizes=[1, 4],
        seq_len=128
    )

    print(f"  + Batch size sweep results:")
    for r in results:
        print(f"    BS={r.batch_size}: {r.latency_ms:.4f} ms, "
              f"{r.tokens_per_second:.0f} tok/s")

    print("  PASSED\n")
    return True


def test_power():
    """Test power estimator"""
    print("=" * 60)
    print("TEST 6: Power Estimator")
    print("=" * 60)

    from transformer_perf.hardware.soc import RISCVSoC
    from transformer_perf.models.parser import TransformerGraphParser
    from transformer_perf.estimators.latency import LatencyEstimator
    from transformer_perf.estimators.power import PowerEstimator

    soc = RISCVSoC(preset="mid-range")
    parser = TransformerGraphParser(dtype="fp32")
    graph = parser.parse_from_preset("bert-base", batch_size=1, seq_len=128)

    lat_est = LatencyEstimator(soc)
    lat_result = lat_est.estimate_model(graph, dtype="fp32", seq_len=128)

    power_est = PowerEstimator(soc, process_node_nm=28)
    energy = power_est.estimate_energy(lat_result, seq_len=128)

    print(f"  + BERT-base energy estimation:")
    print(f"    Total energy: {energy.total_energy_mj:.4f} mJ")
    print(f"    Average power: {energy.average_power_w:.3f} W")
    print(f"    Efficiency: {energy.gflops_per_watt:.2f} GFLOPS/W")

    print("  PASSED\n")
    return True


def test_roofline():
    """Test roofline model"""
    print("=" * 60)
    print("TEST 7: Roofline Model")
    print("=" * 60)

    from transformer_perf.hardware.soc import RISCVSoC
    from transformer_perf.models.parser import TransformerGraphParser
    from transformer_perf.estimators.roofline import RooflineModel

    soc = RISCVSoC(preset="mid-range")
    parser = TransformerGraphParser(dtype="fp32")
    graph = parser.parse_from_preset("bert-base", batch_size=1, seq_len=128)

    roofline = RooflineModel(soc)
    result = roofline.analyze_model(graph, dtype="fp32")

    summary = result.summary()
    print(f"  + Roofline analysis:")
    print(f"    Peak: {summary['peak_gflops']} GFLOP/s")
    print(f"    Ridge point: {summary['ridge_point']}")
    print(f"    Compute-bound ops: {summary['compute_bound_ops']}")
    print(f"    Memory-bound ops: {summary['memory_bound_ops']}")

    print("  PASSED\n")
    return True


def test_end_to_end():
    """Full end-to-end test"""
    print("=" * 60)
    print("TEST 8: End-to-End Pipeline")
    print("=" * 60)

    from transformer_perf.hardware.soc import RISCVSoC
    from transformer_perf.models.parser import TransformerGraphParser
    from transformer_perf.estimators.latency import LatencyEstimator
    from transformer_perf.estimators.power import PowerEstimator
    from transformer_perf.estimators.roofline import RooflineModel

    # Test multiple models on multiple SoCs
    models = ["bert-base", "gpt2-small"]
    soc_presets = ["minimal", "mid-range", "high-perf"]

    print(f"\n  {'Model':<15} {'SoC':<20} {'Latency(ms)':>12} "
          f"{'GFLOP/s':>10} {'Power(W)':>10}")
    print(f"  {'─'*15} {'─'*20} {'─'*12} {'─'*10} {'─'*10}")

    for model_name in models:
        for soc_preset in soc_presets:
            soc = RISCVSoC(preset=soc_preset)
            parser = TransformerGraphParser(dtype="fp32")
            graph = parser.parse_from_preset(
                model_name, batch_size=1, seq_len=128
            )

            lat_est = LatencyEstimator(soc)
            lat_result = lat_est.estimate_model(
                graph, dtype="fp32", seq_len=128
            )

            power_est = PowerEstimator(soc)
            energy = power_est.estimate_energy(lat_result, seq_len=128)

            print(
                f"  {model_name:<15} {soc_preset:<20} "
                f"{lat_result.total_latency_ms:>12.4f} "
                f"{lat_result.achieved_gflops:>10.2f} "
                f"{energy.average_power_w:>10.3f}"
            )

    print("\n  PASSED\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PHASE 2 VERIFICATION")
    print("=" * 60 + "\n")

    tests = [
        ("Tiling", test_tiling),
        ("Dataflow", test_dataflow),
        ("Scheduler", test_scheduler),
        ("Latency", test_latency),
        ("Throughput", test_throughput),
        ("Power", test_power),
        ("Roofline", test_roofline),
        ("End-to-End", test_end_to_end),
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

    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "+" if passed else "X"
        print(f"  {symbol} {name:<20} [{status}]")

    all_passed = all(results.values())
    print(f"\n{'='*60}")
    if all_passed:
        print("ALL PHASE 2 TESTS PASSED!")
        print("Ready to proceed to Phase 3 (Visualization & Dashboard)")
    else:
        print("Some tests FAILED. Please fix errors above.")
    print(f"{'='*60}\n")