# Save as: C:\Users\ankit\transformer-perf-model\verify_structure.py

"""
Verify that the project structure is set up correctly
and all modules can be imported.
"""

import os
import sys

def check_directory_structure():
    """Check all required directories exist"""
    print("=" * 60)
    print("Checking Directory Structure")
    print("=" * 60)

    required_dirs = [
        "transformer_perf/models",
        "transformer_perf/hardware",
        "transformer_perf/mapping",
        "transformer_perf/estimators",
        "transformer_perf/optimizations",
        "transformer_perf/validation",
        "transformer_perf/visualization",
        "configs",
        "configs/models",
        "configs/hardware",
        "tests",
        "examples",
        "docs",
    ]

    all_ok = True
    for directory in required_dirs:
        path = os.path.join(os.getcwd(), directory)
        exists = os.path.isdir(path)
        status = "OK" if exists else "MISSING"
        symbol = "+" if exists else "X"
        print(f"  {symbol} {directory:<30} [{status}]")
        if not exists:
            all_ok = False

    return all_ok


def check_files():
    """Check all required files exist"""
    print(f"\n{'='*60}")
    print("Checking Required Files")
    print(f"{'='*60}")

    required_files = [
        "transformer_perf/models/__init__.py",
        "transformer_perf/models/parser.py",
        "transformer_perf/models/graph.py",
        "transformer_perf/models/layers.py",
        "transformer_perf/hardware/__init__.py",
        "transformer_perf/hardware/core.py",
        "transformer_perf/hardware/memory.py",
        "transformer_perf/hardware/vector.py",
        "transformer_perf/hardware/soc.py",
        "transformer_perf/mapping/__init__.py",
        "transformer_perf/mapping/scheduler.py",
        "transformer_perf/mapping/tiling.py",
        "transformer_perf/mapping/dataflow.py",
        "transformer_perf/estimators/__init__.py",
        "transformer_perf/estimators/latency.py",
        "transformer_perf/estimators/throughput.py",
        "transformer_perf/estimators/power.py",
        "transformer_perf/estimators/roofline.py",
        "transformer_perf/optimizations/__init__.py",
        "transformer_perf/validation/__init__.py",
        "transformer_perf/visualization/__init__.py",
        "configs/hardware/mid_range_soc.yaml",
        "configs/models/bert_base.yaml",
    ]

    all_ok = True
    for filepath in required_files:
        path = os.path.join(os.getcwd(), filepath)
        exists = os.path.isfile(path)
        status = "OK" if exists else "MISSING"
        symbol = "+" if exists else "X"
        print(f"  {symbol} {filepath:<45} [{status}]")
        if not exists:
            all_ok = False

    return all_ok


def check_imports():
    """Check that modules can be imported"""
    print(f"\n{'='*60}")
    print("Checking Module Imports")
    print(f"{'='*60}")

    # Add project root to path
    sys.path.insert(0, os.getcwd())

    modules = [
        ("transformer_perf.models.layers", "LayerProfiler, LayerType"),
        ("transformer_perf.models.graph", "ComputationGraph"),
        ("transformer_perf.models.parser", "TransformerGraphParser, ModelConfig"),
        ("transformer_perf.hardware.core", "RISCVCore"),
        ("transformer_perf.hardware.memory", "MemoryHierarchy"),
        ("transformer_perf.hardware.vector", "RVVExtension"),
        ("transformer_perf.hardware.soc", "RISCVSoC"),
    ]

    all_ok = True
    for module_name, classes in modules:
        try:
            module = __import__(module_name, fromlist=[classes.split(",")[0].strip()])
            print(f"  + {module_name:<30} [{classes}]  OK")
        except Exception as e:
            print(f"  X {module_name:<30} FAILED: {e}")
            all_ok = False

    return all_ok


def run_quick_test():
    """Run a quick functionality test"""
    print(f"\n{'='*60}")
    print("Running Quick Functionality Test")
    print(f"{'='*60}")

    sys.path.insert(0, os.getcwd())

    try:
        # Test 1: Layer Profiling
        from transformer_perf.models.layers import LayerProfiler
        profiler = LayerProfiler(dtype="fp32")
        linear = profiler.profile_linear(
            "test_linear", batch_size=1, seq_len=128,
            in_features=768, out_features=3072
        )
        print(f"  + Layer Profiler  : Linear FLOPs = {linear.flops:,}")

        # Test 2: Graph Building
        from transformer_perf.models.graph import ComputationGraph
        graph = ComputationGraph(name="test")
        node_id = graph.add_node(linear)
        print(f"  + Graph Builder   : Added node {node_id}")

        # Test 3: Model Parsing
        from transformer_perf.models.parser import TransformerGraphParser
        parser = TransformerGraphParser(dtype="fp32")
        model_graph = parser.parse_from_preset(
            "bert-base", batch_size=1, seq_len=128
        )
        print(f"  + Model Parser    : BERT-base nodes = {len(model_graph.nodes)}")
        print(f"                      Total FLOPs = {model_graph.get_total_flops():,}")

        # Test 4: Hardware Modeling
        from transformer_perf.hardware.soc import RISCVSoC
        soc = RISCVSoC(preset="mid-range")
        peak = soc.get_peak_performance("fp32")
        print(f"  + SoC Model       : {soc.name}")
        print(f"                      Peak = {peak['system_gflops']:.2f} GFLOP/s")

        # Test 5: Vector Extension
        from transformer_perf.hardware.vector import RVVExtension
        rvv = RVVExtension(vlen=256)
        vflops = rvv.vector_peak_flops(2e9, "fp32")
        print(f"  + Vector Extension: {vflops/1e9:.2f} GFLOP/s (VLEN=256)")

        print(f"\n{'='*60}")
        print("ALL TESTS PASSED!")
        print(f"{'='*60}")
        return True

    except Exception as e:
        print(f"\n  X Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nTransformer Performance Model - Structure Verification")
    print(f"Working Directory: {os.getcwd()}\n")

    dir_ok = check_directory_structure()
    file_ok = check_files()
    import_ok = check_imports()
    test_ok = run_quick_test()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Directories : {'PASS' if dir_ok else 'FAIL'}")
    print(f"  Files       : {'PASS' if file_ok else 'FAIL'}")
    print(f"  Imports     : {'PASS' if import_ok else 'FAIL'}")
    print(f"  Tests       : {'PASS' if test_ok else 'FAIL'}")
    print(f"{'='*60}")

    if all([dir_ok, file_ok, import_ok, test_ok]):
        print("\nProject structure is READY! Proceed to Phase 2.")
    else:
        print("\nSome checks failed. Please fix the issues above.")