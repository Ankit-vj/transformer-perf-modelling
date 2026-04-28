# Save as: C:\Users\ankit\transformer-perf-model\transformer_perf\estimators\throughput.py

"""
Throughput Estimation Engine.
Estimates inference throughput across different
batch sizes, sequence lengths, and hardware configs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .latency import LatencyEstimator


@dataclass
class ThroughputResult:
    """Throughput estimation result"""
    model_name: str
    batch_size: int
    seq_len: int
    dtype: str
    latency_ms: float
    tokens_per_second: float
    samples_per_second: float
    total_flops: int
    achieved_gflops: float
    peak_gflops: float
    hw_utilization: float
    memory_usage_mb: float


class ThroughputEstimator:
    """
    Estimate throughput for transformer inference
    across different configurations.
    """

    def __init__(self, soc):
        self.soc = soc
        self.latency_estimator = LatencyEstimator(soc)

    def estimate(
        self,
        graph,
        dtype: str = "fp32",
        batch_size: int = 1,
        seq_len: int = 128,
        num_cores: int = 1
    ) -> ThroughputResult:
        """Estimate throughput for a single configuration"""

        result = self.latency_estimator.estimate_model(
            graph, dtype, batch_size, seq_len, num_cores
        )

        memory_usage = graph.get_total_memory() / (1024 * 1024)  # MB

        return ThroughputResult(
            model_name=result.model_name,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=dtype,
            latency_ms=result.total_latency_ms,
            tokens_per_second=result.throughput_tokens_per_sec,
            samples_per_second=result.throughput_samples_per_sec,
            total_flops=result.total_flops,
            achieved_gflops=result.achieved_gflops,
            peak_gflops=result.peak_gflops,
            hw_utilization=result.hardware_utilization,
            memory_usage_mb=memory_usage,
        )

    def sweep_batch_sizes(
        self,
        parser,
        model_preset: str,
        batch_sizes: List[int],
        seq_len: int = 128,
        dtype: str = "fp32",
        num_cores: int = 1
    ) -> List[ThroughputResult]:
        """Sweep across different batch sizes"""

        results = []
        for bs in batch_sizes:
            graph = parser.parse_from_preset(
                model_preset, batch_size=bs, seq_len=seq_len
            )
            result = self.estimate(graph, dtype, bs, seq_len, num_cores)
            results.append(result)

        return results

    def sweep_seq_lengths(
        self,
        parser,
        model_preset: str,
        seq_lengths: List[int],
        batch_size: int = 1,
        dtype: str = "fp32",
        num_cores: int = 1
    ) -> List[ThroughputResult]:
        """Sweep across different sequence lengths"""

        results = []
        for sl in seq_lengths:
            graph = parser.parse_from_preset(
                model_preset, batch_size=batch_size, seq_len=sl
            )
            result = self.estimate(graph, dtype, batch_size, sl, num_cores)
            results.append(result)

        return results

    def compare_dtypes(
        self,
        parser,
        model_preset: str,
        dtypes: List[str],
        batch_size: int = 1,
        seq_len: int = 128,
        num_cores: int = 1
    ) -> List[ThroughputResult]:
        """Compare throughput across data types"""

        results = []
        for dt in dtypes:
            graph = parser.parse_from_preset(
                model_preset, batch_size=batch_size, seq_len=seq_len
            )
            result = self.estimate(graph, dt, batch_size, seq_len, num_cores)
            results.append(result)

        return results

    @staticmethod
    def print_sweep_results(results: List[ThroughputResult], sweep_param: str = "batch_size"):
        """Print sweep results in a table"""

        print(f"\n{'='*90}")
        print(f"Throughput Sweep Results ({sweep_param})")
        print(f"{'='*90}")
        print(
            f"  {'Config':<15} {'Latency(ms)':>12} {'Tok/s':>10} "
            f"{'Samp/s':>10} {'GFLOP/s':>10} {'Util':>8} {'Mem(MB)':>10}"
        )
        print(f"  {'─'*15} {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*10}")

        for r in results:
            if sweep_param == "batch_size":
                config = f"bs={r.batch_size}"
            elif sweep_param == "seq_len":
                config = f"sl={r.seq_len}"
            elif sweep_param == "dtype":
                config = f"{r.dtype}"
            else:
                config = f"{r.batch_size}x{r.seq_len}"

            print(
                f"  {config:<15} "
                f"{r.latency_ms:>12.4f} "
                f"{r.tokens_per_second:>10.0f} "
                f"{r.samples_per_second:>10.1f} "
                f"{r.achieved_gflops:>10.2f} "
                f"{r.hw_utilization:>7.1%} "
                f"{r.memory_usage_mb:>10.1f}"
            )

        print(f"{'='*90}\n")


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from transformer_perf.hardware.soc import RISCVSoC
    from transformer_perf.models.parser import TransformerGraphParser

    soc = RISCVSoC(preset="mid-range")
    parser = TransformerGraphParser(dtype="fp32")

    estimator = ThroughputEstimator(soc)

    # Batch size sweep
    results = estimator.sweep_batch_sizes(
        parser, "bert-base",
        batch_sizes=[1, 2, 4, 8],
        seq_len=128, num_cores=1
    )
    ThroughputEstimator.print_sweep_results(results, "batch_size")