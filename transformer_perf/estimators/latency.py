# Save as: C:\Users\ankit\transformer-perf-model\transformer_perf\estimators\latency.py

"""
Latency Estimation Engine.
Estimates end-to-end inference latency by combining
compute modeling, memory modeling, and scheduling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


@dataclass
class LayerLatency:
    """Latency breakdown for a single layer"""
    layer_name: str
    layer_type: str
    compute_cycles: int = 0
    memory_cycles: int = 0
    total_cycles: int = 0
    latency_us: float = 0.0       # microseconds
    latency_ms: float = 0.0       # milliseconds
    is_compute_bound: bool = True
    flops: int = 0
    achieved_gflops: float = 0.0

    def summary(self) -> Dict:
        return {
            "layer_name": self.layer_name,
            "layer_type": self.layer_type,
            "compute_cycles": self.compute_cycles,
            "memory_cycles": self.memory_cycles,
            "total_cycles": self.total_cycles,
            "latency_ms": f"{self.latency_ms:.4f}",
            "bound": "compute" if self.is_compute_bound else "memory",
            "achieved_gflops": f"{self.achieved_gflops:.2f}",
        }


@dataclass
class ModelLatency:
    """Complete model latency estimation"""
    model_name: str
    total_cycles: int = 0
    total_latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    throughput_samples_per_sec: float = 0.0
    total_flops: int = 0
    achieved_gflops: float = 0.0
    peak_gflops: float = 0.0
    hardware_utilization: float = 0.0
    layer_breakdown: List[LayerLatency] = field(default_factory=list)
    dtype: str = "fp32"
    batch_size: int = 1
    seq_len: int = 128

    def summary(self) -> Dict:
        return {
            "model_name": self.model_name,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "total_latency_ms": f"{self.total_latency_ms:.4f}",
            "throughput_tokens/s": f"{self.throughput_tokens_per_sec:.1f}",
            "throughput_samples/s": f"{self.throughput_samples_per_sec:.1f}",
            "total_gflops": f"{self.total_flops / 1e9:.2f}",
            "achieved_gflops": f"{self.achieved_gflops:.2f}",
            "peak_gflops": f"{self.peak_gflops:.2f}",
            "hw_utilization": f"{self.hardware_utilization:.1%}",
            "num_layers": len(self.layer_breakdown),
        }

    def print_report(self):
        """Print detailed latency report"""
        print(f"\n{'='*80}")
        print(f"LATENCY REPORT: {self.model_name}")
        print(f"{'='*80}")
        print(f"  Configuration:")
        print(f"    Batch Size    : {self.batch_size}")
        print(f"    Sequence Len  : {self.seq_len}")
        print(f"    Data Type     : {self.dtype}")
        print(f"{'─'*80}")
        print(f"  Results:")
        print(f"    Total Latency : {self.total_latency_ms:.4f} ms")
        print(f"    Throughput    : {self.throughput_tokens_per_sec:.1f} tokens/s")
        print(f"    Throughput    : {self.throughput_samples_per_sec:.1f} samples/s")
        print(f"    Total FLOPs   : {self.total_flops/1e9:.2f} GFLOPs")
        print(f"    Achieved      : {self.achieved_gflops:.2f} GFLOP/s")
        print(f"    Peak          : {self.peak_gflops:.2f} GFLOP/s")
        print(f"    HW Utilization: {self.hardware_utilization:.1%}")
        print(f"{'─'*80}")
        print(f"  Layer Breakdown (top 10 by latency):")
        print(f"  {'Layer':<35} {'Type':<15} {'Cycles':>12} {'ms':>10} {'Bound':<8}")
        print(f"  {'─'*35} {'─'*15} {'─'*12} {'─'*10} {'─'*8}")

        sorted_layers = sorted(
            self.layer_breakdown,
            key=lambda x: x.total_cycles,
            reverse=True
        )

        for layer in sorted_layers[:10]:
            bound = "COMP" if layer.is_compute_bound else "MEM"
            print(
                f"  {layer.layer_name:<35} "
                f"{layer.layer_type:<15} "
                f"{layer.total_cycles:>12,} "
                f"{layer.latency_ms:>10.4f} "
                f"{bound:<8}"
            )

        # Layer type summary
        print(f"\n  Summary by Layer Type:")
        print(f"  {'Type':<20} {'Count':>6} {'Total ms':>12} {'% of Total':>12}")
        print(f"  {'─'*20} {'─'*6} {'─'*12} {'─'*12}")

        type_stats = {}
        for layer in self.layer_breakdown:
            lt = layer.layer_type
            if lt not in type_stats:
                type_stats[lt] = {"count": 0, "ms": 0.0}
            type_stats[lt]["count"] += 1
            type_stats[lt]["ms"] += layer.latency_ms

        for lt, stats in sorted(type_stats.items(), key=lambda x: x[1]["ms"], reverse=True):
            pct = stats["ms"] / self.total_latency_ms * 100 if self.total_latency_ms > 0 else 0
            print(
                f"  {lt:<20} {stats['count']:>6} "
                f"{stats['ms']:>12.4f} {pct:>11.1f}%"
            )

        print(f"{'='*80}\n")


class LatencyEstimator:
    """
    Estimates end-to-end inference latency for transformer
    models on RISC-V SoC hardware.

    Combines:
    - Compute cycle estimation (using vector throughput)
    - Memory access cycle estimation (using cache model)
    - Takes max(compute, memory) as the bound
    """

    def __init__(self, soc):
        """
        Args:
            soc: RISCVSoC object with hardware configuration
        """
        self.soc = soc
        self.core = soc.cores[0]
        self.vector = soc.vector
        self.memory = soc.memory
        self.frequency = soc.frequency
        self.num_cores = soc.num_cores

    def estimate_layer(
        self,
        layer,
        dtype: str = "fp32",
        num_cores: int = 1
    ) -> LayerLatency:
        """
        Estimate latency for a single layer.

        Args:
            layer: LayerDefinition object
            dtype: Data type for computation
            num_cores: Number of cores to use

        Returns:
            LayerLatency with detailed breakdown
        """
        layer_type = layer.layer_type.name

        if layer_type in ["LINEAR", "ATTENTION", "MLP"]:
            return self._estimate_compute_heavy(layer, dtype, num_cores)
        elif layer_type in ["LAYERNORM", "RMSNORM"]:
            return self._estimate_normalization(layer, dtype, num_cores)
        elif layer_type in ["SOFTMAX"]:
            return self._estimate_softmax(layer, dtype, num_cores)
        elif layer_type in ["EMBEDDING"]:
            return self._estimate_memory_only(layer, dtype)
        elif layer_type in ["RESIDUAL_ADD", "RESHAPE", "TRANSPOSE"]:
            return self._estimate_lightweight(layer, dtype, num_cores)
        else:
            return self._estimate_generic(layer, dtype, num_cores)

    def _estimate_compute_heavy(
        self, layer, dtype: str, num_cores: int
    ) -> LayerLatency:
        """Estimate latency for compute-heavy layers (Linear, Attention, MLP)"""

        sew_bits = {"fp32": 32, "fp16": 16, "bf16": 16, "int8": 8}.get(dtype, 32)
        bpe = sew_bits // 8

        # Compute cycles
        # Vector throughput for FMA: elements_per_vreg * FMA_throughput
        elements_per_vreg = self.vector.vlen // sew_bits
        fma_unit = self.vector.vector_units.get("VFMA")
        fma_throughput = fma_unit.throughput if fma_unit else 1.0
        fma_latency = fma_unit.latency if fma_unit else 3

        # FLOPs per cycle (FMA = 2 FLOPs, vector processes elements_per_vreg)
        flops_per_cycle = elements_per_vreg * fma_throughput * 2

        # Distribute across cores
        effective_flops_per_cycle = flops_per_cycle * min(num_cores, self.num_cores)

        # Compute cycles
        compute_cycles = math.ceil(layer.flops / effective_flops_per_cycle)
        compute_cycles += fma_latency  # Pipeline startup

        # Memory cycles
        total_data = layer.total_memory_bytes
        weight_data = layer.weight_bytes
        activation_data = layer.activation_bytes + layer.output_bytes

        # Determine which level weights are served from
        mem_result = self.memory.estimate_data_access_cycles(
            data_size_bytes=total_data,
            core_frequency_hz=self.frequency,
        )

        # Weight loading cycles
        if weight_data <= self.memory.l2.size_bytes:
            weight_bw = self.memory.l2.bandwidth_bytes_per_cycle
            weight_latency = self.memory.l2.latency_cycles
        elif weight_data <= self.memory.l3.size_bytes:
            weight_bw = self.memory.l3.bandwidth_bytes_per_cycle
            weight_latency = self.memory.l3.latency_cycles
        else:
            weight_bw = self.memory.dram.peak_bandwidth_gbps * 1e9 / self.frequency
            weight_latency = self.memory.dram.latency_cycles(self.frequency)

        weight_cycles = weight_latency + math.ceil(weight_data / weight_bw)

        # Activation loading cycles (typically from L1/L2)
        if activation_data <= self.memory.l1d.size_bytes:
            act_bw = self.memory.l1d.bandwidth_bytes_per_cycle
        else:
            act_bw = self.memory.l2.bandwidth_bytes_per_cycle

        activation_cycles = math.ceil(activation_data / act_bw)

        memory_cycles = weight_cycles + activation_cycles

        # Total = max(compute, memory) for overlapped execution
        total_cycles = max(compute_cycles, memory_cycles)
        is_compute_bound = compute_cycles >= memory_cycles

        # Convert to time
        latency_sec = total_cycles / self.frequency
        latency_ms = latency_sec * 1000
        latency_us = latency_sec * 1e6

        # Achieved performance
        achieved_gflops = (
            layer.flops / latency_sec / 1e9 if latency_sec > 0 else 0
        )

        return LayerLatency(
            layer_name=layer.name,
            layer_type=layer.layer_type.name,
            compute_cycles=compute_cycles,
            memory_cycles=memory_cycles,
            total_cycles=total_cycles,
            latency_us=latency_us,
            latency_ms=latency_ms,
            is_compute_bound=is_compute_bound,
            flops=layer.flops,
            achieved_gflops=achieved_gflops,
        )

    def _estimate_normalization(
        self, layer, dtype: str, num_cores: int
    ) -> LayerLatency:
        """Estimate latency for normalization layers"""

        sew_bits = {"fp32": 32, "fp16": 16, "int8": 8}.get(dtype, 32)
        elements_per_vreg = self.vector.vlen // sew_bits

        # LayerNorm involves reductions → lower vector efficiency
        # Effective throughput is ~50% of peak due to reductions
        effective_throughput = elements_per_vreg * 0.5

        compute_cycles = math.ceil(layer.flops / effective_throughput)

        # Memory: read input + write output + load gamma/beta
        total_data = layer.total_memory_bytes
        memory_cycles = math.ceil(
            total_data / self.memory.l1d.bandwidth_bytes_per_cycle
        )

        total_cycles = max(compute_cycles, memory_cycles)
        latency_sec = total_cycles / self.frequency

        return LayerLatency(
            layer_name=layer.name,
            layer_type=layer.layer_type.name,
            compute_cycles=compute_cycles,
            memory_cycles=memory_cycles,
            total_cycles=total_cycles,
            latency_us=latency_sec * 1e6,
            latency_ms=latency_sec * 1000,
            is_compute_bound=compute_cycles >= memory_cycles,
            flops=layer.flops,
            achieved_gflops=layer.flops / latency_sec / 1e9 if latency_sec > 0 else 0,
        )

    def _estimate_softmax(
        self, layer, dtype: str, num_cores: int
    ) -> LayerLatency:
        """Estimate latency for softmax"""

        sew_bits = {"fp32": 32, "fp16": 16, "int8": 8}.get(dtype, 32)
        elements_per_vreg = self.vector.vlen // sew_bits

        # Softmax: max_reduce + exp + sum_reduce + div
        # Lower efficiency due to transcendental ops and reductions
        effective_throughput = elements_per_vreg * 0.3

        compute_cycles = math.ceil(layer.flops / effective_throughput)

        total_data = layer.total_memory_bytes
        memory_cycles = math.ceil(
            total_data / self.memory.l1d.bandwidth_bytes_per_cycle
        )

        total_cycles = max(compute_cycles, memory_cycles)
        latency_sec = total_cycles / self.frequency

        return LayerLatency(
            layer_name=layer.name,
            layer_type=layer.layer_type.name,
            compute_cycles=compute_cycles,
            memory_cycles=memory_cycles,
            total_cycles=total_cycles,
            latency_us=latency_sec * 1e6,
            latency_ms=latency_sec * 1000,
            is_compute_bound=compute_cycles >= memory_cycles,
            flops=layer.flops,
            achieved_gflops=layer.flops / latency_sec / 1e9 if latency_sec > 0 else 0,
        )

    def _estimate_memory_only(
        self, layer, dtype: str
    ) -> LayerLatency:
        """Estimate latency for memory-only operations (embedding)"""

        total_data = layer.weight_bytes + layer.output_bytes

        # Embedding is purely memory-bound (lookup)
        mem_result = self.memory.estimate_data_access_cycles(
            data_size_bytes=total_data,
            core_frequency_hz=self.frequency,
        )

        total_cycles = int(mem_result["total_cycles"])
        latency_sec = total_cycles / self.frequency

        return LayerLatency(
            layer_name=layer.name,
            layer_type=layer.layer_type.name,
            compute_cycles=0,
            memory_cycles=total_cycles,
            total_cycles=total_cycles,
            latency_us=latency_sec * 1e6,
            latency_ms=latency_sec * 1000,
            is_compute_bound=False,
            flops=0,
            achieved_gflops=0,
        )

    def _estimate_lightweight(
        self, layer, dtype: str, num_cores: int
    ) -> LayerLatency:
        """Estimate latency for lightweight operations (add, reshape)"""

        sew_bits = {"fp32": 32, "fp16": 16, "int8": 8}.get(dtype, 32)
        elements_per_vreg = self.vector.vlen // sew_bits

        compute_cycles = max(
            1,
            math.ceil(layer.flops / elements_per_vreg)
        )

        total_data = layer.total_memory_bytes
        memory_cycles = max(
            1,
            math.ceil(total_data / self.memory.l1d.bandwidth_bytes_per_cycle)
        )

        total_cycles = max(compute_cycles, memory_cycles)
        latency_sec = total_cycles / self.frequency

        return LayerLatency(
            layer_name=layer.name,
            layer_type=layer.layer_type.name,
            compute_cycles=compute_cycles,
            memory_cycles=memory_cycles,
            total_cycles=total_cycles,
            latency_us=latency_sec * 1e6,
            latency_ms=latency_sec * 1000,
            is_compute_bound=compute_cycles >= memory_cycles,
            flops=layer.flops,
            achieved_gflops=layer.flops / latency_sec / 1e9 if latency_sec > 0 else 0,
        )

    def _estimate_generic(
        self, layer, dtype: str, num_cores: int
    ) -> LayerLatency:
        """Generic estimation for unknown layer types"""
        return self._estimate_lightweight(layer, dtype, num_cores)

    def estimate_model(
        self,
        graph,
        dtype: str = "fp32",
        batch_size: int = 1,
        seq_len: int = 128,
        num_cores: int = 1
    ) -> ModelLatency:
        """
        Estimate end-to-end model latency.

        Args:
            graph: ComputationGraph object
            dtype: Data type for computation
            batch_size: Batch size
            seq_len: Sequence length
            num_cores: Number of cores to use

        Returns:
            ModelLatency with complete breakdown
        """
        layer_results = []
        total_cycles = 0
        total_flops = 0

        # Process layers in execution order
        execution_order = graph.get_execution_schedule()

        for layer in execution_order:
            layer_latency = self.estimate_layer(layer, dtype, num_cores)
            layer_results.append(layer_latency)
            total_cycles += layer_latency.total_cycles
            total_flops += layer_latency.flops

        # Calculate aggregate metrics
        total_latency_sec = total_cycles / self.frequency
        total_latency_ms = total_latency_sec * 1000

        # Throughput
        total_tokens_processed = batch_size * seq_len
        
        tokens_per_sec = (
            total_tokens_processed / total_latency_sec if total_latency_sec > 0 else 0
        )
        samples_per_sec = (
            batch_size / total_latency_sec if total_latency_sec > 0 else 0
        )

        # Hardware utilization
        peak = self.soc.get_peak_performance(dtype)
        peak_gflops = peak["system_gflops"]
        achieved_gflops = (
            total_flops / total_latency_sec / 1e9
            if total_latency_sec > 0 else 0
        )
        hw_utilization = (
            achieved_gflops / peak_gflops if peak_gflops > 0 else 0
        )

        return ModelLatency(
            model_name=graph.name,
            total_cycles=total_cycles,
            total_latency_ms=total_latency_ms,
            throughput_tokens_per_sec=tokens_per_sec,
            throughput_samples_per_sec=samples_per_sec,
            total_flops=total_flops,
            achieved_gflops=achieved_gflops,
            peak_gflops=peak_gflops,
            hardware_utilization=hw_utilization,
            layer_breakdown=layer_results,
            dtype=dtype,
            batch_size=batch_size,
            seq_len=seq_len,
        )


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from transformer_perf.hardware.soc import RISCVSoC
    from transformer_perf.models.parser import TransformerGraphParser

    # Create SoC
    soc = RISCVSoC(preset="mid-range")

    # Parse model
    parser = TransformerGraphParser(dtype="fp32")
    graph = parser.parse_from_preset("bert-base", batch_size=1, seq_len=128)

    # Estimate latency
    estimator = LatencyEstimator(soc)
    result = estimator.estimate_model(
        graph, dtype="fp32", batch_size=1, seq_len=128
    )

    result.print_report()