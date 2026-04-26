# Save as: C:\Users\ankit\transformer-perf-model\transformer_perf\estimators\latency.py

"""
Latency Estimation Engine.
Estimates end-to-end inference latency by combining
compute modeling, memory modeling, and scheduling.

FIXES applied
─────────────
1. tokens_per_sec  = (batch_size × seq_len) / latency_sec   [was seq_len only]
2. samples_per_sec = batch_size / latency_sec                [was 1.0 / latency_sec]
3. Core parallelism applied correctly in _estimate_compute_heavy
4. Weight memory is shared across the batch → weight_cycles
   divided by num_cores, not multiplied
5. Added _effective_num_cores() guard so we never exceed soc.num_cores
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


# ─────────────────────────────────────────────────────────────
#  Data-classes
# ─────────────────────────────────────────────────────────────

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
            "layer_name":     self.layer_name,
            "layer_type":     self.layer_type,
            "compute_cycles": self.compute_cycles,
            "memory_cycles":  self.memory_cycles,
            "total_cycles":   self.total_cycles,
            "latency_ms":     f"{self.latency_ms:.4f}",
            "bound":          "compute" if self.is_compute_bound else "memory",
            "achieved_gflops":f"{self.achieved_gflops:.2f}",
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
            "model_name":         self.model_name,
            "dtype":              self.dtype,
            "batch_size":         self.batch_size,
            "seq_len":            self.seq_len,
            "total_latency_ms":   f"{self.total_latency_ms:.4f}",
            # ── FIX: both throughput numbers now reflect batch ──
            "throughput_tokens/s":  f"{self.throughput_tokens_per_sec:.1f}",
            "throughput_samples/s": f"{self.throughput_samples_per_sec:.1f}",
            "total_gflops":       f"{self.total_flops / 1e9:.2f}",
            "achieved_gflops":    f"{self.achieved_gflops:.2f}",
            "peak_gflops":        f"{self.peak_gflops:.2f}",
            "hw_utilization":     f"{self.hardware_utilization:.1%}",
            "num_layers":         len(self.layer_breakdown),
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
        print(
            f"  {'Layer':<35} {'Type':<15} "
            f"{'Cycles':>12} {'ms':>10} {'Bound':<8}"
        )
        print(
            f"  {'─'*35} {'─'*15} "
            f"{'─'*12} {'─'*10} {'─'*8}"
        )

        sorted_layers = sorted(
            self.layer_breakdown,
            key=lambda x: x.total_cycles,
            reverse=True,
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

        # Layer-type summary
        print(f"\n  Summary by Layer Type:")
        print(
            f"  {'Type':<20} {'Count':>6} "
            f"{'Total ms':>12} {'% of Total':>12}"
        )
        print(
            f"  {'─'*20} {'─'*6} "
            f"{'─'*12} {'─'*12}"
        )

        type_stats: Dict[str, Dict] = {}
        for layer in self.layer_breakdown:
            lt = layer.layer_type
            if lt not in type_stats:
                type_stats[lt] = {"count": 0, "ms": 0.0}
            type_stats[lt]["count"] += 1
            type_stats[lt]["ms"]    += layer.latency_ms

        for lt, stats in sorted(
            type_stats.items(),
            key=lambda x: x[1]["ms"],
            reverse=True,
        ):
            pct = (
                stats["ms"] / self.total_latency_ms * 100
                if self.total_latency_ms > 0 else 0
            )
            print(
                f"  {lt:<20} {stats['count']:>6} "
                f"{stats['ms']:>12.4f} {pct:>11.1f}%"
            )

        print(f"{'='*80}\n")


# ─────────────────────────────────────────────────────────────
#  Estimator
# ─────────────────────────────────────────────────────────────

class LatencyEstimator:
    """
    Estimates end-to-end inference latency for transformer
    models on RISC-V SoC hardware.

    Combines:
    • Compute cycle estimation  (vector throughput model)
    • Memory access cycle estimation (cache / DRAM model)
    • Takes max(compute, memory) per layer
    • Correctly accounts for batch parallelism in throughput
    """

    def __init__(self, soc):
        """
        Args:
            soc: RISCVSoC object with hardware configuration
        """
        self.soc       = soc
        self.core      = soc.cores[0]
        self.vector    = soc.vector
        self.memory    = soc.memory
        self.frequency = soc.frequency
        self.num_cores = soc.num_cores

    # ── helpers ───────────────────────────────────────────────

    def _effective_cores(self, requested: int) -> int:
        """Clamp requested cores to available SoC cores."""
        return max(1, min(requested, self.num_cores))

    def _sew_bits(self, dtype: str) -> int:
        return {"fp32": 32, "fp16": 16, "bf16": 16, "int8": 8}.get(
            dtype, 32
        )

    def _bytes_per_element(self, dtype: str) -> int:
        return self._sew_bits(dtype) // 8

    def _fma_params(self):
        """Return (latency, throughput) for the VFMA unit."""
        fma_unit = self.vector.vector_units.get("VFMA")
        if fma_unit:
            return fma_unit.latency, fma_unit.throughput
        return 3, 1.0

    def _weight_memory_cycles(self, weight_bytes: int) -> int:
        """
        Cycles to load weight data from the appropriate cache level.
        Weights are shared across the whole batch – they are loaded
        once per forward pass, not once per sample.
        """
        mem = self.memory
        if weight_bytes <= mem.l2.size_bytes:
            bw      = mem.l2.bandwidth_bytes_per_cycle
            latency = mem.l2.latency_cycles
        elif weight_bytes <= mem.l3.size_bytes:
            bw      = mem.l3.bandwidth_bytes_per_cycle
            latency = mem.l3.latency_cycles
        else:
            bw      = (
                mem.dram.peak_bandwidth_gbps * 1e9 / self.frequency
            )
            latency = mem.dram.latency_cycles(self.frequency)

        return latency + math.ceil(weight_bytes / bw)

    def _activation_memory_cycles(self, activation_bytes: int) -> int:
        """Cycles to stream activation data (usually L1D / L2)."""
        mem = self.memory
        if activation_bytes <= mem.l1d.size_bytes:
            bw = mem.l1d.bandwidth_bytes_per_cycle
        else:
            bw = mem.l2.bandwidth_bytes_per_cycle
        return math.ceil(activation_bytes / bw)

    # ── per-layer estimators ──────────────────────────────────

    def estimate_layer(
        self,
        layer,
        dtype: str = "fp32",
        num_cores: int = 1,
    ) -> LayerLatency:
        """
        Dispatch to the correct estimation method based on layer type.
        """
        layer_type = layer.layer_type.name

        if layer_type in ("LINEAR", "ATTENTION", "MLP"):
            return self._estimate_compute_heavy(layer, dtype, num_cores)
        elif layer_type in ("LAYERNORM", "RMSNORM"):
            return self._estimate_normalization(layer, dtype, num_cores)
        elif layer_type == "SOFTMAX":
            return self._estimate_softmax(layer, dtype, num_cores)
        elif layer_type == "EMBEDDING":
            return self._estimate_memory_only(layer, dtype)
        elif layer_type in ("RESIDUAL_ADD", "RESHAPE", "TRANSPOSE"):
            return self._estimate_lightweight(layer, dtype, num_cores)
        else:
            return self._estimate_generic(layer, dtype, num_cores)

    def _estimate_compute_heavy(
        self, layer, dtype: str, num_cores: int
    ) -> LayerLatency:
        """
        Estimate latency for compute-heavy layers (Linear, Attention, MLP).

        Key corrections vs original
        ───────────────────────────
        • FLOPs/cycle scaled by effective core count
        • Weight cycles NOT multiplied by batch (weights are shared)
        • Activation cycles scale with batch (activations are per-sample)
        """
        sew           = self._sew_bits(dtype)
        elements_per_vreg = self.vector.vlen // sew
        fma_latency, fma_throughput = self._fma_params()
        cores = self._effective_cores(num_cores)

        # ── Compute cycles ────────────────────────────────────
        # 2 FLOPs per FMA × elements processed per cycle × num_cores
        flops_per_cycle = elements_per_vreg * fma_throughput * 2 * cores
        compute_cycles  = (
            math.ceil(layer.flops / flops_per_cycle) + fma_latency
        )

        # ── Memory cycles ─────────────────────────────────────
        # Weights: loaded ONCE for the whole batch (shared)
        weight_cycles = self._weight_memory_cycles(layer.weight_bytes)

        # Activations + outputs scale with batch size (already baked
        # into layer.activation_bytes by the profiler)
        activation_cycles = self._activation_memory_cycles(
            layer.activation_bytes + layer.output_bytes
        )

        memory_cycles = weight_cycles + activation_cycles

        # ── Totals ────────────────────────────────────────────
        total_cycles      = max(compute_cycles, memory_cycles)
        is_compute_bound  = compute_cycles >= memory_cycles
        latency_sec       = total_cycles / self.frequency
        achieved_gflops   = (
            layer.flops / latency_sec / 1e9 if latency_sec > 0 else 0.0
        )

        return LayerLatency(
            layer_name       = layer.name,
            layer_type       = layer.layer_type.name,
            compute_cycles   = compute_cycles,
            memory_cycles    = memory_cycles,
            total_cycles     = total_cycles,
            latency_us       = latency_sec * 1e6,
            latency_ms       = latency_sec * 1_000,
            is_compute_bound = is_compute_bound,
            flops            = layer.flops,
            achieved_gflops  = achieved_gflops,
        )

    def _estimate_normalization(
        self, layer, dtype: str, num_cores: int
    ) -> LayerLatency:
        """LayerNorm / RMSNorm – reduction-heavy → 50 % vector efficiency."""
        sew               = self._sew_bits(dtype)
        elements_per_vreg = self.vector.vlen // sew
        cores             = self._effective_cores(num_cores)

        # Reductions cut effective throughput to ~50 %
        effective_throughput = elements_per_vreg * 0.5 * cores
        compute_cycles       = math.ceil(
            layer.flops / effective_throughput
        )

        memory_cycles = self._activation_memory_cycles(
            layer.total_memory_bytes
        )

        total_cycles  = max(compute_cycles, memory_cycles)
        latency_sec   = total_cycles / self.frequency

        return LayerLatency(
            layer_name       = layer.name,
            layer_type       = layer.layer_type.name,
            compute_cycles   = compute_cycles,
            memory_cycles    = memory_cycles,
            total_cycles     = total_cycles,
            latency_us       = latency_sec * 1e6,
            latency_ms       = latency_sec * 1_000,
            is_compute_bound = compute_cycles >= memory_cycles,
            flops            = layer.flops,
            achieved_gflops  = (
                layer.flops / latency_sec / 1e9 if latency_sec > 0 else 0.0
            ),
        )

    def _estimate_softmax(
        self, layer, dtype: str, num_cores: int
    ) -> LayerLatency:
        """
        Softmax – transcendental ops + reductions → 30 % vector efficiency.
        """
        sew               = self._sew_bits(dtype)
        elements_per_vreg = self.vector.vlen // sew
        cores             = self._effective_cores(num_cores)

        effective_throughput = elements_per_vreg * 0.3 * cores
        compute_cycles       = math.ceil(
            layer.flops / effective_throughput
        )

        memory_cycles = self._activation_memory_cycles(
            layer.total_memory_bytes
        )

        total_cycles  = max(compute_cycles, memory_cycles)
        latency_sec   = total_cycles / self.frequency

        return LayerLatency(
            layer_name       = layer.name,
            layer_type       = layer.layer_type.name,
            compute_cycles   = compute_cycles,
            memory_cycles    = memory_cycles,
            total_cycles     = total_cycles,
            latency_us       = latency_sec * 1e6,
            latency_ms       = latency_sec * 1_000,
            is_compute_bound = compute_cycles >= memory_cycles,
            flops            = layer.flops,
            achieved_gflops  = (
                layer.flops / latency_sec / 1e9 if latency_sec > 0 else 0.0
            ),
        )

    def _estimate_memory_only(self, layer, dtype: str) -> LayerLatency:
        """Embedding – pure memory lookup, zero compute."""
        total_data = layer.weight_bytes + layer.output_bytes
        mem_result = self.memory.estimate_data_access_cycles(
            data_size_bytes    = total_data,
            core_frequency_hz  = self.frequency,
        )
        total_cycles = int(mem_result["total_cycles"])
        latency_sec  = total_cycles / self.frequency

        return LayerLatency(
            layer_name       = layer.name,
            layer_type       = layer.layer_type.name,
            compute_cycles   = 0,
            memory_cycles    = total_cycles,
            total_cycles     = total_cycles,
            latency_us       = latency_sec * 1e6,
            latency_ms       = latency_sec * 1_000,
            is_compute_bound = False,
            flops            = 0,
            achieved_gflops  = 0.0,
        )

    def _estimate_lightweight(
        self, layer, dtype: str, num_cores: int
    ) -> LayerLatency:
        """Residual add, reshape, transpose – very cheap."""
        sew               = self._sew_bits(dtype)
        elements_per_vreg = self.vector.vlen // sew
        cores             = self._effective_cores(num_cores)

        compute_cycles = max(
            1,
            math.ceil(layer.flops / (elements_per_vreg * cores)),
        )
        memory_cycles  = max(
            1,
            self._activation_memory_cycles(layer.total_memory_bytes),
        )

        total_cycles  = max(compute_cycles, memory_cycles)
        latency_sec   = total_cycles / self.frequency

        return LayerLatency(
            layer_name       = layer.name,
            layer_type       = layer.layer_type.name,
            compute_cycles   = compute_cycles,
            memory_cycles    = memory_cycles,
            total_cycles     = total_cycles,
            latency_us       = latency_sec * 1e6,
            latency_ms       = latency_sec * 1_000,
            is_compute_bound = compute_cycles >= memory_cycles,
            flops            = layer.flops,
            achieved_gflops  = (
                layer.flops / latency_sec / 1e9 if latency_sec > 0 else 0.0
            ),
        )

    def _estimate_generic(
        self, layer, dtype: str, num_cores: int
    ) -> LayerLatency:
        """Fall-through for unknown layer types."""
        return self._estimate_lightweight(layer, dtype, num_cores)

    # ── model-level estimator ─────────────────────────────────

    def estimate_model(
        self,
        graph,
        dtype: str      = "fp32",
        batch_size: int = 1,
        seq_len: int    = 128,
        num_cores: int  = 1,
    ) -> ModelLatency:
        """
        Estimate end-to-end model latency.

        Throughput fix
        ──────────────
        Old (wrong):
            tokens_per_sec  = seq_len / total_latency_sec
            samples_per_sec = 1.0    / total_latency_sec

        New (correct):
            tokens_per_sec  = (batch_size * seq_len) / total_latency_sec
            samples_per_sec = batch_size             / total_latency_sec

        Rationale: the model processes `batch_size` sequences of length
        `seq_len` in one forward pass that takes `total_latency_sec`.
        Dividing only seq_len (and ignoring batch_size) made throughput
        *decrease* as batch_size grew, which is the opposite of reality.

        Args:
            graph:      ComputationGraph object
            dtype:      Data type for computation
            batch_size: Number of sequences in this forward pass
            seq_len:    Length of each sequence
            num_cores:  Number of cores to use

        Returns:
            ModelLatency with complete breakdown
        """
        layer_results: List[LayerLatency] = []
        total_cycles  = 0
        total_flops   = 0

        for layer in graph.get_execution_schedule():
            ll = self.estimate_layer(layer, dtype, num_cores)
            layer_results.append(ll)
            total_cycles += ll.total_cycles
            total_flops  += ll.flops

        # ── Aggregate timing ──────────────────────────────────
        total_latency_sec = total_cycles / self.frequency
        total_latency_ms  = total_latency_sec * 1_000

        # ── FIX: correct throughput accounting ───────────────
        # One forward pass → batch_size samples, each of seq_len tokens
        if total_latency_sec > 0:
            tokens_per_sec  = (batch_size * seq_len) / total_latency_sec
            samples_per_sec = batch_size             / total_latency_sec
        else:
            tokens_per_sec  = 0.0
            samples_per_sec = 0.0

        # ── Hardware utilisation ──────────────────────────────
        peak       = self.soc.get_peak_performance(dtype)
        peak_gflops = peak["system_gflops"]
        achieved_gflops = (
            total_flops / total_latency_sec / 1e9
            if total_latency_sec > 0 else 0.0
        )
        hw_utilization = (
            achieved_gflops / peak_gflops if peak_gflops > 0 else 0.0
        )

        return ModelLatency(
            model_name                 = graph.name,
            total_cycles               = total_cycles,
            total_latency_ms           = total_latency_ms,
            throughput_tokens_per_sec  = tokens_per_sec,
            throughput_samples_per_sec = samples_per_sec,
            total_flops                = total_flops,
            achieved_gflops            = achieved_gflops,
            peak_gflops                = peak_gflops,
            hardware_utilization       = hw_utilization,
            layer_breakdown            = layer_results,
            dtype                      = dtype,
            batch_size                 = batch_size,
            seq_len                    = seq_len,
        )


# ════════════════════════════════════════════════════════════
#  Quick smoke-test
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from hardware.soc import RISCVSoC
    from models.parser import TransformerGraphParser

    soc    = RISCVSoC(preset="mid-range")
    parser = TransformerGraphParser(dtype="fp32")

    print("=" * 65)
    print("Throughput scaling with batch size (should INCREASE)")
    print("=" * 65)
    print(
        f"  {'Batch':>6}  {'Latency(ms)':>12}  "
        f"{'Tokens/s':>12}  {'Samples/s':>12}"
    )
    print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*12}")

    estimator = LatencyEstimator(soc)
    for bs in [1, 2, 4, 8, 16]:
        graph  = parser.parse_from_preset(
            "bert-base", batch_size=bs, seq_len=128
        )
        result = estimator.estimate_model(
            graph, dtype="fp32",
            batch_size=bs, seq_len=128,
        )
        print(
            f"  {bs:>6}  "
            f"{result.total_latency_ms:>12.2f}  "
            f"{result.throughput_tokens_per_sec:>12.1f}  "
            f"{result.throughput_samples_per_sec:>12.2f}"
        )

    print()
    print("Full report for bs=1:")
    graph  = parser.parse_from_preset(
        "bert-base", batch_size=1, seq_len=128
    )
    result = estimator.estimate_model(
        graph, dtype="fp32", batch_size=1, seq_len=128
    )
    result.print_report()