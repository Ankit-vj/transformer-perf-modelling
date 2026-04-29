"""
Quantization Analysis — Roofline-Aware.

Speedup is computed per-layer from first principles using the
roofline model, NOT from a global lookup table.

Core insight:
  - Memory-bound layer  (AI < ridge): fewer bytes to load → proportional BW gain
  - Compute-bound layer (AI ≥ ridge): more elements per RVV register → compute gain

For each layer we:
  1. Compute its arithmetic intensity (AI = FLOPs / bytes) under source dtype
  2. Re-compute AI under target dtype (weights shrink; activations may too)
  3. Determine which roofline ceiling applies for each dtype
  4. Derive speedup as the ratio of the two achievable performances
  5. Weight layer speedups by their share of total model latency
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes (unchanged interface so the rest of the codebase still works)
# ---------------------------------------------------------------------------

@dataclass
class QuantizationConfig:
    """Quantization configuration"""
    source_dtype: str = "fp32"
    target_dtype: str = "int8"
    scheme: str = "symmetric"
    granularity: str = "per_tensor"
    calibration: str = "minmax"

    @property
    def compression_ratio(self) -> float:
        bits = {"fp32": 32, "fp16": 16, "bf16": 16,
                "int8": 8,  "int4": 4,  "int2": 2}
        src = bits.get(self.source_dtype, 32)
        tgt = bits.get(self.target_dtype, 8)
        return src / tgt


@dataclass
class LayerQuantResult:
    """Per-layer quantization result (new — used internally and exposed)"""
    name: str
    layer_type: str
    source_ai: float          # Arithmetic intensity under source dtype
    target_ai: float          # Arithmetic intensity under target dtype
    source_perf: float        # Achievable GFLOP/s under source dtype
    target_perf: float        # Achievable GFLOP/s under target dtype
    layer_speedup: float      # target_perf / source_perf
    bound_type_src: str       # "memory" | "compute" under source
    bound_type_tgt: str       # "memory" | "compute" under target
    weight_fraction: float    # Fraction of model latency this layer takes


@dataclass
class QuantizationImpact:
    """Impact of quantization on a model (same interface as before)"""
    model_name: str
    source_dtype: str
    target_dtype: str
    compression_ratio: float

    # Memory impact
    original_weight_mb: float = 0.0
    quantized_weight_mb: float = 0.0
    memory_savings_mb: float = 0.0
    memory_savings_pct: float = 0.0

    # Performance impact
    original_latency_ms: float = 0.0
    quantized_latency_ms: float = 0.0
    speedup: float = 1.0

    # Estimated accuracy impact
    estimated_accuracy_drop_pct: float = 0.0

    # NEW — per-layer detail
    layer_impacts: List[Dict] = field(default_factory=list)
    layer_quant_results: List[LayerQuantResult] = field(default_factory=list)

    # NEW — breakdown of speedup source
    memory_bound_speedup_contribution: float = 0.0
    compute_bound_speedup_contribution: float = 0.0
    pct_layers_memory_bound: float = 0.0

    def summary(self) -> Dict:
        return {
            "model":              self.model_name,
            "quantization":       f"{self.source_dtype} → {self.target_dtype}",
            "compression_ratio":  f"{self.compression_ratio:.1f}x",
            "memory_original_mb": f"{self.original_weight_mb:.2f}",
            "memory_quantized_mb":f"{self.quantized_weight_mb:.2f}",
            "memory_savings":     f"{self.memory_savings_pct:.1f}%",
            "latency_original_ms":f"{self.original_latency_ms:.4f}",
            "latency_quantized_ms":f"{self.quantized_latency_ms:.4f}",
            "speedup":            f"{self.speedup:.2f}x",
            "est_accuracy_drop":  f"{self.estimated_accuracy_drop_pct:.2f}%",
        }

    def print_report(self):
        print(f"\n{'='*60}")
        print(f"QUANTIZATION ANALYSIS: {self.model_name}")
        print(f"{'='*60}")
        for key, val in self.summary().items():
            print(f"  {key:<25}: {val}")
        if self.layer_quant_results:
            mem_layers = [r for r in self.layer_quant_results
                          if r.bound_type_src == "memory"]
            cmp_layers = [r for r in self.layer_quant_results
                          if r.bound_type_src == "compute"]
            print(f"\n  Memory-bound layers : {len(mem_layers)} "
                  f"({self.pct_layers_memory_bound:.0f}%)")
            print(f"  Compute-bound layers: {len(cmp_layers)}")
            print(f"  Speedup from memory-bound ops : "
                  f"{self.memory_bound_speedup_contribution:.2f}x contribution")
            print(f"  Speedup from compute-bound ops: "
                  f"{self.compute_bound_speedup_contribution:.2f}x contribution")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Dtype constants
# ---------------------------------------------------------------------------

BYTES_PER_ELEMENT: Dict[str, float] = {
    "fp32": 4.0, "fp16": 2.0, "bf16": 2.0,
    "int8": 1.0, "int4": 0.5,
}

# How many more elements fit in the same RVV vector register vs fp32
# fp32 → 1 element/slot, fp16 → 2, int8 → 4, int4 → 8
RVV_ELEMENTS_RATIO: Dict[str, float] = {
    "fp32": 1.0, "fp16": 2.0, "bf16": 2.0,
    "int8": 4.0, "int4": 8.0,
}

# Vectorization efficiency on RVV per dtype
# INT types need dequant overhead; narrow types have better packing but
# non-native FMA pipelines on most RVV cores
RVV_EFFICIENCY: Dict[str, float] = {
    "fp32": 0.85,   # Native FP32 FMA
    "fp16": 0.80,   # FP16 FMA (some cores emulate)
    "bf16": 0.78,
    "int8": 0.70,   # INT8 MACs + dequant overhead
    "int4": 0.55,   # Significant packing/unpacking overhead
}

# Conservative empirical accuracy drop (%) from published BERT/GPT-2 studies
# Sources: SmoothQuant, GPTQ, HuggingFace quantization docs
ACCURACY_DROP: Dict[Tuple[str, str], float] = {
    ("fp32", "fp16"): 0.10,
    ("fp32", "bf16"): 0.15,
    ("fp32", "int8"): 0.50,
    ("fp32", "int4"): 2.50,   # More realistic than 2.0
    ("fp16", "int8"): 0.40,
    ("fp16", "int4"): 1.80,
}


# ---------------------------------------------------------------------------
# Core roofline-aware speedup engine
# ---------------------------------------------------------------------------

class RooflineQuantEngine:
    """
    Computes quantization speedup for each layer using the roofline model.

    For a layer with FLOPs F and memory traffic B (bytes):
      AI_src = F / B_src
      AI_tgt = F / B_tgt   (B_tgt < B_src because weights are smaller)

    Achievable performance:
      perf(AI, dtype) = min(peak_compute(dtype),  peak_bw * AI)

    Layer speedup = perf_tgt / perf_src
    """

    def __init__(self, soc):
        self.soc = soc

    def _peak_compute(self, dtype: str) -> float:
        """
        Peak GFLOP/s for a given dtype on this SoC.
        Scales linearly with RVV element ratio and efficiency.
        """
        base_peak = self.soc.get_peak_performance("fp32")["system_gflops"]
        ratio = RVV_ELEMENTS_RATIO.get(dtype, 1.0)
        eff   = RVV_EFFICIENCY.get(dtype, 0.7)
        src_eff = RVV_EFFICIENCY["fp32"]
        return base_peak * (ratio * eff / src_eff)

    def _peak_bw(self) -> float:
        """Peak DRAM bandwidth (GB/s) — dtype-independent for this SoC."""
        return self.soc.get_peak_performance("fp32")["dram_bandwidth_gbps"]

    def _achievable_perf(self, ai: float, dtype: str) -> float:
        """GFLOP/s achievable at a given arithmetic intensity."""
        peak_c = self._peak_compute(dtype)
        peak_b = self._peak_bw()
        return min(peak_c, peak_b * ai)

    def _ridge_point(self, dtype: str) -> float:
        """FLOP/byte at the roofline knee for a given dtype."""
        return self._peak_compute(dtype) / self._peak_bw()

    def layer_speedup(
        self,
        layer,
        source_dtype: str,
        target_dtype: str,
    ) -> LayerQuantResult:
        """
        Compute roofline-aware speedup for one layer.

        Quantization changes:
          - Weight bytes: always shrink by bytes_ratio
          - Activation bytes: shrink only if activations are also quantized
            (conservative: assume activations stay in source dtype for
             weight-only quantization, halved for full quantization)
        """
        flops      = layer.flops
        w_bytes    = layer.weight_bytes
        act_bytes  = layer.activation_bytes
        out_bytes  = layer.output_bytes

        src_bpe = BYTES_PER_ELEMENT.get(source_dtype, 4.0)
        tgt_bpe = BYTES_PER_ELEMENT.get(target_dtype, 1.0)
        w_ratio = tgt_bpe / src_bpe        # e.g. fp16/fp32 = 0.5

        # Source memory traffic
        b_src = w_bytes + act_bytes + out_bytes
        if b_src == 0:
            # No memory traffic (reshape etc.) → speedup = 1
            return LayerQuantResult(
                name=layer.name, layer_type=layer.layer_type.name,
                source_ai=0, target_ai=0,
                source_perf=0, target_perf=0,
                layer_speedup=1.0,
                bound_type_src="memory", bound_type_tgt="memory",
                weight_fraction=0.0,
            )

        # Target memory traffic
        # Weights shrink fully; activations shrink for full quantization
        # (int8/int4), stay same for fp16 in a typical weight-only scheme.
        # We use a conservative activation_shrink_ratio:
        if target_dtype in ("int8", "int4"):
            act_shrink = w_ratio   # Full quantization — activations also quantized
        else:
            act_shrink = 1.0       # fp16/bf16 — activations stay fp32

        b_tgt = (w_bytes * w_ratio
                 + act_bytes * act_shrink
                 + out_bytes * act_shrink)

        # Arithmetic intensities
        ai_src = flops / b_src if b_src > 0 else 0.0
        ai_tgt = flops / b_tgt if b_tgt > 0 else 0.0

        # Achievable performance
        perf_src = self._achievable_perf(ai_src, source_dtype)
        perf_tgt = self._achievable_perf(ai_tgt, target_dtype)

        layer_speedup = perf_tgt / perf_src if perf_src > 0 else 1.0

        # Bound type
        ridge_src = self._ridge_point(source_dtype)
        ridge_tgt = self._ridge_point(target_dtype)
        bound_src = "compute" if ai_src >= ridge_src else "memory"
        bound_tgt = "compute" if ai_tgt >= ridge_tgt else "memory"

        return LayerQuantResult(
            name=layer.name,
            layer_type=layer.layer_type.name,
            source_ai=round(ai_src, 3),
            target_ai=round(ai_tgt, 3),
            source_perf=round(perf_src, 3),
            target_perf=round(perf_tgt, 3),
            layer_speedup=round(layer_speedup, 4),
            bound_type_src=bound_src,
            bound_type_tgt=bound_tgt,
            weight_fraction=0.0,   # filled in by model-level analysis
        )


# ---------------------------------------------------------------------------
# Public analyzer class (same API as before)
# ---------------------------------------------------------------------------

class QuantizationAnalyzer:
    """
    Analyze quantization impact using per-layer roofline reasoning.

    The model-level speedup is the latency-weighted harmonic mean of
    per-layer speedups (because latency adds, not multiplies).
    """

    def __init__(self, soc=None):
        self.soc  = soc
        self._engine = RooflineQuantEngine(soc) if soc else None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        graph,
        source_dtype: str = "fp32",
        target_dtype: str = "int8",
        latency_estimator=None,
    ) -> QuantizationImpact:

        config = QuantizationConfig(
            source_dtype=source_dtype,
            target_dtype=target_dtype,
        )

        # ── Memory analysis (exact, unchanged) ──────────────────────
        src_bpe = BYTES_PER_ELEMENT.get(source_dtype, 4.0)
        tgt_bpe = BYTES_PER_ELEMENT.get(target_dtype, 1.0)

        original_weights  = graph.get_total_weights()
        quantized_weights = original_weights * (tgt_bpe / src_bpe)

        original_mb  = original_weights  / (1024 * 1024)
        quantized_mb = quantized_weights / (1024 * 1024)

        # ── Speedup: roofline-aware or fallback ──────────────────────
        original_latency  = 0.0
        quantized_latency = 0.0
        layer_quant_results: List[LayerQuantResult] = []
        speedup = 1.0

        if latency_estimator and self._engine:
            # Get baseline latency per layer
            lat_result = latency_estimator.estimate_model(
                graph, dtype=source_dtype
            )
            original_latency = lat_result.total_latency_ms

            # Per-layer roofline speedup
            total_latency_for_weight = sum(
                lr.layer_latency_ms
                for lr in lat_result.layer_results
                if hasattr(lr, "layer_latency_ms")
            ) or original_latency

            layer_lookup = {}
            if hasattr(lat_result, "layer_results"):
                for lr in lat_result.layer_results:
                    if hasattr(lr, "layer_name") and hasattr(lr, "layer_latency_ms"):
                        layer_lookup[lr.layer_name] = lr.layer_latency_ms

            # Walk graph and compute per-layer speedup
            quantized_latency = 0.0
            mem_speedup_contrib  = 0.0
            comp_speedup_contrib = 0.0

            for node_id in graph.get_topological_order():
                node  = graph.get_node(node_id)
                layer = node.layer

                lq = self._engine.layer_speedup(
                    layer, source_dtype, target_dtype
                )

                layer_lat = layer_lookup.get(layer.name, 0.0)
                if layer_lat == 0.0 and original_latency > 0:
                    # Fallback: split evenly (shouldn't happen if lat_result is rich)
                    n_layers = len(graph.get_topological_order())
                    layer_lat = original_latency / n_layers

                lq.weight_fraction = (
                    layer_lat / original_latency if original_latency > 0 else 0.0
                )
                layer_quant_results.append(lq)

                quantized_layer_lat = layer_lat / lq.layer_speedup
                quantized_latency  += quantized_layer_lat

                if lq.bound_type_src == "memory":
                    mem_speedup_contrib  += lq.weight_fraction * lq.layer_speedup
                else:
                    comp_speedup_contrib += lq.weight_fraction * lq.layer_speedup

            speedup = (
                original_latency / quantized_latency
                if quantized_latency > 0 else 1.0
            )

            mem_layers  = [r for r in layer_quant_results
                           if r.bound_type_src == "memory"]
            pct_mem = (
                len(mem_layers) / len(layer_quant_results) * 100
                if layer_quant_results else 0.0
            )

        else:
            # No SoC or no latency estimator — use conservative analytic estimate
            speedup, original_latency, quantized_latency = (
                self._analytic_speedup(source_dtype, target_dtype,
                                       latency_estimator, graph)
            )
            mem_speedup_contrib  = 0.0
            comp_speedup_contrib = 0.0
            pct_mem = 0.0

        # ── Accuracy drop (literature-based) ────────────────────────
        accuracy_drop = ACCURACY_DROP.get(
            (source_dtype, target_dtype), 1.5
        )

        # ── Per-layer memory table (for the UI) ─────────────────────
        layer_impacts = self._build_layer_table(
            graph, src_bpe, tgt_bpe, layer_quant_results
        )

        return QuantizationImpact(
            model_name=graph.name,
            source_dtype=source_dtype,
            target_dtype=target_dtype,
            compression_ratio=config.compression_ratio,
            original_weight_mb=original_mb,
            quantized_weight_mb=quantized_mb,
            memory_savings_mb=original_mb - quantized_mb,
            memory_savings_pct=(
                (1 - quantized_mb / original_mb) * 100
                if original_mb > 0 else 0.0
            ),
            original_latency_ms=original_latency,
            quantized_latency_ms=quantized_latency,
            speedup=round(speedup, 2),
            estimated_accuracy_drop_pct=accuracy_drop,
            layer_impacts=layer_impacts,
            layer_quant_results=layer_quant_results,
            memory_bound_speedup_contribution=round(mem_speedup_contrib, 3),
            compute_bound_speedup_contribution=round(comp_speedup_contrib, 3),
            pct_layers_memory_bound=round(pct_mem, 1),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _analytic_speedup(
        self,
        source_dtype: str,
        target_dtype: str,
        latency_estimator,
        graph,
    ) -> Tuple[float, float, float]:
        """
        Fallback when no SoC is attached.
        Uses bandwidth-ratio for memory-bound assumption (conservative).
        """
        src_bpe = BYTES_PER_ELEMENT.get(source_dtype, 4.0)
        tgt_bpe = BYTES_PER_ELEMENT.get(target_dtype, 1.0)

        # Pure memory-bound assumption → speedup ∝ byte ratio
        # Discounted by 0.75 to account for compute-bound layers and overhead
        speedup = (src_bpe / tgt_bpe) * 0.75

        # Clamp to plausible range
        speedup = max(1.0, min(speedup, src_bpe / tgt_bpe))

        original_latency  = 0.0
        quantized_latency = 0.0

        if latency_estimator:
            result = latency_estimator.estimate_model(graph, dtype=source_dtype)
            original_latency  = result.total_latency_ms
            quantized_latency = original_latency / speedup

        return speedup, original_latency, quantized_latency

    def _build_layer_table(
        self,
        graph,
        src_bpe: float,
        tgt_bpe: float,
        layer_quant_results: List[LayerQuantResult],
    ) -> List[Dict]:
        """Build per-layer memory impact table (for web UI display)."""
        lqr_map = {r.name: r for r in layer_quant_results}
        impacts = []

        for node_id in graph.get_topological_order():
            node  = graph.get_node(node_id)
            layer = node.layer
            if layer.weight_bytes > 0:
                orig_w  = layer.weight_bytes
                quant_w = orig_w * (tgt_bpe / src_bpe)
                lqr     = lqr_map.get(layer.name)
                entry = {
                    "name":         layer.name,
                    "type":         layer.layer_type.name,
                    "original_kb":  round(orig_w  / 1024, 2),
                    "quantized_kb": round(quant_w / 1024, 2),
                    "savings_pct":  round(
                        (1 - quant_w / orig_w) * 100 if orig_w > 0 else 0, 1
                    ),
                }
                if lqr:
                    entry["layer_speedup"]    = round(lqr.layer_speedup, 2)
                    entry["bound_type"]        = lqr.bound_type_src
                    entry["source_ai"]         = round(lqr.source_ai, 2)
                    entry["target_ai"]         = round(lqr.target_ai, 2)
                impacts.append(entry)

        return impacts

    def compare_quantization_options(
        self,
        graph,
        target_dtypes: List[str] = None,
        latency_estimator=None,
    ) -> List[QuantizationImpact]:
        """Compare multiple quantization targets."""
        if target_dtypes is None:
            target_dtypes = ["fp16", "int8", "int4"]

        return [
            self.analyze(
                graph,
                source_dtype="fp32",
                target_dtype=dt,
                latency_estimator=latency_estimator,
            )
            for dt in target_dtypes
        ]