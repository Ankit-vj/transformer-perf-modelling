# Save as: C:\Users\ankit\transformer-perf-model\transformer_perf\optimizations\quantization.py

"""
Quantization Analysis.
Models the impact of quantization (FP32→FP16→INT8→INT4)
on performance, memory, and accuracy.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QuantizationConfig:
    """Quantization configuration"""
    source_dtype: str = "fp32"
    target_dtype: str = "int8"
    scheme: str = "symmetric"           # symmetric, asymmetric
    granularity: str = "per_tensor"     # per_tensor, per_channel
    calibration: str = "minmax"         # minmax, percentile, entropy

    @property
    def compression_ratio(self) -> float:
        """Compute compression ratio"""
        bits = {
            "fp32": 32, "fp16": 16, "bf16": 16,
            "int8": 8, "int4": 4, "int2": 2
        }
        src = bits.get(self.source_dtype, 32)
        tgt = bits.get(self.target_dtype, 8)
        return src / tgt


@dataclass
class QuantizationImpact:
    """Impact of quantization on a model"""
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

    # Layer-wise breakdown
    layer_impacts: List[Dict] = field(default_factory=list)

    def summary(self) -> Dict:
        return {
            "model": self.model_name,
            "quantization": f"{self.source_dtype} → {self.target_dtype}",
            "compression_ratio": f"{self.compression_ratio:.1f}x",
            "memory_original_mb": f"{self.original_weight_mb:.2f}",
            "memory_quantized_mb": f"{self.quantized_weight_mb:.2f}",
            "memory_savings": f"{self.memory_savings_pct:.1f}%",
            "latency_original_ms": f"{self.original_latency_ms:.4f}",
            "latency_quantized_ms": f"{self.quantized_latency_ms:.4f}",
            "speedup": f"{self.speedup:.2f}x",
            "est_accuracy_drop": f"{self.estimated_accuracy_drop_pct:.2f}%",
        }

    def print_report(self):
        print(f"\n{'='*60}")
        print(f"QUANTIZATION ANALYSIS: {self.model_name}")
        print(f"{'='*60}")
        for key, val in self.summary().items():
            print(f"  {key:<25}: {val}")
        print(f"{'='*60}\n")


class QuantizationAnalyzer:
    """
    Analyze the impact of quantization on model
    performance and memory on RISC-V hardware.
    """

    # Empirical accuracy drop estimates (conservative)
    ACCURACY_DROP_ESTIMATES = {
        ("fp32", "fp16"): 0.1,    # Minimal drop
        ("fp32", "bf16"): 0.2,    # Slightly more
        ("fp32", "int8"): 0.5,    # Small drop with good calibration
        ("fp32", "int4"): 2.0,    # Noticeable drop
        ("fp16", "int8"): 0.4,
        ("fp16", "int4"): 1.8,
    }

    # Speedup factors for different dtypes on RISC-V RVV
    SPEEDUP_FACTORS = {
        "fp32": 1.0,
        "fp16": 1.8,     # ~2x elements per vector, minus overhead
        "bf16": 1.7,
        "int8": 3.2,     # ~4x elements per vector, minus overhead
        "int4": 5.0,     # ~8x elements, significant overhead
    }

    def __init__(self, soc=None):
        self.soc = soc

    def analyze(
        self,
        graph,
        source_dtype: str = "fp32",
        target_dtype: str = "int8",
        latency_estimator=None
    ) -> QuantizationImpact:
        """
        Analyze quantization impact on a model.

        Args:
            graph: ComputationGraph
            source_dtype: Original data type
            target_dtype: Target quantized data type
            latency_estimator: LatencyEstimator for timing

        Returns:
            QuantizationImpact with analysis results
        """
        config = QuantizationConfig(
            source_dtype=source_dtype,
            target_dtype=target_dtype
        )

        # Memory analysis
        source_bpe = {"fp32": 4, "fp16": 2, "bf16": 2,
                      "int8": 1, "int4": 0.5}.get(source_dtype, 4)
        target_bpe = {"fp32": 4, "fp16": 2, "bf16": 2,
                      "int8": 1, "int4": 0.5}.get(target_dtype, 1)

        original_weights = graph.get_total_weights()
        quantized_weights = original_weights * (target_bpe / source_bpe)

        original_mb = original_weights / (1024 * 1024)
        quantized_mb = quantized_weights / (1024 * 1024)

        # Latency analysis
        speedup_factor = (
            self.SPEEDUP_FACTORS.get(target_dtype, 1.0)
            / self.SPEEDUP_FACTORS.get(source_dtype, 1.0)
        )

        original_latency = 0.0
        quantized_latency = 0.0

        if latency_estimator:
            original_result = latency_estimator.estimate_model(
                graph, dtype=source_dtype
            )
            original_latency = original_result.total_latency_ms
            quantized_latency = original_latency / speedup_factor
        else:
            original_latency = 1.0  # Placeholder
            quantized_latency = original_latency / speedup_factor

        # Accuracy impact
        accuracy_drop = self.ACCURACY_DROP_ESTIMATES.get(
            (source_dtype, target_dtype), 1.0
        )

        # Layer-wise analysis
        layer_impacts = []
        for node_id in graph.get_topological_order():
            node = graph.get_node(node_id)
            layer = node.layer
            if layer.weight_bytes > 0:
                orig_w = layer.weight_bytes
                quant_w = orig_w * (target_bpe / source_bpe)
                layer_impacts.append({
                    "name": layer.name,
                    "type": layer.layer_type.name,
                    "original_kb": orig_w / 1024,
                    "quantized_kb": quant_w / 1024,
                    "savings_pct": (1 - quant_w / orig_w) * 100 if orig_w > 0 else 0,
                })

        return QuantizationImpact(
            model_name=graph.name,
            source_dtype=source_dtype,
            target_dtype=target_dtype,
            compression_ratio=config.compression_ratio,
            original_weight_mb=original_mb,
            quantized_weight_mb=quantized_mb,
            memory_savings_mb=original_mb - quantized_mb,
            memory_savings_pct=(1 - quantized_mb / original_mb) * 100 if original_mb > 0 else 0,
            original_latency_ms=original_latency,
            quantized_latency_ms=quantized_latency,
            speedup=speedup_factor,
            estimated_accuracy_drop_pct=accuracy_drop,
            layer_impacts=layer_impacts,
        )

    def compare_quantization_options(
        self,
        graph,
        target_dtypes: List[str] = None,
        latency_estimator=None
    ) -> List[QuantizationImpact]:
        """Compare multiple quantization options"""

        if target_dtypes is None:
            target_dtypes = ["fp16", "int8", "int4"]

        results = []
        for dtype in target_dtypes:
            impact = self.analyze(
                graph, source_dtype="fp32",
                target_dtype=dtype,
                latency_estimator=latency_estimator
            )
            results.append(impact)

        return results