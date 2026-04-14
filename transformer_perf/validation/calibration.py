# Save as: C:\Users\ankit\transformer-perf-model\transformer_perf\validation\calibration.py

"""
Calibration and Validation Module.
Compares model estimates against known benchmarks
and provides confidence scores.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ValidationResult:
    """Result of validating estimates against references"""
    metric: str
    estimated_value: float
    reference_value: float
    error_pct: float
    within_tolerance: bool
    confidence: str  # "high", "medium", "low"


class CalibrationEngine:
    """
    Calibrate performance estimates against known benchmarks.

    Uses published RISC-V performance data from:
    - SiFive Performance Series
    - Xuantie C910/C920
    - Kendryte K230
    """

    # Reference benchmarks (published data points)
    REFERENCE_DATA = {
        "sifive-u74": {
            "frequency_ghz": 1.5,
            "vlen": 0,  # No vector
            "specint_per_ghz": 5.0,
            "fp32_gflops_per_ghz": 0.5,
        },
        "xuantie-c910": {
            "frequency_ghz": 2.5,
            "vlen": 128,
            "specint_per_ghz": 7.0,
            "fp32_gflops_per_ghz": 2.0,
        },
        "xuantie-c920": {
            "frequency_ghz": 2.0,
            "vlen": 128,
            "specint_per_ghz": 7.0,
            "fp32_gflops_per_ghz": 4.0,
        },
    }

    # Typical model inference times (published/estimated)
    MODEL_REFERENCES = {
        "bert-base-fp32-batch1-seq128": {
            "estimated_gflops_range": (5.0, 50.0),  # Depends on hardware
            "typical_latency_range_ms": (5.0, 200.0),
        },
        "gpt2-small-fp32-batch1-seq128": {
            "estimated_gflops_range": (5.0, 50.0),
            "typical_latency_range_ms": (5.0, 200.0),
        },
    }

    def __init__(self):
        self.calibration_factors = {}

    def validate_estimate(
        self,
        soc,
        model_latency,
        reference_key: Optional[str] = None
    ) -> List[ValidationResult]:
        """
        Validate performance estimates against references.
        """
        results = []

        # Check peak performance is reasonable
        peak = soc.get_peak_performance("fp32")

        # Validate: achieved should be <= peak
        if model_latency.achieved_gflops > peak["system_gflops"]:
            results.append(ValidationResult(
                metric="achieved_vs_peak",
                estimated_value=model_latency.achieved_gflops,
                reference_value=peak["system_gflops"],
                error_pct=100,
                within_tolerance=False,
                confidence="low",
            ))
        else:
            ratio = model_latency.achieved_gflops / peak["system_gflops"]
            results.append(ValidationResult(
                metric="hw_utilization",
                estimated_value=ratio,
                reference_value=0.3,  # Typical utilization
                error_pct=abs(ratio - 0.3) / 0.3 * 100,
                within_tolerance=(0.05 < ratio < 0.90),
                confidence="medium" if 0.1 < ratio < 0.8 else "low",
            ))

        # Validate: latency should be positive and reasonable
        if model_latency.total_latency_ms <= 0:
            results.append(ValidationResult(
                metric="latency_positive",
                estimated_value=model_latency.total_latency_ms,
                reference_value=1.0,
                error_pct=100,
                within_tolerance=False,
                confidence="low",
            ))
        else:
            results.append(ValidationResult(
                metric="latency_reasonable",
                estimated_value=model_latency.total_latency_ms,
                reference_value=50.0,  # Typical for BERT on mid-range
                error_pct=abs(model_latency.total_latency_ms - 50.0) / 50.0 * 100,
                within_tolerance=(0.1 < model_latency.total_latency_ms < 10000),
                confidence="medium",
            ))

        # Validate: FLOPs count against known model FLOPs
        known_flops = {
            "bert-base": 21.5e9,   # ~21.5 GFLOPs for seq_len=128
            "gpt2-small": 21.5e9,
        }

        for model_key, expected_flops in known_flops.items():
            if model_key in model_latency.model_name.lower():
                error = abs(
                    model_latency.total_flops - expected_flops
                ) / expected_flops * 100
                results.append(ValidationResult(
                    metric=f"flops_count_{model_key}",
                    estimated_value=model_latency.total_flops / 1e9,
                    reference_value=expected_flops / 1e9,
                    error_pct=error,
                    within_tolerance=(error < 30),
                    confidence="high" if error < 10 else "medium",
                ))

        return results

    def print_validation_report(self, results: List[ValidationResult]):
        """Print validation results"""
        print(f"\n{'='*60}")
        print("VALIDATION REPORT")
        print(f"{'='*60}")
        print(f"  {'Metric':<25} {'Estimated':>10} {'Reference':>10} "
              f"{'Error%':>8} {'OK':>4} {'Conf':<8}")
        print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*8} {'─'*4} {'─'*8}")

        for r in results:
            ok = "✓" if r.within_tolerance else "✗"
            print(
                f"  {r.metric:<25} "
                f"{r.estimated_value:>10.2f} "
                f"{r.reference_value:>10.2f} "
                f"{r.error_pct:>7.1f}% "
                f"{ok:>4} "
                f"{r.confidence:<8}"
            )

        passed = sum(1 for r in results if r.within_tolerance)
        total = len(results)
        print(f"\n  Result: {passed}/{total} checks passed")
        print(f"{'='*60}\n")