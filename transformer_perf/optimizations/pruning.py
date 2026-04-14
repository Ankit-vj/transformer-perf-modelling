# Save as: C:\Users\ankit\transformer-perf-model\transformer_perf\optimizations\pruning.py

"""
Pruning Analysis.
Models the impact of weight pruning (structured/unstructured)
on performance and memory.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PruningImpact:
    """Impact of pruning on model"""
    model_name: str
    pruning_type: str           # "unstructured", "structured"
    sparsity: float             # 0.0 to 1.0
    original_params: int = 0
    pruned_params: int = 0
    remaining_params: int = 0
    original_flops: int = 0
    effective_flops: int = 0
    flop_reduction_pct: float = 0.0
    memory_savings_pct: float = 0.0
    estimated_speedup: float = 1.0
    estimated_accuracy_drop: float = 0.0

    def summary(self) -> Dict:
        return {
            "model": self.model_name,
            "type": self.pruning_type,
            "sparsity": f"{self.sparsity:.0%}",
            "params_original": f"{self.original_params:,}",
            "params_remaining": f"{self.remaining_params:,}",
            "flop_reduction": f"{self.flop_reduction_pct:.1f}%",
            "memory_savings": f"{self.memory_savings_pct:.1f}%",
            "speedup": f"{self.estimated_speedup:.2f}x",
            "accuracy_drop": f"{self.estimated_accuracy_drop:.2f}%",
        }


class PruningAnalyzer:
    """
    Analyze the impact of pruning on transformer models.
    """

    def analyze(
        self,
        graph,
        sparsity: float = 0.5,
        pruning_type: str = "unstructured"
    ) -> PruningImpact:
        """
        Analyze pruning impact.

        Args:
            graph: ComputationGraph
            sparsity: Target sparsity (0.5 = 50% zeros)
            pruning_type: "unstructured" or "structured"
        """
        total_flops = graph.get_total_flops()
        total_weights = graph.get_total_weights()

        # Count parameters (assuming fp32)
        total_params = total_weights // 4

        # Effective computation after pruning
        if pruning_type == "structured":
            # Structured pruning removes entire rows/columns
            effective_flops = int(total_flops * (1 - sparsity))
            speedup = 1.0 / (1 - sparsity) if sparsity < 1.0 else 1.0
            memory_savings = sparsity * 100
        else:
            # Unstructured: sparse but hardware may not exploit it well
            effective_flops = int(total_flops * (1 - sparsity))
            # RISC-V without sparse hardware gets limited speedup
            speedup = 1.0 / (1 - sparsity * 0.3)  # Only 30% effective
            memory_savings = sparsity * 60  # Need indices → less savings

        pruned_params = int(total_params * sparsity)
        remaining_params = total_params - pruned_params

        # Accuracy impact (empirical estimates)
        if sparsity <= 0.5:
            accuracy_drop = sparsity * 1.0
        elif sparsity <= 0.8:
            accuracy_drop = 0.5 + (sparsity - 0.5) * 4.0
        else:
            accuracy_drop = 1.7 + (sparsity - 0.8) * 15.0

        return PruningImpact(
            model_name=graph.name,
            pruning_type=pruning_type,
            sparsity=sparsity,
            original_params=total_params,
            pruned_params=pruned_params,
            remaining_params=remaining_params,
            original_flops=total_flops,
            effective_flops=effective_flops,
            flop_reduction_pct=sparsity * 100,
            memory_savings_pct=memory_savings,
            estimated_speedup=speedup,
            estimated_accuracy_drop=accuracy_drop,
        )