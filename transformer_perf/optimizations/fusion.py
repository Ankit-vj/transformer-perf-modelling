# Save as: C:\Users\ankit\transformer-perf-model\transformer_perf\optimizations\fusion.py

"""
Operator Fusion Analysis.
Identifies fusible operations and estimates
performance improvement from kernel fusion.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class FusionOpportunity:
    """A potential operator fusion"""
    name: str
    layers: List[str]
    fusion_type: str        # "elementwise", "matmul_bias", "attention"
    memory_savings_bytes: int = 0
    cycle_savings: int = 0
    speedup: float = 1.0


@dataclass
class FusionResult:
    """Complete fusion analysis"""
    model_name: str
    opportunities: List[FusionOpportunity] = field(default_factory=list)
    total_memory_savings_mb: float = 0.0
    total_cycle_savings: int = 0
    overall_speedup: float = 1.0
    fused_layer_count: int = 0
    original_layer_count: int = 0

    def summary(self) -> Dict:
        return {
            "model": self.model_name,
            "fusion_opportunities": len(self.opportunities),
            "original_layers": self.original_layer_count,
            "fused_layers": self.fused_layer_count,
            "memory_savings_mb": f"{self.total_memory_savings_mb:.2f}",
            "overall_speedup": f"{self.overall_speedup:.2f}x",
        }

    def print_report(self):
        print(f"\n{'='*60}")
        print(f"OPERATOR FUSION ANALYSIS: {self.model_name}")
        print(f"{'='*60}")
        for key, val in self.summary().items():
            print(f"  {key:<25}: {val}")

        if self.opportunities:
            print(f"\n  Fusion Opportunities:")
            for opp in self.opportunities[:10]:
                print(f"    {opp.name}: {opp.fusion_type} "
                      f"(speedup: {opp.speedup:.2f}x)")
        print(f"{'='*60}\n")


class FusionAnalyzer:
    """
    Identify and analyze operator fusion opportunities
    in transformer models.

    Common fusions:
    1. MatMul + Bias Add
    2. MatMul + Activation (GELU, ReLU)
    3. LayerNorm fusion (mean + var + normalize + scale)
    4. Attention fusion (QKV + scaled dot product)
    5. Residual Add + LayerNorm
    """

    # Fusion rules and their speedup estimates
    FUSION_RULES = {
        "matmul_bias": {
            "pattern": ["LINEAR", "RESIDUAL_ADD"],
            "speedup": 1.05,
            "memory_reduction": 0.3,  # Save intermediate
        },
        "matmul_activation": {
            "pattern": ["LINEAR", "GELU"],
            "speedup": 1.10,
            "memory_reduction": 0.4,
        },
        "residual_layernorm": {
            "pattern": ["RESIDUAL_ADD", "LAYERNORM"],
            "speedup": 1.15,
            "memory_reduction": 0.5,
        },
        "qkv_fusion": {
            "pattern": ["LINEAR", "LINEAR", "LINEAR"],
            "speedup": 1.20,
            "memory_reduction": 0.3,
        },
        "attention_block": {
            "pattern": ["ATTENTION"],
            "speedup": 1.25,  # Flash attention style
            "memory_reduction": 0.6,
        },
    }

    def analyze(self, graph) -> FusionResult:
        """
        Analyze fusion opportunities in the computation graph.
        """
        execution_order = graph.get_execution_schedule()
        opportunities = []
        total_mem_savings = 0
        total_cycle_savings = 0
        fused_count = 0

        # Scan for fusible patterns
        i = 0
        while i < len(execution_order):
            layer = execution_order[i]
            lt = layer.layer_type.name

            # Check for residual + layernorm fusion
            if (lt == "RESIDUAL_ADD" and
                    i + 1 < len(execution_order) and
                    execution_order[i + 1].layer_type.name == "LAYERNORM"):
                next_layer = execution_order[i + 1]
                mem_save = layer.output_bytes
                opp = FusionOpportunity(
                    name=f"fuse_{layer.name}+{next_layer.name}",
                    layers=[layer.name, next_layer.name],
                    fusion_type="residual_layernorm",
                    memory_savings_bytes=mem_save,
                    speedup=1.15,
                )
                opportunities.append(opp)
                total_mem_savings += mem_save
                fused_count += 1
                i += 2
                continue

            # Check for attention block fusion (flash attention)
            if lt == "ATTENTION":
                mem_save = int(layer.activation_bytes * 0.6)
                opp = FusionOpportunity(
                    name=f"flash_{layer.name}",
                    layers=[layer.name],
                    fusion_type="flash_attention",
                    memory_savings_bytes=mem_save,
                    speedup=1.25,
                )
                opportunities.append(opp)
                total_mem_savings += mem_save
                fused_count += 1

            # Check for MLP fusion (linear + activation + linear)
            if lt == "MLP":
                mem_save = int(layer.activation_bytes * 0.4)
                opp = FusionOpportunity(
                    name=f"fuse_{layer.name}",
                    layers=[layer.name],
                    fusion_type="mlp_fusion",
                    memory_savings_bytes=mem_save,
                    speedup=1.10,
                )
                opportunities.append(opp)
                total_mem_savings += mem_save
                fused_count += 1

            i += 1

        # Calculate overall speedup (multiplicative, capped)
        overall_speedup = 1.0
        for opp in opportunities:
            # Diminishing returns for multiple fusions
            marginal_speedup = 1.0 + (opp.speedup - 1.0) * 0.7
            overall_speedup *= marginal_speedup

        overall_speedup = min(overall_speedup, 2.0)  # Cap at 2x

        return FusionResult(
            model_name=graph.name,
            opportunities=opportunities,
            total_memory_savings_mb=total_mem_savings / (1024 * 1024),
            total_cycle_savings=total_cycle_savings,
            overall_speedup=overall_speedup,
            fused_layer_count=fused_count,
            original_layer_count=len(execution_order),
        )