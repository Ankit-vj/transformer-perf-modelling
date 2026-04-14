# Save as: C:\Users\ankit\transformer-perf-model\transformer_perf\estimators\roofline.py

"""
Roofline Performance Model.
Analyzes operations against hardware performance ceilings
to identify compute-bound vs memory-bound behavior.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class RooflinePoint:
    """A single point on the roofline plot"""
    name: str
    flops: int
    data_bytes: int
    arithmetic_intensity: float     # FLOP/byte
    achieved_gflops: float
    peak_gflops: float
    bandwidth_gflops: float         # Performance if memory-bound
    bound_type: str                 # "compute" or "memory"
    efficiency: float               # Percentage of peak
    distance_to_roof: float         # Gap to roofline


@dataclass
class RooflineResult:
    """Complete roofline analysis"""
    soc_name: str
    dtype: str
    peak_compute_gflops: float
    peak_bandwidth_gbps: float
    ridge_point: float              # FLOP/byte at intersection
    points: List[RooflinePoint] = field(default_factory=list)
    cache_rooflines: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict:
        compute_bound = sum(1 for p in self.points if p.bound_type == "compute")
        memory_bound = sum(1 for p in self.points if p.bound_type == "memory")

        return {
            "soc_name": self.soc_name,
            "dtype": self.dtype,
            "peak_gflops": f"{self.peak_compute_gflops:.2f}",
            "peak_bandwidth_gbps": f"{self.peak_bandwidth_gbps:.1f}",
            "ridge_point": f"{self.ridge_point:.2f} FLOP/byte",
            "total_operations": len(self.points),
            "compute_bound_ops": compute_bound,
            "memory_bound_ops": memory_bound,
        }

    def print_report(self):
        """Print roofline analysis report"""
        print(f"\n{'='*80}")
        print(f"ROOFLINE ANALYSIS: {self.soc_name}")
        print(f"{'='*80}")
        print(f"  Peak Compute  : {self.peak_compute_gflops:.2f} GFLOP/s")
        print(f"  Peak DRAM BW  : {self.peak_bandwidth_gbps:.1f} GB/s")
        print(f"  Ridge Point   : {self.ridge_point:.2f} FLOP/byte")

        if self.cache_rooflines:
            print(f"\n  Cache Bandwidth Ceilings:")
            for level, bw in self.cache_rooflines.items():
                print(f"    {level}: {bw:.1f} GB/s")

        print(f"\n  {'Operation':<30} {'AI':>8} {'Achv GFLOP/s':>13} "
              f"{'Peak':>8} {'Effic':>8} {'Bound':<8}")
        print(f"  {'─'*30} {'─'*8} {'─'*13} {'─'*8} {'─'*8} {'─'*8}")

        for p in sorted(self.points, key=lambda x: x.arithmetic_intensity):
            print(
                f"  {p.name:<30} "
                f"{p.arithmetic_intensity:>8.2f} "
                f"{p.achieved_gflops:>13.2f} "
                f"{p.peak_gflops:>8.2f} "
                f"{p.efficiency:>7.1f}% "
                f"{p.bound_type:<8}"
            )

        print(f"{'='*80}\n")


class RooflineModel:
    """
    Roofline performance analysis for transformer
    operations on RISC-V SoCs.

    The roofline model defines performance ceilings:
    - Compute ceiling: Peak GFLOP/s
    - Memory ceiling: Peak Bandwidth × Arithmetic Intensity

    Performance = min(Peak_Compute, Peak_BW × AI)
    """

    def __init__(self, soc):
        self.soc = soc
        self.frequency = soc.frequency

    def analyze_model(
        self,
        graph,
        dtype: str = "fp32"
    ) -> RooflineResult:
        """
        Perform roofline analysis on entire model.

        Args:
            graph: ComputationGraph
            dtype: Data type

        Returns:
            RooflineResult with all operation points
        """
        peak = self.soc.get_peak_performance(dtype)
        peak_gflops = peak["system_gflops"]
        peak_bw_gbps = peak["dram_bandwidth_gbps"]
        ridge_point = peak_gflops / peak_bw_gbps  # FLOP/byte

        # Cache bandwidth ceilings
        cache_bw = {}
        for level in ["L1D", "L2", "L3"]:
            bw = self.soc.memory.get_bandwidth_at_level(
                level, self.frequency
            )
            cache_bw[level] = bw

        points = []

        for node_id in graph.get_topological_order():
            node = graph.get_node(node_id)
            layer = node.layer

            if layer.flops == 0 and layer.total_memory_bytes == 0:
                continue

            point = self._analyze_operation(
                layer, peak_gflops, peak_bw_gbps, dtype
            )
            points.append(point)

        return RooflineResult(
            soc_name=self.soc.name,
            dtype=dtype,
            peak_compute_gflops=peak_gflops,
            peak_bandwidth_gbps=peak_bw_gbps,
            ridge_point=ridge_point,
            points=points,
            cache_rooflines=cache_bw,
        )

    def _analyze_operation(
        self,
        layer,
        peak_gflops: float,
        peak_bw_gbps: float,
        dtype: str
    ) -> RooflinePoint:
        """Analyze a single operation against the roofline"""

        flops = layer.flops
        data_bytes = layer.total_memory_bytes

        # Arithmetic intensity
        ai = flops / data_bytes if data_bytes > 0 else 0

        # Ridge point
        ridge = peak_gflops / peak_bw_gbps if peak_bw_gbps > 0 else 0

        # Performance ceiling at this AI
        bandwidth_ceiling = ai * peak_bw_gbps  # GFLOP/s if memory-bound
        roofline_perf = min(peak_gflops, bandwidth_ceiling)

        # Determine bound type
        if ai >= ridge:
            bound_type = "compute"
        else:
            bound_type = "memory"

        # Estimate achieved performance (with efficiency factors)
        efficiency_factor = self._estimate_efficiency(layer, dtype)
        achieved_gflops = roofline_perf * efficiency_factor

        # Efficiency vs peak
        efficiency = (
            achieved_gflops / peak_gflops * 100 if peak_gflops > 0 else 0
        )

        # Distance to roofline
        distance = roofline_perf - achieved_gflops

        return RooflinePoint(
            name=layer.name,
            flops=flops,
            data_bytes=data_bytes,
            arithmetic_intensity=ai,
            achieved_gflops=achieved_gflops,
            peak_gflops=roofline_perf,
            bandwidth_gflops=bandwidth_ceiling,
            bound_type=bound_type,
            efficiency=efficiency,
            distance_to_roof=distance,
        )

    def _estimate_efficiency(self, layer, dtype: str) -> float:
        """
        Estimate efficiency factor based on operation type.
        Accounts for vectorization efficiency, pipeline stalls, etc.
        """
        efficiency_map = {
            "LINEAR": 0.75,          # Good vectorization
            "MATMUL": 0.75,
            "ATTENTION": 0.60,       # Mixed ops (matmul + softmax)
            "MLP": 0.70,             # Good vectorization + activation
            "LAYERNORM": 0.45,       # Reductions reduce efficiency
            "RMSNORM": 0.45,
            "SOFTMAX": 0.35,         # Transcendental + reduction
            "GELU": 0.50,            # Transcendental ops
            "RELU": 0.80,            # Simple comparison
            "SILU": 0.50,
            "EMBEDDING": 0.20,       # Memory-bound lookup
            "RESIDUAL_ADD": 0.80,    # Simple element-wise
            "RESHAPE": 0.95,         # Nearly free
            "TRANSPOSE": 0.60,       # Memory pattern issues
        }

        return efficiency_map.get(layer.layer_type.name, 0.50)

    def get_roofline_coordinates(
        self, dtype: str = "fp32"
    ) -> Dict[str, List]:
        """
        Get coordinates for plotting the roofline.

        Returns x (AI) and y (GFLOP/s) arrays for the roofline line.
        """
        peak = self.soc.get_peak_performance(dtype)
        peak_gflops = peak["system_gflops"]
        peak_bw = peak["dram_bandwidth_gbps"]

        ridge = peak_gflops / peak_bw if peak_bw > 0 else 1.0

        # Generate points for the roofline line
        ai_values = []
        perf_values = []

        # Memory-bound region
        for ai in [0.01, 0.1, 0.5, 1.0, ridge]:
            ai_values.append(ai)
            perf_values.append(ai * peak_bw)

        # Compute-bound region
        for ai in [ridge, ridge * 2, ridge * 5, ridge * 10, 100]:
            ai_values.append(ai)
            perf_values.append(peak_gflops)

        return {
            "ai_values": ai_values,
            "perf_values": perf_values,
            "ridge_point": ridge,
            "peak_gflops": peak_gflops,
            "peak_bandwidth": peak_bw,
        }


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
    graph = parser.parse_from_preset("bert-base", batch_size=1, seq_len=128)

    roofline = RooflineModel(soc)
    result = roofline.analyze_model(graph, dtype="fp32")
    result.print_report()