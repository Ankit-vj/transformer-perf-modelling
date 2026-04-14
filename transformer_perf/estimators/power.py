# Save as: C:\Users\ankit\transformer-perf-model\transformer_perf\estimators\power.py

"""
Power and Energy Estimation.
Estimates power consumption and energy efficiency
for transformer inference on RISC-V SoCs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


@dataclass
class PowerBreakdown:
    """Power breakdown for the SoC"""
    core_dynamic_w: float = 0.0     # Core dynamic power (W)
    core_leakage_w: float = 0.0     # Core leakage power (W)
    vector_unit_w: float = 0.0       # Vector unit power (W)
    cache_power_w: float = 0.0       # Cache power (W)
    dram_power_w: float = 0.0        # DRAM power (W)
    noc_power_w: float = 0.0         # Network-on-chip power (W)
    total_power_w: float = 0.0       # Total power (W)

    def summary(self) -> Dict:
        return {
            "core_dynamic_w": f"{self.core_dynamic_w:.3f}",
            "core_leakage_w": f"{self.core_leakage_w:.3f}",
            "vector_unit_w": f"{self.vector_unit_w:.3f}",
            "cache_power_w": f"{self.cache_power_w:.3f}",
            "dram_power_w": f"{self.dram_power_w:.3f}",
            "noc_power_w": f"{self.noc_power_w:.3f}",
            "total_power_w": f"{self.total_power_w:.3f}",
        }


@dataclass
class EnergyResult:
    """Energy estimation result"""
    model_name: str
    total_energy_mj: float = 0.0        # millijoules
    energy_per_token_uj: float = 0.0     # microjoules per token
    energy_per_flop_pj: float = 0.0      # picojoules per FLOP
    average_power_w: float = 0.0         # Watts
    power_breakdown: Optional[PowerBreakdown] = None
    gflops_per_watt: float = 0.0         # Energy efficiency
    tokens_per_joule: float = 0.0

    def summary(self) -> Dict:
        return {
            "model_name": self.model_name,
            "total_energy_mJ": f"{self.total_energy_mj:.4f}",
            "energy_per_token_uJ": f"{self.energy_per_token_uj:.2f}",
            "energy_per_FLOP_pJ": f"{self.energy_per_flop_pj:.4f}",
            "average_power_W": f"{self.average_power_w:.3f}",
            "GFLOPS_per_Watt": f"{self.gflops_per_watt:.2f}",
            "tokens_per_Joule": f"{self.tokens_per_joule:.0f}",
        }

    def print_report(self):
        """Print energy report"""
        print(f"\n{'='*60}")
        print(f"ENERGY REPORT: {self.model_name}")
        print(f"{'='*60}")
        print(f"  Total Energy       : {self.total_energy_mj:.4f} mJ")
        print(f"  Energy/Token       : {self.energy_per_token_uj:.2f} µJ")
        print(f"  Energy/FLOP        : {self.energy_per_flop_pj:.4f} pJ")
        print(f"  Average Power      : {self.average_power_w:.3f} W")
        print(f"  Efficiency         : {self.gflops_per_watt:.2f} GFLOPS/W")
        print(f"  Tokens/Joule       : {self.tokens_per_joule:.0f}")

        if self.power_breakdown:
            print(f"\n  Power Breakdown:")
            for key, val in self.power_breakdown.summary().items():
                print(f"    {key:<20}: {val} W")

        print(f"{'='*60}\n")


class PowerEstimator:
    """
    Estimate power consumption using activity-based
    analytical power models.

    Uses empirical power coefficients based on published
    RISC-V processor data (SiFive, Xuantie, etc.)
    """

    # Power coefficients (mW per unit activity)
    # Based on typical 28nm/22nm RISC-V implementations
    POWER_COEFFICIENTS = {
        "core_base_mw_per_ghz": 150.0,       # Base core power per GHz
        "fpu_mw_per_gflop": 5.0,             # FPU power per GFLOP/s
        "vector_mw_per_element": 0.05,        # Vector unit per element/cycle
        "l1_mw_per_access": 0.005,            # L1 cache per access
        "l2_mw_per_access": 0.02,             # L2 cache per access
        "l3_mw_per_access": 0.05,             # L3 cache per access
        "dram_mw_per_gb": 300.0,              # DRAM power per GB/s bandwidth
        "leakage_fraction": 0.15,             # Leakage as fraction of dynamic
        "noc_mw_per_gbps": 10.0,              # NoC power per GB/s
    }

    def __init__(self, soc, process_node_nm: int = 28):
        self.soc = soc
        self.process_node = process_node_nm

        # Scale power by process node
        self.process_scale = (process_node_nm / 28.0) ** 1.5

        self.coefficients = self.POWER_COEFFICIENTS.copy()

    def estimate_power_breakdown(
        self,
        utilization: float = 0.5,
        memory_bandwidth_fraction: float = 0.3
    ) -> PowerBreakdown:
        """
        Estimate power breakdown for the SoC.

        Args:
            utilization: Fraction of peak compute utilization
            memory_bandwidth_fraction: Fraction of peak memory bandwidth used
        """
        freq_ghz = self.soc.frequency / 1e9
        num_cores = self.soc.num_cores

        # Core dynamic power
        core_dynamic = (
            self.coefficients["core_base_mw_per_ghz"]
            * freq_ghz * num_cores * utilization
            * self.process_scale / 1000
        )

        # Core leakage
        core_leakage = core_dynamic * self.coefficients["leakage_fraction"]

        # Vector unit power
        peak = self.soc.get_peak_performance("fp32")
        vector_gflops = peak["vector_gflops_per_core"] * num_cores
        vector_power = (
            self.coefficients["fpu_mw_per_gflop"]
            * vector_gflops * utilization
            * self.process_scale / 1000
        )

        # Cache power (estimated from access rates)
        l1_accesses_per_cycle = 2.0 * utilization  # Approximate
        l2_accesses_per_cycle = 0.3 * utilization
        l3_accesses_per_cycle = 0.05 * utilization

        cache_power = (
            (self.coefficients["l1_mw_per_access"] * l1_accesses_per_cycle * freq_ghz * 1e9
             + self.coefficients["l2_mw_per_access"] * l2_accesses_per_cycle * freq_ghz * 1e9
             + self.coefficients["l3_mw_per_access"] * l3_accesses_per_cycle * freq_ghz * 1e9)
            * num_cores * self.process_scale / 1e6
        )

        # DRAM power
        dram_bw = peak["dram_bandwidth_gbps"] * memory_bandwidth_fraction
        dram_power = (
            self.coefficients["dram_mw_per_gb"]
            * dram_bw / 1000
        )

        # NoC power
        noc_power = (
            self.coefficients["noc_mw_per_gbps"]
            * dram_bw * 0.5  # Internal bandwidth
            / 1000
        )

        total = (
            core_dynamic + core_leakage + vector_power
            + cache_power + dram_power + noc_power
        )

        return PowerBreakdown(
            core_dynamic_w=core_dynamic,
            core_leakage_w=core_leakage,
            vector_unit_w=vector_power,
            cache_power_w=cache_power,
            dram_power_w=dram_power,
            noc_power_w=noc_power,
            total_power_w=total,
        )

    def estimate_energy(
        self,
        latency_result,
        seq_len: int = 128
    ) -> EnergyResult:
        """
        Estimate energy consumption for inference.

        Args:
            latency_result: ModelLatency object from LatencyEstimator
            seq_len: Sequence length for per-token energy
        """
        # Get power at the estimated utilization
        power = self.estimate_power_breakdown(
            utilization=latency_result.hardware_utilization,
            memory_bandwidth_fraction=0.3,
        )

        # Energy = Power × Time
        latency_sec = latency_result.total_latency_ms / 1000
        total_energy_j = power.total_power_w * latency_sec
        total_energy_mj = total_energy_j * 1000

        # Per-token energy
        energy_per_token_j = total_energy_j / seq_len if seq_len > 0 else 0
        energy_per_token_uj = energy_per_token_j * 1e6

        # Per-FLOP energy
        energy_per_flop_j = (
            total_energy_j / latency_result.total_flops
            if latency_result.total_flops > 0 else 0
        )
        energy_per_flop_pj = energy_per_flop_j * 1e12

        # Efficiency metrics
        gflops_per_watt = (
            latency_result.achieved_gflops / power.total_power_w
            if power.total_power_w > 0 else 0
        )

        tokens_per_joule = (
            seq_len / total_energy_j if total_energy_j > 0 else 0
        )

        return EnergyResult(
            model_name=latency_result.model_name,
            total_energy_mj=total_energy_mj,
            energy_per_token_uj=energy_per_token_uj,
            energy_per_flop_pj=energy_per_flop_pj,
            average_power_w=power.total_power_w,
            power_breakdown=power,
            gflops_per_watt=gflops_per_watt,
            tokens_per_joule=tokens_per_joule,
        )


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from transformer_perf.hardware.soc import RISCVSoC
    from transformer_perf.models.parser import TransformerGraphParser
    from transformer_perf.estimators.latency import LatencyEstimator

    soc = RISCVSoC(preset="mid-range")
    parser = TransformerGraphParser(dtype="fp32")
    graph = parser.parse_from_preset("bert-base", batch_size=1, seq_len=128)

    lat_est = LatencyEstimator(soc)
    lat_result = lat_est.estimate_model(graph, dtype="fp32", seq_len=128)

    power_est = PowerEstimator(soc, process_node_nm=28)
    energy = power_est.estimate_energy(lat_result, seq_len=128)

    energy.print_report()