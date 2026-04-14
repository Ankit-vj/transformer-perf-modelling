# Save as: C:\Users\ankit\transformer-perf-model\hardware\soc.py

"""
Full RISC-V System-on-Chip (SoC) model.
Combines cores, memory hierarchy, and vector extensions.
"""

from typing import Dict, Optional
import yaml

from .core import RISCVCore
from .memory import MemoryHierarchy
from .vector import RVVExtension


class RISCVSoC:
    """
    Complete RISC-V SoC model combining:
    - Multiple RISC-V cores
    - Memory hierarchy (caches + DRAM)
    - Vector extension (RVV)
    - Interconnect
    """

    # Pre-defined SoC configurations
    PRESETS = {
        "minimal": {
            "name": "RISC-V Minimal",
            "num_cores": 1,
            "frequency_ghz": 1.0,
            "vlen": 128,
            "l1d_kb": 16,
            "l2_kb": 128,
            "l3_kb": 0,
            "dram_type": "DDR4",
            "dram_gb": 2,
            "dram_channels": 1,
        },
        "mid-range": {
            "name": "RISC-V Mid-Range",
            "num_cores": 4,
            "frequency_ghz": 2.0,
            "vlen": 256,
            "l1d_kb": 32,
            "l2_kb": 256,
            "l3_kb": 4096,
            "dram_type": "DDR4",
            "dram_gb": 8,
            "dram_channels": 2,
        },
        "high-perf": {
            "name": "RISC-V High Performance",
            "num_cores": 8,
            "frequency_ghz": 3.0,
            "vlen": 512,
            "l1d_kb": 64,
            "l2_kb": 512,
            "l3_kb": 16384,
            "dram_type": "DDR5",
            "dram_gb": 32,
            "dram_channels": 4,
        },
    }

    def __init__(
        self,
        preset: Optional[str] = None,
        num_cores: int = 4,
        config: Optional[Dict] = None
    ):
        if preset and preset in self.PRESETS:
            self._init_from_preset(preset)
        elif config:
            self._init_from_config(config)
        else:
            self._init_default(num_cores)

    def _init_default(self, num_cores: int = 4):
        """Initialize with default configuration"""
        self.name = "RISC-V SoC (Default)"
        self.num_cores = num_cores

        # Create cores
        self.cores = [RISCVCore() for _ in range(num_cores)]
        self.frequency = self.cores[0].frequency

        # Memory hierarchy
        self.memory = MemoryHierarchy()

        # Vector extension (same for all cores)
        self.vector = RVVExtension(vlen=256)

        # Interconnect
        self.noc_bandwidth_bytes_per_cycle = 128
        self.noc_latency_cycles = 5

    def _init_from_preset(self, preset: str):
        """Initialize from preset configuration"""
        cfg = self.PRESETS[preset]

        self.name = cfg["name"]
        self.num_cores = cfg["num_cores"]

        # Cores
        core_config = {
            "name": f"rv64gcv-{preset}",
            "frequency_ghz": cfg["frequency_ghz"],
        }
        self.cores = [
            RISCVCore(config=core_config)
            for _ in range(self.num_cores)
        ]
        self.frequency = cfg["frequency_ghz"] * 1e9

        # Memory
        mem_config = {
            "l1d": {"size_kb": cfg["l1d_kb"]},
            "l2": {"size_kb": cfg["l2_kb"]},
            "l3": {"size_kb": cfg.get("l3_kb", 4096)},
            "dram": {
                "type": cfg["dram_type"],
                "capacity_gb": cfg["dram_gb"],
                "channels": cfg["dram_channels"],
            },
        }
        self.memory = MemoryHierarchy(config=mem_config)

        # Vector
        self.vector = RVVExtension(vlen=cfg["vlen"])

        # Interconnect
        self.noc_bandwidth_bytes_per_cycle = 128
        self.noc_latency_cycles = 5

    def _init_from_config(self, config: Dict):
        """Initialize from full configuration dictionary"""
        self.name = config.get("name", "Custom RISC-V SoC")
        self.num_cores = config.get("num_cores", 4)

        core_cfg = config.get("core", {})
        self.cores = [
            RISCVCore(config=core_cfg)
            for _ in range(self.num_cores)
        ]
        self.frequency = core_cfg.get("frequency_ghz", 2.0) * 1e9

        self.memory = MemoryHierarchy(
            config=config.get("memory", {})
        )
        self.vector = RVVExtension(
            config=config.get("vector", {})
        )

        noc = config.get("interconnect", {})
        self.noc_bandwidth_bytes_per_cycle = noc.get("bandwidth", 128)
        self.noc_latency_cycles = noc.get("latency", 5)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RISCVSoC":
        """Load SoC configuration from YAML file"""
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config=config)

    def get_peak_performance(self, dtype: str = "fp32") -> Dict:
        """
        Calculate peak theoretical performance.

        Returns:
            Dictionary with peak metrics
        """
        core = self.cores[0]

        # Scalar peak
        scalar_flops = core.get_scalar_peak_flops()

        # Vector peak
        vector_flops = self.vector.vector_peak_flops(
            self.frequency, dtype
        )

        # Per-core total
        per_core_flops = scalar_flops + vector_flops

        # System total
        system_flops = per_core_flops * self.num_cores

        # Memory bandwidth
        dram_bandwidth = self.memory.dram.peak_bandwidth_gbps

        # Compute intensity at ridge point
        ridge_point = system_flops / (dram_bandwidth * 1e9)

        return {
            "scalar_gflops_per_core": scalar_flops / 1e9,
            "vector_gflops_per_core": vector_flops / 1e9,
            "total_gflops_per_core": per_core_flops / 1e9,
            "system_gflops": system_flops / 1e9,
            "dram_bandwidth_gbps": dram_bandwidth,
            "ridge_point_flops_per_byte": ridge_point,
            "num_cores": self.num_cores,
            "frequency_ghz": self.frequency / 1e9,
            "vlen": self.vector.vlen,
        }

    def summary(self) -> Dict:
        """Return complete SoC summary"""
        peak = self.get_peak_performance("fp32")

        return {
            "name": self.name,
            "num_cores": self.num_cores,
            "frequency_ghz": self.frequency / 1e9,
            "core": self.cores[0].summary(),
            "vector": self.vector.summary(),
            "memory": self.memory.summary(self.frequency),
            "peak_performance": peak,
        }

    def print_summary(self):
        """Print formatted SoC summary"""
        peak = self.get_peak_performance()

        print(f"\n{'='*60}")
        print(f"SoC: {self.name}")
        print(f"{'='*60}")
        print(f"Cores          : {self.num_cores}")
        print(f"Frequency      : {self.frequency/1e9:.2f} GHz")
        print(f"VLEN           : {self.vector.vlen} bits")
        print(f"{'─'*60}")
        print(f"Peak Performance ({self.num_cores} cores):")
        print(f"  Scalar       : {peak['scalar_gflops_per_core']:.2f} GFLOP/s per core")
        print(f"  Vector       : {peak['vector_gflops_per_core']:.2f} GFLOP/s per core")
        print(f"  Total/Core   : {peak['total_gflops_per_core']:.2f} GFLOP/s")
        print(f"  System       : {peak['system_gflops']:.2f} GFLOP/s")
        print(f"{'─'*60}")
        print(f"Memory:")
        print(f"  L1D          : {self.memory.l1d.size_kb} KB per core")
        print(f"  L2           : {self.memory.l2.size_kb} KB per core")
        print(f"  L3           : {self.memory.l3.size_kb} KB shared")
        print(f"  DRAM         : {self.memory.dram.capacity_gb} GB {self.memory.dram.type}")
        print(f"  DRAM BW      : {peak['dram_bandwidth_gbps']:.1f} GB/s")
        print(f"{'─'*60}")
        print(f"Ridge Point    : {peak['ridge_point_flops_per_byte']:.2f} FLOP/byte")
        print(f"{'='*60}")


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("\n--- Default SoC ---")
    soc = RISCVSoC()
    soc.print_summary()

    print("\n--- Mid-Range Preset ---")
    soc_mid = RISCVSoC(preset="mid-range")
    soc_mid.print_summary()

    print("\n--- High-Performance Preset ---")
    soc_high = RISCVSoC(preset="high-perf")
    soc_high.print_summary()