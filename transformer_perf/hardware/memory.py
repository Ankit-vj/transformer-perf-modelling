# Save as: C:\Users\ankit\transformer-perf-model\hardware\memory.py

"""
Memory Hierarchy Model.
Models cache levels, main memory, and bandwidth.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import math


@dataclass
class CacheLevel:
    """Configuration for one level of cache"""
    name: str                    # e.g., "L1D", "L1I", "L2", "L3"
    size_kb: int                 # Size in kilobytes
    line_size: int = 64          # Cache line size in bytes
    associativity: int = 4       # Set associativity
    latency_cycles: int = 2      # Access latency in cycles
    bandwidth_bytes_per_cycle: float = 64.0  # bytes/cycle
    is_shared: bool = False      # Shared across cores?
    write_policy: str = "writeback"  # writeback or writethrough

    @property
    def size_bytes(self) -> int:
        return self.size_kb * 1024

    @property
    def num_sets(self) -> int:
        return self.size_bytes // (self.line_size * self.associativity)

    def can_hold(self, data_bytes: int) -> bool:
        """Check if data fits in this cache level"""
        return data_bytes <= self.size_bytes

    def bandwidth_gb_per_sec(self, frequency_hz: float) -> float:
        """Calculate bandwidth in GB/s"""
        return (
            self.bandwidth_bytes_per_cycle * frequency_hz / 1e9
        )


@dataclass
class DRAMConfig:
    """Main memory (DRAM) configuration"""
    type: str = "DDR4"           # DDR4, DDR5, HBM2, LPDDR5
    capacity_gb: int = 8         # Total capacity
    channels: int = 2            # Number of memory channels
    bus_width_bits: int = 64     # Bus width per channel
    frequency_mhz: int = 3200   # Memory clock frequency
    latency_ns: float = 50.0     # Access latency

    @property
    def peak_bandwidth_gbps(self) -> float:
        """Calculate peak bandwidth in GB/s"""
        # DDR = Double Data Rate, so x2
        bytes_per_transfer = self.bus_width_bits // 8
        transfers_per_sec = self.frequency_mhz * 1e6 * 2  # DDR
        bandwidth_per_channel = (
            bytes_per_transfer * transfers_per_sec / 1e9
        )
        return bandwidth_per_channel * self.channels

    def latency_cycles(self, core_frequency_hz: float) -> int:
        """Convert DRAM latency to core cycles"""
        return int(self.latency_ns * core_frequency_hz / 1e9)


class MemoryHierarchy:
    """
    Complete memory subsystem model including
    cache hierarchy and main memory.
    """

    def __init__(self, config: Optional[Dict] = None):
        if config:
            self._load_from_dict(config)
        else:
            self._init_default()

    def _init_default(self):
        """Initialize default memory hierarchy"""
        # L1 Instruction Cache (per core)
        self.l1i = CacheLevel(
            name="L1I",
            size_kb=32,
            line_size=64,
            associativity=4,
            latency_cycles=2,
            bandwidth_bytes_per_cycle=16,
            is_shared=False,
        )

        # L1 Data Cache (per core)
        self.l1d = CacheLevel(
            name="L1D",
            size_kb=32,
            line_size=64,
            associativity=8,
            latency_cycles=2,
            bandwidth_bytes_per_cycle=64,
            is_shared=False,
        )

        # L2 Cache (per core)
        self.l2 = CacheLevel(
            name="L2",
            size_kb=256,
            line_size=64,
            associativity=8,
            latency_cycles=10,
            bandwidth_bytes_per_cycle=32,
            is_shared=False,
        )

        # L3 Cache (shared)
        self.l3 = CacheLevel(
            name="L3",
            size_kb=4096,
            line_size=64,
            associativity=16,
            latency_cycles=40,
            bandwidth_bytes_per_cycle=16,
            is_shared=True,
        )

        # Main Memory (DRAM)
        self.dram = DRAMConfig(
            type="DDR4",
            capacity_gb=8,
            channels=2,
            bus_width_bits=64,
            frequency_mhz=3200,
            latency_ns=50.0,
        )

        # Ordered list of cache levels
        self.cache_levels = [self.l1d, self.l2, self.l3]

    def _load_from_dict(self, config: Dict):
        """Load memory hierarchy from dictionary config"""
        # Load L1D
        l1d_cfg = config.get("l1d", {})
        self.l1d = CacheLevel(
            name="L1D",
            size_kb=l1d_cfg.get("size_kb", 32),
            line_size=l1d_cfg.get("line_size", 64),
            associativity=l1d_cfg.get("associativity", 8),
            latency_cycles=l1d_cfg.get("latency_cycles", 2),
            bandwidth_bytes_per_cycle=l1d_cfg.get("bandwidth", 64),
        )

        # Load L1I
        l1i_cfg = config.get("l1i", {})
        self.l1i = CacheLevel(
            name="L1I",
            size_kb=l1i_cfg.get("size_kb", 32),
            line_size=l1i_cfg.get("line_size", 64),
            associativity=l1i_cfg.get("associativity", 4),
            latency_cycles=l1i_cfg.get("latency_cycles", 2),
            bandwidth_bytes_per_cycle=l1i_cfg.get("bandwidth", 16),
        )

        # Load L2
        l2_cfg = config.get("l2", {})
        self.l2 = CacheLevel(
            name="L2",
            size_kb=l2_cfg.get("size_kb", 256),
            line_size=l2_cfg.get("line_size", 64),
            associativity=l2_cfg.get("associativity", 8),
            latency_cycles=l2_cfg.get("latency_cycles", 10),
            bandwidth_bytes_per_cycle=l2_cfg.get("bandwidth", 32),
        )

        # Load L3
        l3_cfg = config.get("l3", {})
        self.l3 = CacheLevel(
            name="L3",
            size_kb=l3_cfg.get("size_kb", 4096),
            line_size=l3_cfg.get("line_size", 64),
            associativity=l3_cfg.get("associativity", 16),
            latency_cycles=l3_cfg.get("latency_cycles", 40),
            bandwidth_bytes_per_cycle=l3_cfg.get("bandwidth", 16),
            is_shared=True,
        )

        # Load DRAM
        dram_cfg = config.get("dram", {})
        self.dram = DRAMConfig(
            type=dram_cfg.get("type", "DDR4"),
            capacity_gb=dram_cfg.get("capacity_gb", 8),
            channels=dram_cfg.get("channels", 2),
            bus_width_bits=dram_cfg.get("bus_width_bits", 64),
            frequency_mhz=dram_cfg.get("frequency_mhz", 3200),
            latency_ns=dram_cfg.get("latency_ns", 50.0),
        )

        self.cache_levels = [self.l1d, self.l2, self.l3]

    def estimate_data_access_cycles(
        self,
        data_size_bytes: int,
        access_pattern: str = "streaming",
        reuse_factor: float = 1.0,
        core_frequency_hz: float = 2e9
    ) -> Dict:
        """
        Estimate memory access cycles for a data block.

        Args:
            data_size_bytes: Size of data to access
            access_pattern: "streaming", "random", "reuse"
            reuse_factor: How much data is reused (1.0 = no reuse)
            core_frequency_hz: Core frequency for DRAM latency conversion

        Returns:
            Dictionary with cycles and hit/miss breakdown
        """
        effective_data = data_size_bytes / reuse_factor

        # Determine which cache level the data fits in
        serving_level = None
        for cache in self.cache_levels:
            if cache.can_hold(effective_data):
                serving_level = cache
                break

        if serving_level is None:
            # Data doesn't fit in any cache → DRAM
            dram_cycles = self.dram.latency_cycles(core_frequency_hz)
            bandwidth_cycles = data_size_bytes / (
                self.dram.peak_bandwidth_gbps * 1e9
                / core_frequency_hz
            )
            return {
                "serving_level": "DRAM",
                "latency_cycles": dram_cycles,
                "bandwidth_cycles": bandwidth_cycles,
                "total_cycles": max(dram_cycles, bandwidth_cycles),
                "bandwidth_bound": bandwidth_cycles > dram_cycles,
            }

        # Data fits in cache
        bandwidth_cycles = (
            data_size_bytes / serving_level.bandwidth_bytes_per_cycle
        )
        latency_cycles = serving_level.latency_cycles

        return {
            "serving_level": serving_level.name,
            "latency_cycles": latency_cycles,
            "bandwidth_cycles": bandwidth_cycles,
            "total_cycles": max(latency_cycles, bandwidth_cycles),
            "bandwidth_bound": bandwidth_cycles > latency_cycles,
        }

    def get_bandwidth_at_level(
        self,
        level: str,
        frequency_hz: float = 2e9
    ) -> float:
        """
        Get bandwidth in GB/s at a specific cache level.
        """
        level_map = {
            "L1D": self.l1d,
            "L1I": self.l1i,
            "L2": self.l2,
            "L3": self.l3,
        }

        if level in level_map:
            cache = level_map[level]
            return cache.bandwidth_gb_per_sec(frequency_hz)
        elif level == "DRAM":
            return self.dram.peak_bandwidth_gbps
        else:
            return 0.0

    def summary(self, frequency_hz: float = 2e9) -> Dict:
        """Return memory hierarchy summary"""
        return {
            "L1I": {
                "size_kb": self.l1i.size_kb,
                "latency": self.l1i.latency_cycles,
                "bandwidth_gbps": self.l1i.bandwidth_gb_per_sec(
                    frequency_hz
                ),
            },
            "L1D": {
                "size_kb": self.l1d.size_kb,
                "latency": self.l1d.latency_cycles,
                "bandwidth_gbps": self.l1d.bandwidth_gb_per_sec(
                    frequency_hz
                ),
            },
            "L2": {
                "size_kb": self.l2.size_kb,
                "latency": self.l2.latency_cycles,
                "bandwidth_gbps": self.l2.bandwidth_gb_per_sec(
                    frequency_hz
                ),
            },
            "L3": {
                "size_kb": self.l3.size_kb,
                "latency": self.l3.latency_cycles,
                "bandwidth_gbps": self.l3.bandwidth_gb_per_sec(
                    frequency_hz
                ),
            },
            "DRAM": {
                "type": self.dram.type,
                "capacity_gb": self.dram.capacity_gb,
                "peak_bandwidth_gbps": self.dram.peak_bandwidth_gbps,
                "latency_ns": self.dram.latency_ns,
            },
        }


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    mem = MemoryHierarchy()
    freq = 2e9

    print("Memory Hierarchy Configuration:")
    for level, info in mem.summary(freq).items():
        print(f"\n  {level}:")
        for key, val in info.items():
            print(f"    {key}: {val}")

    print(f"\n  Access estimate for 1MB data:")
    result = mem.estimate_data_access_cycles(
        1024 * 1024, core_frequency_hz=freq
    )
    for key, val in result.items():
        print(f"    {key}: {val}")