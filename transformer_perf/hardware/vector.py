# Save as: C:\Users\ankit\transformer-perf-model\hardware\vector.py

"""
RISC-V Vector Extension (RVV) Model.
Models vector processing capabilities for
AI/ML workload acceleration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class VectorUnitConfig:
    """Configuration for a vector functional unit"""
    name: str
    latency: int          # Cycles per vector operation
    throughput: float     # Vector ops per cycle
    supported_sew: List[int] = field(
        default_factory=lambda: [8, 16, 32, 64]
    )


class RVVExtension:
    """
    Models the RISC-V Vector Extension (RVV 1.0).

    Key parameters:
    - VLEN: Vector register length in bits
    - ELEN: Maximum element width in bits
    - SEW: Selected Element Width
    - LMUL: Vector Length Multiplier
    - VL: Vector Length (number of elements processed)
    """

    def __init__(
        self,
        vlen: int = 256,
        elen: int = 64,
        num_vector_regs: int = 32,
        config: Optional[Dict] = None
    ):
        self.vlen = vlen              # bits
        self.elen = elen              # bits
        self.num_vector_regs = num_vector_regs
        self.enabled = True

        # Vector functional units
        self.vector_units = {
            "VADD": VectorUnitConfig(
                name="VADD", latency=1, throughput=1.0
            ),
            "VMUL": VectorUnitConfig(
                name="VMUL", latency=2, throughput=1.0
            ),
            "VFMA": VectorUnitConfig(
                name="VFMA", latency=3, throughput=1.0
            ),
            "VDIV": VectorUnitConfig(
                name="VDIV", latency=12, throughput=0.1
            ),
            "VLOAD": VectorUnitConfig(
                name="VLOAD", latency=3, throughput=1.0
            ),
            "VSTORE": VectorUnitConfig(
                name="VSTORE", latency=1, throughput=1.0
            ),
            "VREDUCE": VectorUnitConfig(
                name="VREDUCE", latency=4, throughput=0.5
            ),
        }

        if config:
            self._load_config(config)

    def _load_config(self, config: Dict):
        """Load configuration from dictionary"""
        self.vlen = config.get("vlen", self.vlen)
        self.elen = config.get("elen", self.elen)
        self.num_vector_regs = config.get(
            "num_vector_regs", self.num_vector_regs
        )
        self.enabled = config.get("enabled", True)

    def get_vlmax(self, sew: int, lmul: float = 1.0) -> int:
        """
        Calculate maximum vector length.
        VLMAX = (VLEN / SEW) * LMUL

        Args:
            sew: Selected Element Width in bits (8, 16, 32, 64)
            lmul: Vector Length Multiplier (0.125 to 8)

        Returns:
            Maximum number of elements per vector operation
        """
        return int((self.vlen / sew) * lmul)

    def elements_per_vreg(self, dtype: str = "fp32") -> int:
        """
        Calculate number of elements per vector register.

        Args:
            dtype: Data type string

        Returns:
            Number of elements that fit in one vector register
        """
        sew_bits = {
            "fp64": 64, "fp32": 32, "fp16": 16, "bf16": 16,
            "int64": 64, "int32": 32, "int16": 16, "int8": 8,
            "int4": 4,
        }
        sew = sew_bits.get(dtype, 32)
        return self.vlen // sew

    def vector_throughput_ops_per_cycle(
        self,
        operation: str,
        dtype: str = "fp32",
        lmul: float = 1.0
    ) -> float:
        """
        Calculate vector operation throughput in operations per cycle.

        Args:
            operation: Operation type (add, mul, fma, etc.)
            dtype: Data type
            lmul: Vector length multiplier

        Returns:
            Operations per cycle
        """
        if not self.enabled:
            return 0.0

        # Map operation to vector unit
        op_to_unit = {
            "add": "VADD", "sub": "VADD",
            "mul": "VMUL",
            "fma": "VFMA", "fmadd": "VFMA", "mac": "VFMA",
            "div": "VDIV",
            "load": "VLOAD",
            "store": "VSTORE",
            "reduce": "VREDUCE",
        }

        unit_name = op_to_unit.get(operation, "VADD")
        unit = self.vector_units.get(unit_name)

        if unit is None:
            return 0.0

        # Elements per operation
        elements = self.get_vlmax(
            sew=self._dtype_to_sew(dtype),
            lmul=lmul
        )

        # Throughput = elements * unit_throughput / latency
        if unit.latency > 0:
            return elements * unit.throughput
        return 0.0

    def vector_peak_flops(
        self,
        frequency: float,
        dtype: str = "fp32",
        lmul: float = 1.0
    ) -> float:
        """
        Calculate peak vector FLOP/s.

        For FMA: each operation = 2 FLOPs
        """
        fma_throughput = self.vector_throughput_ops_per_cycle(
            "fma", dtype, lmul
        )
        # FMA = 2 FLOPs per element
        return fma_throughput * 2 * frequency

    def estimate_vector_op_cycles(
        self,
        total_elements: int,
        operation: str,
        dtype: str = "fp32",
        lmul: float = 1.0
    ) -> float:
        """
        Estimate cycles needed for a vector operation
        on a given number of elements.

        Args:
            total_elements: Total number of elements to process
            operation: Operation type
            dtype: Data type
            lmul: Vector length multiplier

        Returns:
            Estimated cycles
        """
        if not self.enabled:
            return float("inf")

        vlmax = self.get_vlmax(self._dtype_to_sew(dtype), lmul)
        if vlmax == 0:
            return float("inf")

        # Number of vector iterations needed
        num_iterations = (total_elements + vlmax - 1) // vlmax

        # Map to vector unit
        op_to_unit = {
            "add": "VADD", "sub": "VADD",
            "mul": "VMUL",
            "fma": "VFMA",
            "div": "VDIV",
            "load": "VLOAD",
            "store": "VSTORE",
            "reduce": "VREDUCE",
        }

        unit_name = op_to_unit.get(operation, "VADD")
        unit = self.vector_units.get(unit_name)

        if unit is None:
            return float("inf")

        # Total cycles = num_iterations * latency (if pipelined, just startup)
        # Simplified: assume pipelined execution
        startup_latency = unit.latency
        throughput_cycles = num_iterations / unit.throughput

        return startup_latency + throughput_cycles

    @staticmethod
    def _dtype_to_sew(dtype: str) -> int:
        """Convert dtype string to SEW in bits"""
        sew_map = {
            "fp64": 64, "fp32": 32, "fp16": 16, "bf16": 16,
            "int64": 64, "int32": 32, "int16": 16, "int8": 8,
            "int4": 4,
        }
        return sew_map.get(dtype, 32)

    def register_file_size_bytes(self) -> int:
        """Total vector register file size in bytes"""
        return (self.num_vector_regs * self.vlen) // 8

    def summary(self) -> Dict:
        """Return RVV summary"""
        return {
            "enabled": self.enabled,
            "vlen": self.vlen,
            "elen": self.elen,
            "num_vector_regs": self.num_vector_regs,
            "register_file_bytes": self.register_file_size_bytes(),
            "elements_per_reg": {
                "fp32": self.elements_per_vreg("fp32"),
                "fp16": self.elements_per_vreg("fp16"),
                "int8": self.elements_per_vreg("int8"),
            },
        }


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    rvv = RVVExtension(vlen=512)

    print("RISC-V Vector Extension Configuration:")
    print(f"  VLEN: {rvv.vlen} bits")
    print(f"  ELEN: {rvv.elen} bits")
    print(f"  Register File: {rvv.register_file_size_bytes()} bytes")

    print(f"\n  Elements per register:")
    for dtype in ["fp32", "fp16", "int8"]:
        print(f"    {dtype}: {rvv.elements_per_vreg(dtype)}")

    print(f"\n  Throughput (ops/cycle) for FMA:")
    for dtype in ["fp32", "fp16", "int8"]:
        tp = rvv.vector_throughput_ops_per_cycle("fma", dtype)
        print(f"    {dtype}: {tp:.1f} ops/cycle")

    print(f"\n  Peak FLOP/s @ 2GHz:")
    for dtype in ["fp32", "fp16", "int8"]:
        flops = rvv.vector_peak_flops(2e9, dtype)
        print(f"    {dtype}: {flops/1e9:.2f} GFLOP/s")