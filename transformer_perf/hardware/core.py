# Save as: C:\Users\ankit\transformer-perf-model\hardware\core.py

"""
RISC-V Core Microarchitecture Model.
Models the pipeline, execution units, and instruction throughput.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import yaml


@dataclass
class PipelineConfig:
    """CPU Pipeline Configuration"""
    num_stages: int = 5          # Pipeline depth
    issue_width: int = 2         # Superscalar width
    retire_width: int = 2        # Retirement width
    rob_size: int = 64           # Reorder buffer entries
    branch_predictor: str = "gshare"
    branch_penalty: int = 5      # Misprediction penalty cycles


@dataclass
class ExecutionUnit:
    """Functional Unit Specification"""
    name: str                    # Unit name
    count: int = 1               # Number of identical units
    latency: int = 1             # Execution latency in cycles
    throughput: float = 1.0      # Operations per cycle per unit
    pipelined: bool = True       # Is the unit pipelined?

    @property
    def total_throughput(self) -> float:
        """Total throughput across all units of this type"""
        return self.count * self.throughput


class RISCVCore:
    """
    Models a single RISC-V processor core including:
    - Pipeline configuration
    - Scalar execution units
    - Instruction timing
    """

    def __init__(self, config: Optional[Dict] = None):
        if config:
            self._load_from_dict(config)
        else:
            self._init_default()

    def _init_default(self):
        """Initialize with default RISC-V configuration"""
        self.name = "rv64gc-default"
        self.frequency_ghz = 2.0  # GHz
        self.frequency = self.frequency_ghz * 1e9  # Hz

        # Pipeline
        self.pipeline = PipelineConfig(
            num_stages=5,
            issue_width=2,
            retire_width=2,
            rob_size=64,
        )

        # Scalar Execution Units
        self.execution_units = {
            "INT_ALU": ExecutionUnit(
                name="INT_ALU", count=2,
                latency=1, throughput=1.0
            ),
            "INT_MUL": ExecutionUnit(
                name="INT_MUL", count=1,
                latency=3, throughput=1.0
            ),
            "INT_DIV": ExecutionUnit(
                name="INT_DIV", count=1,
                latency=20, throughput=0.05,
                pipelined=False
            ),
            "FP_ADD": ExecutionUnit(
                name="FP_ADD", count=2,
                latency=4, throughput=1.0
            ),
            "FP_MUL": ExecutionUnit(
                name="FP_MUL", count=2,
                latency=4, throughput=1.0
            ),
            "FP_FMA": ExecutionUnit(
                name="FP_FMA", count=2,
                latency=5, throughput=1.0
            ),
            "FP_DIV": ExecutionUnit(
                name="FP_DIV", count=1,
                latency=20, throughput=0.05,
                pipelined=False
            ),
            "LOAD": ExecutionUnit(
                name="LOAD", count=2,
                latency=3, throughput=1.0
            ),
            "STORE": ExecutionUnit(
                name="STORE", count=1,
                latency=1, throughput=1.0
            ),
        }

    def _load_from_dict(self, config: Dict):
        """Load core configuration from dictionary"""
        self.name = config.get("name", "custom-core")
        self.frequency_ghz = config.get("frequency_ghz", 2.0)
        self.frequency = self.frequency_ghz * 1e9

        # Pipeline
        pipe_cfg = config.get("pipeline", {})
        self.pipeline = PipelineConfig(
            num_stages=pipe_cfg.get("num_stages", 5),
            issue_width=pipe_cfg.get("issue_width", 2),
            retire_width=pipe_cfg.get("retire_width", 2),
            rob_size=pipe_cfg.get("rob_size", 64),
        )

        # Execution units
        self.execution_units = {}
        for unit_cfg in config.get("execution_units", []):
            name = unit_cfg["name"]
            self.execution_units[name] = ExecutionUnit(
                name=name,
                count=unit_cfg.get("count", 1),
                latency=unit_cfg.get("latency", 1),
                throughput=unit_cfg.get("throughput", 1.0),
                pipelined=unit_cfg.get("pipelined", True),
            )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RISCVCore":
        """Load core configuration from YAML file"""
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config=config.get("core", config))

    def get_scalar_peak_flops(self) -> float:
        """
        Calculate peak scalar floating-point operations per second.
        FMA counts as 2 FLOPs (multiply + add).
        """
        fma_unit = self.execution_units.get("FP_FMA")
        if fma_unit:
            # FMA: 2 FLOPs per operation
            return fma_unit.total_throughput * 2 * self.frequency
        else:
            # Separate add and multiply units
            fp_add = self.execution_units.get("FP_ADD")
            fp_mul = self.execution_units.get("FP_MUL")
            add_flops = fp_add.total_throughput * self.frequency if fp_add else 0
            mul_flops = fp_mul.total_throughput * self.frequency if fp_mul else 0
            return add_flops + mul_flops

    def get_instruction_latency(self, op_type: str) -> int:
        """Get latency in cycles for an instruction type"""
        unit_mapping = {
            "add": "INT_ALU",
            "sub": "INT_ALU",
            "mul": "INT_MUL",
            "div": "INT_DIV",
            "fadd": "FP_ADD",
            "fmul": "FP_MUL",
            "fmadd": "FP_FMA",
            "fdiv": "FP_DIV",
            "load": "LOAD",
            "store": "STORE",
        }

        unit_name = unit_mapping.get(op_type, "INT_ALU")
        unit = self.execution_units.get(unit_name)
        return unit.latency if unit else 1

    def get_instruction_throughput(self, op_type: str) -> float:
        """Get throughput (ops/cycle) for an instruction type"""
        unit_mapping = {
            "add": "INT_ALU",
            "fadd": "FP_ADD",
            "fmul": "FP_MUL",
            "fmadd": "FP_FMA",
            "fdiv": "FP_DIV",
            "load": "LOAD",
            "store": "STORE",
        }

        unit_name = unit_mapping.get(op_type, "INT_ALU")
        unit = self.execution_units.get(unit_name)
        return unit.total_throughput if unit else 1.0

    def summary(self) -> Dict:
        """Return core summary"""
        return {
            "name": self.name,
            "frequency_ghz": self.frequency_ghz,
            "pipeline_stages": self.pipeline.num_stages,
            "issue_width": self.pipeline.issue_width,
            "scalar_peak_gflops": self.get_scalar_peak_flops() / 1e9,
            "execution_units": {
                name: {
                    "count": eu.count,
                    "latency": eu.latency,
                    "throughput": eu.total_throughput,
                }
                for name, eu in self.execution_units.items()
            },
        }


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    core = RISCVCore()
    print("RISC-V Core Configuration:")
    print(f"  Name: {core.name}")
    print(f"  Frequency: {core.frequency_ghz} GHz")
    print(f"  Pipeline Stages: {core.pipeline.num_stages}")
    print(f"  Issue Width: {core.pipeline.issue_width}")
    print(f"  Scalar Peak: {core.get_scalar_peak_flops()/1e9:.2f} GFLOPS")
    print(f"\n  Execution Units:")
    for name, eu in core.execution_units.items():
        print(
            f"    {name:<10} count={eu.count} "
            f"latency={eu.latency} throughput={eu.total_throughput}"
        )