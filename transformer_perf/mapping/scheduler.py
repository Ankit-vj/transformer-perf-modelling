# Save as: C:\Users\ankit\transformer-perf-model\transformer_perf\mapping\scheduler.py

"""
Instruction Scheduling for RISC-V Execution.
Maps high-level operations to execution unit schedules
and estimates pipeline utilization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class ScheduleEntry:
    """A single entry in the execution schedule"""
    operation: str
    execution_unit: str
    start_cycle: int
    end_cycle: int
    elements: int = 0
    is_vector: bool = False

    @property
    def duration(self) -> int:
        return self.end_cycle - self.start_cycle


@dataclass
class ScheduleResult:
    """Complete scheduling result for a layer"""
    layer_name: str
    entries: List[ScheduleEntry] = field(default_factory=list)
    total_cycles: int = 0
    compute_cycles: int = 0
    memory_cycles: int = 0
    pipeline_utilization: float = 0.0
    vector_utilization: float = 0.0
    unit_utilization: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict:
        return {
            "layer_name": self.layer_name,
            "total_cycles": self.total_cycles,
            "compute_cycles": self.compute_cycles,
            "memory_cycles": self.memory_cycles,
            "pipeline_utilization": f"{self.pipeline_utilization:.1%}",
            "vector_utilization": f"{self.vector_utilization:.1%}",
        }


class InstructionScheduler:
    """
    Schedule operations across RISC-V execution units.
    Estimates how transformer operations map to the
    processor pipeline.
    """

    def __init__(self, core=None, vector_ext=None):
        """
        Args:
            core: RISCVCore object
            vector_ext: RVVExtension object
        """
        self.core = core
        self.vector = vector_ext

        # Default values if no hardware provided
        self.issue_width = 2
        self.frequency = 2e9

        # Default vector config
        self.vlen = 256
        self.vector_enabled = True

        if core:
            self.issue_width = core.pipeline.issue_width
            self.frequency = core.frequency

        if vector_ext:
            self.vlen = vector_ext.vlen
            self.vector_enabled = vector_ext.enabled

    def schedule_matmul(
        self,
        name: str,
        M: int,
        N: int,
        K: int,
        dtype: str = "fp32"
    ) -> ScheduleResult:
        """
        Schedule a matrix multiplication C[M,N] = A[M,K] × B[K,N].

        Decomposes into:
        1. Vector load operations for A and B tiles
        2. Vector FMA operations for computation
        3. Vector store operations for C tiles
        """
        sew_bits = {"fp32": 32, "fp16": 16, "int8": 8}.get(dtype, 32)
        bpe = sew_bits // 8
        elements_per_vreg = self.vlen // sew_bits

        entries = []
        current_cycle = 0

        # Total MAC operations
        total_macs = M * N * K

        # Vector MAC iterations
        # Inner loop processes elements_per_vreg output elements at a time
        vec_iterations = math.ceil(N / elements_per_vreg) * M * math.ceil(K / elements_per_vreg)

        # Compute phase: Vector FMA operations
        fma_latency = 3  # cycles for VFMA
        fma_throughput = 1.0  # ops per cycle (pipelined)

        if self.vector and self.vector_enabled:
            fma_unit = self.vector.vector_units.get("VFMA")
            if fma_unit:
                fma_latency = fma_unit.latency
                fma_throughput = fma_unit.throughput

        # Number of vector FMA instructions needed
        # Each VFMA processes elements_per_vreg MACs
        num_fma_instructions = math.ceil(total_macs / elements_per_vreg)

        # Compute cycles (pipelined: startup + throughput)
        compute_cycles = fma_latency + int(num_fma_instructions / fma_throughput)

        entries.append(ScheduleEntry(
            operation="VFMA",
            execution_unit="VFMA",
            start_cycle=current_cycle,
            end_cycle=current_cycle + compute_cycles,
            elements=total_macs,
            is_vector=True,
        ))

        # Memory phase: loads and stores
        load_data = (M * K + K * N) * bpe  # A + B matrices
        store_data = M * N * bpe            # C matrix

        load_latency = 3  # cycles per vector load
        if self.vector and self.vector_enabled:
            vload = self.vector.vector_units.get("VLOAD")
            if vload:
                load_latency = vload.latency

        # Number of vector loads
        num_loads = math.ceil(load_data / (elements_per_vreg * bpe))
        memory_load_cycles = load_latency + num_loads

        num_stores = math.ceil(store_data / (elements_per_vreg * bpe))
        memory_store_cycles = 1 + num_stores

        memory_cycles = memory_load_cycles + memory_store_cycles

        entries.append(ScheduleEntry(
            operation="VLOAD",
            execution_unit="VLOAD",
            start_cycle=0,
            end_cycle=memory_load_cycles,
            elements=num_loads * elements_per_vreg,
            is_vector=True,
        ))

        entries.append(ScheduleEntry(
            operation="VSTORE",
            execution_unit="VSTORE",
            start_cycle=compute_cycles,
            end_cycle=compute_cycles + memory_store_cycles,
            elements=num_stores * elements_per_vreg,
            is_vector=True,
        ))

        # Total cycles: max(compute, memory) for overlapped execution
        total_cycles = max(compute_cycles, memory_cycles)

        # Utilization calculations
        pipeline_util = min(1.0, compute_cycles / total_cycles) if total_cycles > 0 else 0
        vector_util = (num_fma_instructions * elements_per_vreg) / total_macs if total_macs > 0 else 0

        return ScheduleResult(
            layer_name=name,
            entries=entries,
            total_cycles=total_cycles,
            compute_cycles=compute_cycles,
            memory_cycles=memory_cycles,
            pipeline_utilization=pipeline_util,
            vector_utilization=min(1.0, vector_util),
            unit_utilization={
                "VFMA": compute_cycles / total_cycles if total_cycles > 0 else 0,
                "VLOAD": memory_load_cycles / total_cycles if total_cycles > 0 else 0,
                "VSTORE": memory_store_cycles / total_cycles if total_cycles > 0 else 0,
            },
        )

    def schedule_elementwise(
        self,
        name: str,
        num_elements: int,
        operation: str = "add",
        dtype: str = "fp32"
    ) -> ScheduleResult:
        """
        Schedule an element-wise operation (add, mul, activation).
        """
        sew_bits = {"fp32": 32, "fp16": 16, "int8": 8}.get(dtype, 32)
        elements_per_vreg = self.vlen // sew_bits

        # Vector iterations
        num_vec_ops = math.ceil(num_elements / elements_per_vreg)

        # Operation latency
        op_latencies = {
            "add": 1, "mul": 2, "fma": 3,
            "gelu": 8, "relu": 1, "silu": 5,
            "exp": 6, "div": 12, "sqrt": 8,
        }
        latency = op_latencies.get(operation, 2)

        compute_cycles = latency + num_vec_ops

        # Memory: load input + store output
        bpe = sew_bits // 8
        load_bytes = num_elements * bpe
        store_bytes = num_elements * bpe

        num_loads = math.ceil(num_elements / elements_per_vreg)
        num_stores = num_loads

        memory_cycles = 3 + num_loads + 1 + num_stores

        total_cycles = max(compute_cycles, memory_cycles)

        return ScheduleResult(
            layer_name=name,
            total_cycles=total_cycles,
            compute_cycles=compute_cycles,
            memory_cycles=memory_cycles,
            pipeline_utilization=compute_cycles / total_cycles if total_cycles > 0 else 0,
            vector_utilization=1.0,
        )

    def schedule_softmax(
        self,
        name: str,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        dtype: str = "fp32"
    ) -> ScheduleResult:
        """
        Schedule softmax operation.

        Steps per row:
        1. Find max (reduction)
        2. Subtract max and exp (element-wise)
        3. Sum (reduction)
        4. Divide (element-wise)
        """
        num_rows = batch_size * num_heads * seq_len
        row_len = seq_len

        sew_bits = {"fp32": 32, "fp16": 16, "int8": 8}.get(dtype, 32)
        elements_per_vreg = self.vlen // sew_bits

        vec_ops_per_row = math.ceil(row_len / elements_per_vreg)

        # Per row: max_reduce + sub_exp + sum_reduce + divide
        max_cycles = 4 + vec_ops_per_row      # reduce max
        exp_cycles = 6 + vec_ops_per_row       # sub + exp
        sum_cycles = 4 + vec_ops_per_row       # reduce sum
        div_cycles = 1 + vec_ops_per_row       # divide

        cycles_per_row = max_cycles + exp_cycles + sum_cycles + div_cycles
        compute_cycles = cycles_per_row * num_rows

        # Memory
        data_bytes = num_rows * row_len * (sew_bits // 8)
        memory_cycles = math.ceil(data_bytes / 64) * 2  # Read + write

        total_cycles = max(compute_cycles, memory_cycles)

        return ScheduleResult(
            layer_name=name,
            total_cycles=total_cycles,
            compute_cycles=compute_cycles,
            memory_cycles=memory_cycles,
            pipeline_utilization=compute_cycles / total_cycles if total_cycles > 0 else 0,
            vector_utilization=0.8,  # Softmax has reductions → lower utilization
        )

    def schedule_layernorm(
        self,
        name: str,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        dtype: str = "fp32"
    ) -> ScheduleResult:
        """
        Schedule layer normalization.

        Steps per token:
        1. Compute mean (reduction)
        2. Compute variance (reduction)
        3. Normalize (element-wise)
        4. Scale and shift (element-wise)
        """
        num_tokens = batch_size * seq_len

        sew_bits = {"fp32": 32, "fp16": 16, "int8": 8}.get(dtype, 32)
        elements_per_vreg = self.vlen // sew_bits

        vec_ops = math.ceil(hidden_dim / elements_per_vreg)

        # Per token operations
        mean_cycles = 4 + vec_ops          # sum + divide
        var_cycles = 4 + vec_ops           # squared diff sum + divide
        norm_cycles = 1 + vec_ops          # subtract + divide
        scale_cycles = 1 + vec_ops         # multiply + add

        cycles_per_token = mean_cycles + var_cycles + norm_cycles + scale_cycles
        compute_cycles = cycles_per_token * num_tokens

        # Memory
        data_bytes = num_tokens * hidden_dim * (sew_bits // 8)
        weight_bytes = 2 * hidden_dim * (sew_bits // 8)  # gamma + beta
        memory_cycles = math.ceil((data_bytes * 2 + weight_bytes) / 64)

        total_cycles = max(compute_cycles, memory_cycles)

        return ScheduleResult(
            layer_name=name,
            total_cycles=total_cycles,
            compute_cycles=compute_cycles,
            memory_cycles=memory_cycles,
            pipeline_utilization=compute_cycles / total_cycles if total_cycles > 0 else 0,
            vector_utilization=0.85,
        )


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    scheduler = InstructionScheduler()

    print("=" * 60)
    print("MatMul Schedule: [128,768] × [768,3072]")
    print("=" * 60)

    result = scheduler.schedule_matmul(
        "fc1", M=128, N=3072, K=768
    )
    for key, val in result.summary().items():
        print(f"  {key}: {val}")

    print(f"\n{'='*60}")
    print("Softmax Schedule: [1, 12, 128, 128]")
    print(f"{'='*60}")

    result = scheduler.schedule_softmax(
        "softmax", batch_size=1, num_heads=12, seq_len=128
    )
    for key, val in result.summary().items():
        print(f"  {key}: {val}")