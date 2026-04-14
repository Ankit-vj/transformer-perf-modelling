# Save as: C:\Users\ankit\transformer-perf-model\transformer_perf\mapping\dataflow.py

"""
Data Movement Pattern Analysis.
Analyzes how data flows between compute units and memory
during transformer inference.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


@dataclass
class DataMovement:
    """Represents a single data movement operation"""
    name: str
    source: str          # "DRAM", "L3", "L2", "L1D", "REG"
    destination: str     # "DRAM", "L3", "L2", "L1D", "REG"
    size_bytes: int
    num_transfers: int = 1
    reuse_count: int = 1

    @property
    def total_bytes(self) -> int:
        return self.size_bytes * self.num_transfers

    @property
    def effective_bytes(self) -> int:
        """Bytes after accounting for reuse"""
        return self.total_bytes // max(self.reuse_count, 1)


@dataclass
class DataflowResult:
    """Complete dataflow analysis result"""
    layer_name: str
    movements: List[DataMovement] = field(default_factory=list)
    total_read_bytes: int = 0
    total_write_bytes: int = 0
    total_data_movement_bytes: int = 0
    arithmetic_intensity: float = 0.0
    memory_bound: bool = False
    bottleneck_level: str = "L1D"

    def summary(self) -> Dict:
        return {
            "layer_name": self.layer_name,
            "total_read_bytes": self.total_read_bytes,
            "total_write_bytes": self.total_write_bytes,
            "total_movement_bytes": self.total_data_movement_bytes,
            "arithmetic_intensity": f"{self.arithmetic_intensity:.2f} FLOP/byte",
            "memory_bound": self.memory_bound,
            "bottleneck": self.bottleneck_level,
            "num_movements": len(self.movements),
        }


class DataflowAnalyzer:
    """
    Analyze data movement patterns for transformer layers.
    Determines where data resides and how it moves through
    the memory hierarchy during computation.
    """

    def __init__(self, memory_hierarchy=None):
        self.memory = memory_hierarchy

        # Default cache sizes
        self.cache_sizes = {
            "L1D": 32 * 1024,
            "L2": 256 * 1024,
            "L3": 4 * 1024 * 1024,
            "DRAM": 8 * 1024 * 1024 * 1024,
        }

        # Default bandwidths (bytes/cycle)
        self.bandwidths = {
            "L1D": 64.0,
            "L2": 32.0,
            "L3": 16.0,
            "DRAM": 8.0,
        }

        if memory_hierarchy:
            self._load_from_hierarchy(memory_hierarchy)

    def _load_from_hierarchy(self, mem):
        """Load cache sizes from memory hierarchy"""
        self.cache_sizes = {
            "L1D": mem.l1d.size_bytes,
            "L2": mem.l2.size_bytes,
            "L3": mem.l3.size_bytes,
            "DRAM": mem.dram.capacity_gb * 1024 * 1024 * 1024,
        }
        self.bandwidths = {
            "L1D": mem.l1d.bandwidth_bytes_per_cycle,
            "L2": mem.l2.bandwidth_bytes_per_cycle,
            "L3": mem.l3.bandwidth_bytes_per_cycle,
            "DRAM": mem.dram.peak_bandwidth_gbps * 1e9 / 2e9,
        }

    def _find_data_level(self, data_size: int) -> str:
        """Determine which cache level data resides in"""
        for level in ["L1D", "L2", "L3"]:
            if data_size <= self.cache_sizes[level]:
                return level
        return "DRAM"

    def analyze_linear_layer(
        self,
        name: str,
        batch_size: int,
        seq_len: int,
        in_features: int,
        out_features: int,
        bytes_per_element: int = 4,
        flops: int = 0
    ) -> DataflowResult:
        """
        Analyze data movement for a linear layer.

        Data:
          - Input activation: [batch*seq, in_features]
          - Weight matrix: [in_features, out_features]
          - Output activation: [batch*seq, out_features]
          - Bias: [out_features]
        """
        M = batch_size * seq_len
        bpe = bytes_per_element

        # Data sizes
        input_bytes = M * in_features * bpe
        weight_bytes = in_features * out_features * bpe
        output_bytes = M * out_features * bpe
        bias_bytes = out_features * bpe

        # Determine where data lives
        weight_level = self._find_data_level(weight_bytes)
        input_level = self._find_data_level(input_bytes)

        movements = []

        # Weight loading
        movements.append(DataMovement(
            name="load_weights",
            source=weight_level,
            destination="REG",
            size_bytes=weight_bytes,
            num_transfers=1,
            reuse_count=M,  # Weights reused across all M rows
        ))

        # Input activation loading
        movements.append(DataMovement(
            name="load_input",
            source=input_level,
            destination="REG",
            size_bytes=input_bytes,
            num_transfers=1,
            reuse_count=1,
        ))

        # Bias loading
        movements.append(DataMovement(
            name="load_bias",
            source=weight_level,
            destination="REG",
            size_bytes=bias_bytes,
            num_transfers=1,
            reuse_count=M,
        ))

        # Output writing
        output_level = self._find_data_level(output_bytes)
        movements.append(DataMovement(
            name="store_output",
            source="REG",
            destination=output_level,
            size_bytes=output_bytes,
            num_transfers=1,
        ))

        # Calculate totals
        total_read = input_bytes + weight_bytes + bias_bytes
        total_write = output_bytes
        total_movement = total_read + total_write

        # Arithmetic intensity
        if flops == 0:
            flops = 2 * M * in_features * out_features
        ai = flops / total_movement if total_movement > 0 else 0

        # Determine bottleneck
        bottleneck = self._find_bottleneck(movements)

        return DataflowResult(
            layer_name=name,
            movements=movements,
            total_read_bytes=total_read,
            total_write_bytes=total_write,
            total_data_movement_bytes=total_movement,
            arithmetic_intensity=ai,
            memory_bound=(ai < 10.0),  # Rough threshold
            bottleneck_level=bottleneck,
        )

    def analyze_attention_layer(
        self,
        name: str,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        hidden_dim: int,
        bytes_per_element: int = 4
    ) -> DataflowResult:
        """
        Analyze data movement for multi-head attention.
        """
        head_dim = hidden_dim // num_heads
        bpe = bytes_per_element
        B, H, S, D = batch_size, num_heads, seq_len, head_dim

        movements = []

        # QKV projection weights
        qkv_weight_bytes = 3 * hidden_dim * hidden_dim * bpe
        qkv_weight_level = self._find_data_level(qkv_weight_bytes)
        movements.append(DataMovement(
            name="load_qkv_weights",
            source=qkv_weight_level,
            destination="REG",
            size_bytes=qkv_weight_bytes,
            reuse_count=B * S,
        ))

        # Input activation
        input_bytes = B * S * hidden_dim * bpe
        movements.append(DataMovement(
            name="load_input",
            source=self._find_data_level(input_bytes),
            destination="REG",
            size_bytes=input_bytes,
        ))

        # Q, K, V tensors (intermediate)
        qkv_bytes = 3 * B * H * S * D * bpe
        movements.append(DataMovement(
            name="store_qkv",
            source="REG",
            destination=self._find_data_level(qkv_bytes // 3),
            size_bytes=qkv_bytes,
        ))

        # Attention scores: [B, H, S, S]
        attn_score_bytes = B * H * S * S * bpe
        movements.append(DataMovement(
            name="attention_scores",
            source=self._find_data_level(attn_score_bytes),
            destination=self._find_data_level(attn_score_bytes),
            size_bytes=attn_score_bytes * 2,  # Read Q,K + write scores
        ))

        # Softmax (in-place, read+write)
        movements.append(DataMovement(
            name="softmax",
            source=self._find_data_level(attn_score_bytes),
            destination=self._find_data_level(attn_score_bytes),
            size_bytes=attn_score_bytes * 2,
        ))

        # Attention @ V
        attn_v_bytes = B * H * S * D * bpe
        movements.append(DataMovement(
            name="attention_value",
            source=self._find_data_level(attn_v_bytes),
            destination=self._find_data_level(attn_v_bytes),
            size_bytes=attn_score_bytes + attn_v_bytes,
        ))

        # Output projection
        out_weight_bytes = hidden_dim * hidden_dim * bpe
        movements.append(DataMovement(
            name="output_projection",
            source=self._find_data_level(out_weight_bytes),
            destination="REG",
            size_bytes=out_weight_bytes + input_bytes,
        ))

        # Calculate totals
        total_read = sum(m.size_bytes for m in movements if m.destination == "REG")
        total_write = sum(m.size_bytes for m in movements if m.source == "REG")
        total_movement = sum(m.total_bytes for m in movements)

        # Total FLOPs for attention
        total_flops = (
            3 * 2 * B * S * hidden_dim * hidden_dim +  # QKV projections
            2 * B * H * S * S * D +                      # QK^T
            5 * B * H * S * S +                           # Softmax
            2 * B * H * S * S * D +                      # Attn @ V
            2 * B * S * hidden_dim * hidden_dim           # Output projection
        )

        ai = total_flops / total_movement if total_movement > 0 else 0
        bottleneck = self._find_bottleneck(movements)

        return DataflowResult(
            layer_name=name,
            movements=movements,
            total_read_bytes=total_read,
            total_write_bytes=total_write,
            total_data_movement_bytes=total_movement,
            arithmetic_intensity=ai,
            memory_bound=(ai < 10.0),
            bottleneck_level=bottleneck,
        )

    def analyze_graph(
        self,
        graph,
        bytes_per_element: int = 4
    ) -> List[DataflowResult]:
        """
        Analyze dataflow for entire computation graph.

        Args:
            graph: ComputationGraph object
            bytes_per_element: Size of each data element

        Returns:
            List of DataflowResult for each layer
        """
        results = []

        for node_id in graph.get_topological_order():
            node = graph.get_node(node_id)
            layer = node.layer
            params = layer.params

            if layer.layer_type.name == "LINEAR":
                result = self.analyze_linear_layer(
                    name=layer.name,
                    batch_size=layer.input_shapes[0].dims[0],
                    seq_len=layer.input_shapes[0].dims[1],
                    in_features=params.get("in_features", 768),
                    out_features=params.get("out_features", 768),
                    bytes_per_element=bytes_per_element,
                    flops=layer.flops,
                )
                results.append(result)

            elif layer.layer_type.name == "ATTENTION":
                result = self.analyze_attention_layer(
                    name=layer.name,
                    batch_size=layer.input_shapes[0].dims[0],
                    num_heads=params.get("num_heads", 12),
                    seq_len=params.get("seq_len", 128),
                    hidden_dim=params.get("hidden_dim", 768),
                    bytes_per_element=bytes_per_element,
                )
                results.append(result)

            else:
                # Generic analysis for other layers
                total_data = layer.total_memory_bytes
                ai = layer.flops / total_data if total_data > 0 else 0
                results.append(DataflowResult(
                    layer_name=layer.name,
                    total_read_bytes=layer.activation_bytes,
                    total_write_bytes=layer.output_bytes,
                    total_data_movement_bytes=total_data,
                    arithmetic_intensity=ai,
                    memory_bound=(ai < 10.0),
                    bottleneck_level=self._find_data_level(total_data),
                ))

        return results

    def _find_bottleneck(self, movements: List[DataMovement]) -> str:
        """Find the memory level that is the bottleneck"""
        level_bytes = {}
        for m in movements:
            for level in [m.source, m.destination]:
                if level != "REG":
                    level_bytes[level] = (
                        level_bytes.get(level, 0) + m.total_bytes
                    )

        if not level_bytes:
            return "L1D"

        # Bottleneck = level with most data movement relative to bandwidth
        max_time = 0
        bottleneck = "L1D"

        for level, total_bytes in level_bytes.items():
            bw = self.bandwidths.get(level, 64.0)
            time = total_bytes / bw
            if time > max_time:
                max_time = time
                bottleneck = level

        return bottleneck


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    analyzer = DataflowAnalyzer()

    print("=" * 60)
    print("Linear Layer Dataflow Analysis")
    print("=" * 60)

    result = analyzer.analyze_linear_layer(
        name="fc1",
        batch_size=1, seq_len=128,
        in_features=768, out_features=3072,
    )

    for key, val in result.summary().items():
        print(f"  {key}: {val}")

    print("\n" + "=" * 60)
    print("Attention Layer Dataflow Analysis")
    print("=" * 60)

    result = analyzer.analyze_attention_layer(
        name="self_attn",
        batch_size=1, num_heads=12,
        seq_len=128, hidden_dim=768,
    )

    for key, val in result.summary().items():
        print(f"  {key}: {val}")