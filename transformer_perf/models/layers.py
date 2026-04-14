# Save as: C:\Users\ankit\transformer-perf-model\models\layers.py

"""
Layer-wise operation definitions for transformer models.
Defines compute and memory characteristics of each layer type.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional
import math


class LayerType(Enum):
    """Enumeration of supported transformer layer types"""
    LINEAR = auto()
    MATMUL = auto()
    LAYERNORM = auto()
    RMSNORM = auto()
    SOFTMAX = auto()
    GELU = auto()
    RELU = auto()
    SILU = auto()
    EMBEDDING = auto()
    ATTENTION = auto()
    MLP = auto()
    DROPOUT = auto()
    RESIDUAL_ADD = auto()
    RESHAPE = auto()
    TRANSPOSE = auto()
    UNKNOWN = auto()


@dataclass
class TensorShape:
    """Represents the shape of a tensor"""
    dims: Tuple[int, ...]

    @property
    def numel(self) -> int:
        """Total number of elements"""
        result = 1
        for d in self.dims:
            result *= d
        return result

    def size_bytes(self, dtype: str = "fp32") -> int:
        """Calculate size in bytes based on data type"""
        bytes_per_element = {
            "fp32": 4, "fp16": 2, "bf16": 2,
            "int8": 1, "int4": 0.5, "int32": 4
        }
        return int(self.numel * bytes_per_element.get(dtype, 4))

    def __repr__(self):
        return f"TensorShape({self.dims})"


@dataclass
class LayerDefinition:
    """
    Complete definition of a neural network layer
    including compute and memory characteristics.
    """

    # Basic info
    name: str
    layer_type: LayerType
    layer_id: int = 0

    # Shape information
    input_shapes: List[TensorShape] = field(default_factory=list)
    output_shape: Optional[TensorShape] = None
    weight_shapes: List[TensorShape] = field(default_factory=list)

    # Compute characteristics
    flops: int = 0                    # Floating point operations
    mac_ops: int = 0                  # Multiply-accumulate operations
    comparison_ops: int = 0           # For softmax, relu etc.
    transcendental_ops: int = 0       # exp, log, sqrt etc.

    # Memory characteristics
    weight_bytes: int = 0             # Weight memory in bytes
    activation_bytes: int = 0         # Activation memory in bytes
    output_bytes: int = 0             # Output memory in bytes
    total_memory_bytes: int = 0       # Total memory footprint

    # Additional parameters
    params: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived values after initialization"""
        if self.total_memory_bytes == 0:
            self.total_memory_bytes = (
                self.weight_bytes + self.activation_bytes + self.output_bytes
            )

    def arithmetic_intensity(self) -> float:
        """
        Calculate arithmetic intensity (FLOPs / Bytes)
        Higher = more compute bound
        Lower = more memory bound
        """
        total_data = self.total_memory_bytes
        if total_data == 0:
            return 0.0
        return self.flops / total_data

    def summary(self) -> Dict:
        """Return summary dictionary"""
        return {
            "name": self.name,
            "type": self.layer_type.name,
            "flops": self.flops,
            "mac_ops": self.mac_ops,
            "weight_bytes": self.weight_bytes,
            "activation_bytes": self.activation_bytes,
            "output_bytes": self.output_bytes,
            "total_memory_bytes": self.total_memory_bytes,
            "arithmetic_intensity": self.arithmetic_intensity(),
        }


class LayerProfiler:
    """
    Profiles transformer layers to compute FLOPs,
    memory requirements, and operation counts.
    """

    def __init__(self, dtype: str = "fp32"):
        self.dtype = dtype
        self.bytes_per_element = {
            "fp32": 4, "fp16": 2, "bf16": 2,
            "int8": 1, "int4": 0.5, "int32": 4
        }.get(dtype, 4)

    def profile_linear(
        self,
        name: str,
        batch_size: int,
        seq_len: int,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        layer_id: int = 0
    ) -> LayerDefinition:
        """
        Profile a Linear (fully connected) layer.

        FLOPs = 2 * batch * seq_len * in_features * out_features
        (multiply + add for each MAC)
        """

        # FLOPs calculation
        mac_ops = batch_size * seq_len * in_features * out_features
        flops = 2 * mac_ops  # Each MAC = 1 multiply + 1 add

        if has_bias:
            flops += batch_size * seq_len * out_features

        # Memory calculation
        weight_bytes = int(
            in_features * out_features * self.bytes_per_element
        )
        if has_bias:
            weight_bytes += int(out_features * self.bytes_per_element)

        activation_bytes = int(
            batch_size * seq_len * in_features * self.bytes_per_element
        )
        output_bytes = int(
            batch_size * seq_len * out_features * self.bytes_per_element
        )

        return LayerDefinition(
            name=name,
            layer_type=LayerType.LINEAR,
            layer_id=layer_id,
            input_shapes=[TensorShape((batch_size, seq_len, in_features))],
            output_shape=TensorShape((batch_size, seq_len, out_features)),
            weight_shapes=[TensorShape((out_features, in_features))],
            flops=flops,
            mac_ops=mac_ops,
            weight_bytes=weight_bytes,
            activation_bytes=activation_bytes,
            output_bytes=output_bytes,
            params={
                "in_features": in_features,
                "out_features": out_features,
                "has_bias": has_bias,
            },
        )

    def profile_attention(
        self,
        name: str,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        layer_id: int = 0
    ) -> LayerDefinition:
        """
        Profile Multi-Head Self-Attention layer.

        Components:
        1. Q, K, V projections: 3 × Linear(hidden, hidden)
        2. Q @ K^T: batch * heads * seq * seq * head_dim
        3. Softmax: batch * heads * seq * seq
        4. Attn @ V: batch * heads * seq * head_dim * seq
        5. Output projection: Linear(hidden, hidden)
        """

        head_dim = hidden_dim // num_heads

        # 1. Q, K, V projections (3 linear layers)
        qkv_macs = 3 * batch_size * seq_len * hidden_dim * hidden_dim
        qkv_flops = 2 * qkv_macs

        # 2. Q @ K^T
        qk_macs = batch_size * num_heads * seq_len * seq_len * head_dim
        qk_flops = 2 * qk_macs

        # 3. Softmax (exp, sum, divide per row)
        softmax_ops = batch_size * num_heads * seq_len * seq_len * 5
        transcendental = batch_size * num_heads * seq_len * seq_len  # exp ops

        # 4. Attention @ V
        attn_v_macs = batch_size * num_heads * seq_len * seq_len * head_dim
        attn_v_flops = 2 * attn_v_macs

        # 5. Output projection
        out_macs = batch_size * seq_len * hidden_dim * hidden_dim
        out_flops = 2 * out_macs

        # Totals
        total_macs = qkv_macs + qk_macs + attn_v_macs + out_macs
        total_flops = qkv_flops + qk_flops + attn_v_flops + out_flops + softmax_ops

        # Memory
        weight_bytes = int(
            4 * hidden_dim * hidden_dim * self.bytes_per_element  # Q,K,V,O weights
        )
        weight_bytes += int(
            4 * hidden_dim * self.bytes_per_element  # biases
        )

        activation_bytes = int(
            batch_size * seq_len * hidden_dim * self.bytes_per_element
        )

        # Intermediate: attention scores + attention weights
        intermediate_bytes = int(
            2 * batch_size * num_heads * seq_len * seq_len * self.bytes_per_element
        )

        output_bytes = int(
            batch_size * seq_len * hidden_dim * self.bytes_per_element
        )

        return LayerDefinition(
            name=name,
            layer_type=LayerType.ATTENTION,
            layer_id=layer_id,
            input_shapes=[TensorShape((batch_size, seq_len, hidden_dim))],
            output_shape=TensorShape((batch_size, seq_len, hidden_dim)),
            flops=total_flops,
            mac_ops=total_macs,
            transcendental_ops=transcendental,
            weight_bytes=weight_bytes,
            activation_bytes=activation_bytes + intermediate_bytes,
            output_bytes=output_bytes,
            params={
                "hidden_dim": hidden_dim,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "seq_len": seq_len,
            },
        )

    def profile_layernorm(
        self,
        name: str,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        layer_id: int = 0
    ) -> LayerDefinition:
        """
        Profile Layer Normalization.

        Operations: mean, variance, normalize, scale, shift
        FLOPs ≈ 5 * batch * seq_len * hidden_dim
        """

        num_elements = batch_size * seq_len * hidden_dim
        flops = 5 * num_elements  # mean, var, sub, div, scale+shift

        weight_bytes = int(
            2 * hidden_dim * self.bytes_per_element  # gamma + beta
        )
        activation_bytes = int(num_elements * self.bytes_per_element)
        output_bytes = int(num_elements * self.bytes_per_element)

        return LayerDefinition(
            name=name,
            layer_type=LayerType.LAYERNORM,
            layer_id=layer_id,
            input_shapes=[TensorShape((batch_size, seq_len, hidden_dim))],
            output_shape=TensorShape((batch_size, seq_len, hidden_dim)),
            flops=flops,
            weight_bytes=weight_bytes,
            activation_bytes=activation_bytes,
            output_bytes=output_bytes,
            params={"hidden_dim": hidden_dim},
        )

    def profile_mlp(
        self,
        name: str,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        intermediate_dim: int,
        activation: str = "gelu",
        layer_id: int = 0
    ) -> LayerDefinition:
        """
        Profile MLP/Feed-Forward Network.

        Components:
        1. Linear(hidden_dim → intermediate_dim)
        2. Activation (GELU/ReLU/SiLU)
        3. Linear(intermediate_dim → hidden_dim)
        """

        # Linear 1: hidden → intermediate
        l1_macs = batch_size * seq_len * hidden_dim * intermediate_dim
        l1_flops = 2 * l1_macs

        # Activation function
        act_elements = batch_size * seq_len * intermediate_dim
        if activation == "gelu":
            act_flops = act_elements * 8  # approximate GELU ops
            transcendental = act_elements
        elif activation == "relu":
            act_flops = act_elements  # comparison only
            transcendental = 0
        elif activation == "silu":
            act_flops = act_elements * 4  # sigmoid + multiply
            transcendental = act_elements
        else:
            act_flops = act_elements
            transcendental = 0

        # Linear 2: intermediate → hidden
        l2_macs = batch_size * seq_len * intermediate_dim * hidden_dim
        l2_flops = 2 * l2_macs

        # Totals
        total_macs = l1_macs + l2_macs
        total_flops = l1_flops + act_flops + l2_flops

        # Memory
        weight_bytes = int(
            (hidden_dim * intermediate_dim + intermediate_dim * hidden_dim)
            * self.bytes_per_element
        )
        weight_bytes += int(
            (intermediate_dim + hidden_dim) * self.bytes_per_element  # biases
        )

        activation_bytes = int(
            batch_size * seq_len * hidden_dim * self.bytes_per_element
        )
        intermediate_bytes = int(
            batch_size * seq_len * intermediate_dim * self.bytes_per_element
        )
        output_bytes = int(
            batch_size * seq_len * hidden_dim * self.bytes_per_element
        )

        return LayerDefinition(
            name=name,
            layer_type=LayerType.MLP,
            layer_id=layer_id,
            input_shapes=[TensorShape((batch_size, seq_len, hidden_dim))],
            output_shape=TensorShape((batch_size, seq_len, hidden_dim)),
            flops=total_flops,
            mac_ops=total_macs,
            transcendental_ops=transcendental,
            weight_bytes=weight_bytes,
            activation_bytes=activation_bytes + intermediate_bytes,
            output_bytes=output_bytes,
            params={
                "hidden_dim": hidden_dim,
                "intermediate_dim": intermediate_dim,
                "activation": activation,
            },
        )

    def profile_softmax(
        self,
        name: str,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        layer_id: int = 0
    ) -> LayerDefinition:
        """
        Profile Softmax operation.

        Operations per element: exp, sum-reduce, divide
        """

        num_elements = batch_size * num_heads * seq_len * seq_len
        flops = 5 * num_elements  # max, sub, exp, sum, div
        transcendental = num_elements  # exp operations

        activation_bytes = int(num_elements * self.bytes_per_element)
        output_bytes = int(num_elements * self.bytes_per_element)

        return LayerDefinition(
            name=name,
            layer_type=LayerType.SOFTMAX,
            layer_id=layer_id,
            input_shapes=[
                TensorShape((batch_size, num_heads, seq_len, seq_len))
            ],
            output_shape=TensorShape(
                (batch_size, num_heads, seq_len, seq_len)
            ),
            flops=flops,
            transcendental_ops=transcendental,
            activation_bytes=activation_bytes,
            output_bytes=output_bytes,
            params={"num_heads": num_heads, "seq_len": seq_len},
        )

    def profile_embedding(
        self,
        name: str,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        hidden_dim: int,
        layer_id: int = 0
    ) -> LayerDefinition:
        """Profile Embedding lookup layer"""

        # Embedding is primarily a memory operation (lookup)
        flops = 0  # No compute, just memory lookup
        weight_bytes = int(
            vocab_size * hidden_dim * self.bytes_per_element
        )
        output_bytes = int(
            batch_size * seq_len * hidden_dim * self.bytes_per_element
        )

        return LayerDefinition(
            name=name,
            layer_type=LayerType.EMBEDDING,
            layer_id=layer_id,
            input_shapes=[TensorShape((batch_size, seq_len))],
            output_shape=TensorShape((batch_size, seq_len, hidden_dim)),
            flops=flops,
            weight_bytes=weight_bytes,
            activation_bytes=0,
            output_bytes=output_bytes,
            params={
                "vocab_size": vocab_size,
                "hidden_dim": hidden_dim,
            },
        )

    def profile_residual_add(
        self,
        name: str,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        layer_id: int = 0
    ) -> LayerDefinition:
        """Profile Residual Addition"""

        num_elements = batch_size * seq_len * hidden_dim
        flops = num_elements  # One add per element

        activation_bytes = int(
            2 * num_elements * self.bytes_per_element  # Two inputs
        )
        output_bytes = int(num_elements * self.bytes_per_element)

        return LayerDefinition(
            name=name,
            layer_type=LayerType.RESIDUAL_ADD,
            layer_id=layer_id,
            input_shapes=[
                TensorShape((batch_size, seq_len, hidden_dim)),
                TensorShape((batch_size, seq_len, hidden_dim)),
            ],
            output_shape=TensorShape((batch_size, seq_len, hidden_dim)),
            flops=flops,
            activation_bytes=activation_bytes,
            output_bytes=output_bytes,
        )


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    profiler = LayerProfiler(dtype="fp32")

    # Profile a BERT-base attention layer
    attn = profiler.profile_attention(
        name="encoder.layer.0.attention",
        batch_size=1,
        seq_len=128,
        hidden_dim=768,
        num_heads=12,
    )
    print("=== Attention Layer ===")
    for key, value in attn.summary().items():
        if isinstance(value, (int, float)) and value > 1000:
            print(f"  {key}: {value:,.0f}")
        else:
            print(f"  {key}: {value}")

    print()

    # Profile a Linear layer
    linear = profiler.profile_linear(
        name="encoder.layer.0.intermediate.dense",
        batch_size=1,
        seq_len=128,
        in_features=768,
        out_features=3072,
    )
    print("=== Linear Layer ===")
    for key, value in linear.summary().items():
        if isinstance(value, (int, float)) and value > 1000:
            print(f"  {key}: {value:,.0f}")
        else:
            print(f"  {key}: {value}")