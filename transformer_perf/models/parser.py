# Save as: C:\Users\ankit\transformer-perf-model\models\parser.py

"""
Parse transformer models from various frameworks
(PyTorch, ONNX, HuggingFace) into our internal
ComputationGraph representation.
"""

from typing import Dict, Optional
from .graph import ComputationGraph
from .layers import LayerProfiler, LayerType


class ModelConfig:
    """Configuration for a transformer model"""

    # Pre-defined model configurations
    PRESETS = {
        "bert-base": {
            "hidden_dim": 768,
            "num_heads": 12,
            "num_layers": 12,
            "intermediate_dim": 3072,
            "vocab_size": 30522,
            "max_seq_len": 512,
            "activation": "gelu",
            "model_type": "encoder",
        },
        "bert-large": {
            "hidden_dim": 1024,
            "num_heads": 16,
            "num_layers": 24,
            "intermediate_dim": 4096,
            "vocab_size": 30522,
            "max_seq_len": 512,
            "activation": "gelu",
            "model_type": "encoder",
        },
        "gpt2-small": {
            "hidden_dim": 768,
            "num_heads": 12,
            "num_layers": 12,
            "intermediate_dim": 3072,
            "vocab_size": 50257,
            "max_seq_len": 1024,
            "activation": "gelu",
            "model_type": "decoder",
        },
        "gpt2-medium": {
            "hidden_dim": 1024,
            "num_heads": 16,
            "num_layers": 24,
            "intermediate_dim": 4096,
            "vocab_size": 50257,
            "max_seq_len": 1024,
            "activation": "gelu",
            "model_type": "decoder",
        },
        "llama-7b": {
            "hidden_dim": 4096,
            "num_heads": 32,
            "num_layers": 32,
            "intermediate_dim": 11008,
            "vocab_size": 32000,
            "max_seq_len": 2048,
            "activation": "silu",
            "model_type": "decoder",
        },
        "vit-base": {
            "hidden_dim": 768,
            "num_heads": 12,
            "num_layers": 12,
            "intermediate_dim": 3072,
            "vocab_size": 0,  # No vocab for ViT
            "max_seq_len": 197,  # 196 patches + 1 CLS token
            "activation": "gelu",
            "model_type": "encoder",
        },
    }

    def __init__(self, preset: str = None, **kwargs):
        if preset and preset in self.PRESETS:
            config = self.PRESETS[preset].copy()
            config.update(kwargs)  # Allow overrides
        else:
            config = kwargs

        self.hidden_dim = config.get("hidden_dim", 768)
        self.num_heads = config.get("num_heads", 12)
        self.num_layers = config.get("num_layers", 12)
        self.intermediate_dim = config.get("intermediate_dim", 3072)
        self.vocab_size = config.get("vocab_size", 30522)
        self.max_seq_len = config.get("max_seq_len", 512)
        self.activation = config.get("activation", "gelu")
        self.model_type = config.get("model_type", "encoder")

    def summary(self) -> Dict:
        return {
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "intermediate_dim": self.intermediate_dim,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "activation": self.activation,
            "model_type": self.model_type,
        }


class TransformerGraphParser:
    """
    Parse transformer models into ComputationGraph.

    Supports:
    - Pre-defined model configs (BERT, GPT-2, LLaMA, ViT)
    - Custom model configurations
    - PyTorch model files (future)
    - ONNX model files (future)
    """

    def __init__(self, dtype: str = "fp32"):
        self.dtype = dtype
        self.profiler = LayerProfiler(dtype=dtype)

    def parse_from_config(
        self,
        model_config: ModelConfig,
        batch_size: int = 1,
        seq_len: int = 128
    ) -> ComputationGraph:
        """
        Build computation graph from model configuration.

        Args:
            model_config: Model architecture configuration
            batch_size: Batch size for inference
            seq_len: Input sequence length

        Returns:
            ComputationGraph with all layers profiled
        """
        graph = ComputationGraph(name=f"transformer-b{batch_size}-s{seq_len}")

        cfg = model_config
        layer_id = 0

        # ---- Embedding Layer ----
        if cfg.vocab_size > 0:
            emb = self.profiler.profile_embedding(
                name="embedding",
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=cfg.vocab_size,
                hidden_dim=cfg.hidden_dim,
                layer_id=layer_id,
            )
            emb_id = graph.add_node(emb)
            prev_id = emb_id
            layer_id += 1
        else:
            prev_id = None

        # ---- Transformer Blocks ----
        for block_idx in range(cfg.num_layers):
            prev_id, layer_id = self._build_transformer_block(
                graph=graph,
                block_idx=block_idx,
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                intermediate_dim=cfg.intermediate_dim,
                activation=cfg.activation,
                prev_id=prev_id,
                start_layer_id=layer_id,
            )

        # ---- Final Layer Norm ----
        final_ln = self.profiler.profile_layernorm(
            name="final_layernorm",
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=cfg.hidden_dim,
            layer_id=layer_id,
        )
        predecessors = [prev_id] if prev_id is not None else []
        graph.add_node(final_ln, predecessors=predecessors)

        return graph

    def _build_transformer_block(
        self,
        graph: ComputationGraph,
        block_idx: int,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: int,
        activation: str,
        prev_id: Optional[int],
        start_layer_id: int
    ) -> tuple:
        """
        Build a single transformer block and add to graph.

        Structure:
            LayerNorm → Attention → Residual →
            LayerNorm → MLP → Residual
        """

        lid = start_layer_id
        prefix = f"block_{block_idx}"

        # ---- Pre-Attention LayerNorm ----
        ln1 = self.profiler.profile_layernorm(
            name=f"{prefix}.layernorm_1",
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            layer_id=lid,
        )
        predecessors = [prev_id] if prev_id is not None else []
        ln1_id = graph.add_node(ln1, predecessors=predecessors)
        lid += 1

        # ---- Self-Attention ----
        attn = self.profiler.profile_attention(
            name=f"{prefix}.self_attention",
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            layer_id=lid,
        )
        attn_id = graph.add_node(attn, predecessors=[ln1_id])
        lid += 1

        # ---- Residual Connection 1 ----
        res1 = self.profiler.profile_residual_add(
            name=f"{prefix}.residual_1",
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            layer_id=lid,
        )
        # Residual connects attention output + skip connection
        res1_predecessors = [attn_id]
        if prev_id is not None:
            res1_predecessors.append(prev_id)
        res1_id = graph.add_node(res1, predecessors=res1_predecessors)
        lid += 1

        # ---- Pre-MLP LayerNorm ----
        ln2 = self.profiler.profile_layernorm(
            name=f"{prefix}.layernorm_2",
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            layer_id=lid,
        )
        ln2_id = graph.add_node(ln2, predecessors=[res1_id])
        lid += 1

        # ---- MLP ----
        mlp = self.profiler.profile_mlp(
            name=f"{prefix}.mlp",
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            activation=activation,
            layer_id=lid,
        )
        mlp_id = graph.add_node(mlp, predecessors=[ln2_id])
        lid += 1

        # ---- Residual Connection 2 ----
        res2 = self.profiler.profile_residual_add(
            name=f"{prefix}.residual_2",
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            layer_id=lid,
        )
        res2_id = graph.add_node(res2, predecessors=[mlp_id, res1_id])
        lid += 1

        return res2_id, lid

    def parse_from_preset(
        self,
        preset_name: str,
        batch_size: int = 1,
        seq_len: int = 128
    ) -> ComputationGraph:
        """
        Parse a pre-defined model by name.

        Args:
            preset_name: One of 'bert-base', 'gpt2-small', 'llama-7b', etc.
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            ComputationGraph
        """
        config = ModelConfig(preset=preset_name)
        graph = self.parse_from_config(config, batch_size, seq_len)
        graph.name = f"{preset_name}-b{batch_size}-s{seq_len}"
        return graph


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    parser = TransformerGraphParser(dtype="fp32")

    # Parse BERT-base
    graph = parser.parse_from_preset(
        "bert-base", batch_size=1, seq_len=128
    )
    graph.print_summary()

    print(f"\nTotal FLOPs: {graph.get_total_flops():,.0f}")
    print(f"Total Memory: {graph.get_total_memory()/1e6:.2f} MB")
    print(f"Total Weights: {graph.get_total_weights()/1e6:.2f} MB")