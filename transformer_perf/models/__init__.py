# Save as: C:\Users\ankit\transformer-perf-model\models\__init__.py

"""
Models Module
=============
Handles transformer model parsing, computational graph
representation, and layer-wise operation definitions.

Components:
    parser  - Parse models from PyTorch, ONNX, TFLite formats
    graph   - Computational graph representation using DAG
    layers  - Layer-wise operation definitions with FLOP counts
"""

from .parser import TransformerGraphParser
from .graph import ComputationGraph
from .layers import LayerDefinition, LayerType

__all__ = [
    "TransformerGraphParser",
    "ComputationGraph",
    "LayerDefinition",
    "LayerType",
]