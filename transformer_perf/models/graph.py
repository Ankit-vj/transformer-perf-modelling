# Save as: C:\Users\ankit\transformer-perf-model\models\graph.py

"""
Computational graph representation for transformer models.
Uses Directed Acyclic Graph (DAG) to represent data flow.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import networkx as nx

from .layers import LayerDefinition, LayerType


@dataclass
class GraphNode:
    """A node in the computational graph"""
    node_id: int
    layer: LayerDefinition
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)


class ComputationGraph:
    """
    Directed Acyclic Graph (DAG) representation of
    a transformer model's computation flow.
    """

    def __init__(self, name: str = "transformer"):
        self.name = name
        self.nodes: Dict[int, GraphNode] = {}
        self.graph = nx.DiGraph()
        self._next_id = 0

    def add_node(
        self,
        layer: LayerDefinition,
        predecessors: Optional[List[int]] = None
    ) -> int:
        """
        Add a computation node to the graph.

        Args:
            layer: Layer definition with compute/memory info
            predecessors: List of predecessor node IDs

        Returns:
            Node ID of the added node
        """
        node_id = self._next_id
        self._next_id += 1

        pred_list = predecessors if predecessors else []

        node = GraphNode(
            node_id=node_id,
            layer=layer,
            predecessors=pred_list,
        )

        self.nodes[node_id] = node
        self.graph.add_node(
            node_id,
            name=layer.name,
            layer_type=layer.layer_type.name,
            flops=layer.flops,
            memory=layer.total_memory_bytes,
        )

        # Add edges from predecessors
        for pred_id in pred_list:
            self.graph.add_edge(pred_id, node_id)
            if pred_id in self.nodes:
                self.nodes[pred_id].successors.append(node_id)

        return node_id

    def get_node(self, node_id: int) -> Optional[GraphNode]:
        """Get a node by its ID"""
        return self.nodes.get(node_id)

    def get_topological_order(self) -> List[int]:
        """
        Get nodes in topological order (execution order).
        """
        return list(nx.topological_sort(self.graph))

    def get_critical_path(self) -> Tuple[List[int], int]:
        """
        Find the critical path (longest path) through the graph.
        Returns node IDs on the critical path and total FLOPs.
        """
        # Use longest path in DAG
        longest_path = nx.dag_longest_path(
            self.graph,
            weight="flops",
            default_weight=0
        )
        total_flops = sum(
            self.nodes[nid].layer.flops for nid in longest_path
        )
        return longest_path, total_flops

    def get_total_flops(self) -> int:
        """Get total FLOPs across all nodes"""
        return sum(node.layer.flops for node in self.nodes.values())

    def get_total_memory(self) -> int:
        """Get total memory footprint in bytes"""
        return sum(
            node.layer.total_memory_bytes for node in self.nodes.values()
        )

    def get_total_weights(self) -> int:
        """Get total weight memory in bytes"""
        return sum(
            node.layer.weight_bytes for node in self.nodes.values()
        )

    def get_layer_breakdown(self) -> Dict[str, Dict]:
        """
        Get breakdown of compute and memory by layer type.
        """
        breakdown = {}

        for node in self.nodes.values():
            lt = node.layer.layer_type.name
            if lt not in breakdown:
                breakdown[lt] = {
                    "count": 0,
                    "total_flops": 0,
                    "total_memory": 0,
                    "total_weights": 0,
                }
            breakdown[lt]["count"] += 1
            breakdown[lt]["total_flops"] += node.layer.flops
            breakdown[lt]["total_memory"] += node.layer.total_memory_bytes
            breakdown[lt]["total_weights"] += node.layer.weight_bytes

        return breakdown

    def get_execution_schedule(self) -> List[LayerDefinition]:
        """Get layers in execution order"""
        order = self.get_topological_order()
        return [self.nodes[nid].layer for nid in order]

    def summary(self) -> Dict:
        """Get graph summary statistics"""
        return {
            "name": self.name,
            "num_nodes": len(self.nodes),
            "num_edges": self.graph.number_of_edges(),
            "total_flops": self.get_total_flops(),
            "total_memory_bytes": self.get_total_memory(),
            "total_weight_bytes": self.get_total_weights(),
            "layer_breakdown": self.get_layer_breakdown(),
        }

    def print_summary(self):
        """Print a formatted summary"""
        summary = self.summary()

        print(f"\n{'='*60}")
        print(f"Computation Graph: {summary['name']}")
        print(f"{'='*60}")
        print(f"Total Nodes      : {summary['num_nodes']}")
        print(f"Total Edges      : {summary['num_edges']}")
        print(f"Total FLOPs      : {summary['total_flops']:,.0f}")
        print(f"Total Memory     : {summary['total_memory_bytes']/1e6:.2f} MB")
        print(f"Total Weights    : {summary['total_weight_bytes']/1e6:.2f} MB")
        print(f"\nLayer Breakdown:")
        print(f"{'-'*60}")

        for lt, info in summary["layer_breakdown"].items():
            pct = (
                info["total_flops"] / summary["total_flops"] * 100
                if summary["total_flops"] > 0 else 0
            )
            print(
                f"  {lt:<20} count={info['count']:<4} "
                f"FLOPs={info['total_flops']:>15,}  ({pct:5.1f}%)"
            )

    def export_to_dict(self) -> Dict:
        """Export graph to a serializable dictionary"""
        nodes_data = []
        for nid, node in self.nodes.items():
            nodes_data.append({
                "id": nid,
                "name": node.layer.name,
                "type": node.layer.layer_type.name,
                "flops": node.layer.flops,
                "memory": node.layer.total_memory_bytes,
                "predecessors": node.predecessors,
                "successors": node.successors,
            })

        edges_data = list(self.graph.edges())

        return {
            "name": self.name,
            "nodes": nodes_data,
            "edges": edges_data,
        }


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    from .layers import LayerProfiler

    profiler = LayerProfiler(dtype="fp32")
    graph = ComputationGraph(name="bert-base-layer-0")

    # Build one transformer block
    batch, seq, hidden, heads = 1, 128, 768, 12
    intermediate = 3072

    # Layer Norm 1
    ln1 = profiler.profile_layernorm(
        "ln1", batch, seq, hidden
    )
    ln1_id = graph.add_node(ln1)

    # Self-Attention
    attn = profiler.profile_attention(
        "self_attn", batch, seq, hidden, heads
    )
    attn_id = graph.add_node(attn, predecessors=[ln1_id])

    # Residual Add 1
    res1 = profiler.profile_residual_add(
        "residual_1", batch, seq, hidden
    )
    res1_id = graph.add_node(res1, predecessors=[attn_id])

    # Layer Norm 2
    ln2 = profiler.profile_layernorm(
        "ln2", batch, seq, hidden
    )
    ln2_id = graph.add_node(ln2, predecessors=[res1_id])

    # MLP
    mlp = profiler.profile_mlp(
        "mlp", batch, seq, hidden, intermediate
    )
    mlp_id = graph.add_node(mlp, predecessors=[ln2_id])

    # Residual Add 2
    res2 = profiler.profile_residual_add(
        "residual_2", batch, seq, hidden
    )
    res2_id = graph.add_node(res2, predecessors=[mlp_id])

    graph.print_summary()