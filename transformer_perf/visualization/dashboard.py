# Save as: C:\Users\ankit\transformer-perf-model\transformer_perf\visualization\dashboard.py

"""
Visualization Dashboard for Performance Analysis.
Generates charts, reports, and visual summaries of
transformer inference performance on RISC-V SoCs.
"""

import os
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving files
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class PerformanceDashboard:
    """
    Generate comprehensive performance visualization
    dashboards for transformer inference analysis.
    """

    def __init__(self, output_dir: str = "results"):
        """
        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Color scheme
        self.colors = {
            "primary": "#2196F3",
            "secondary": "#FF9800",
            "success": "#4CAF50",
            "danger": "#F44336",
            "warning": "#FFC107",
            "info": "#00BCD4",
            "dark": "#37474F",
            "light": "#ECEFF1",
            "compute": "#1976D2",
            "memory": "#E64A19",
            "attention": "#7B1FA2",
            "mlp": "#388E3C",
            "layernorm": "#FFA000",
            "other": "#78909C",
        }

        # Layer type color mapping
        self.layer_colors = {
            "ATTENTION": self.colors["attention"],
            "MLP": self.colors["mlp"],
            "LINEAR": self.colors["compute"],
            "LAYERNORM": self.colors["layernorm"],
            "SOFTMAX": self.colors["warning"],
            "EMBEDDING": self.colors["info"],
            "RESIDUAL_ADD": self.colors["other"],
        }

    def plot_latency_breakdown(
        self,
        model_latency,
        filename: str = "latency_breakdown.png"
    ):
        """
        Plot latency breakdown by layer type.

        Args:
            model_latency: ModelLatency object from LatencyEstimator
            filename: Output filename
        """
        if not HAS_MATPLOTLIB:
            print("WARNING: matplotlib not available, skipping plot")
            return

        # Aggregate by layer type
        type_latency = {}
        for layer in model_latency.layer_breakdown:
            lt = layer.layer_type
            if lt not in type_latency:
                type_latency[lt] = 0.0
            type_latency[lt] += layer.latency_ms

        # Sort by latency
        sorted_types = sorted(
            type_latency.items(), key=lambda x: x[1], reverse=True
        )

        labels = [t[0] for t in sorted_types]
        values = [t[1] for t in sorted_types]
        colors = [self.layer_colors.get(l, self.colors["other"]) for l in labels]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart
        wedges, texts, autotexts = axes[0].pie(
            values, labels=None, autopct='%1.1f%%',
            colors=colors, startangle=90,
            pctdistance=0.85
        )
        axes[0].set_title(
            f'Latency Breakdown by Layer Type\n'
            f'Total: {model_latency.total_latency_ms:.4f} ms',
            fontsize=12, fontweight='bold'
        )
        axes[0].legend(
            wedges, labels, title="Layer Type",
            loc="center left", bbox_to_anchor=(0, 0, -0.2, 1)
        )

        # Bar chart
        y_pos = range(len(labels))
        bars = axes[1].barh(y_pos, values, color=colors, edgecolor='white')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(labels)
        axes[1].set_xlabel('Latency (ms)')
        axes[1].set_title('Latency by Layer Type', fontsize=12, fontweight='bold')
        axes[1].invert_yaxis()

        # Add value labels
        for bar, val in zip(bars, values):
            axes[1].text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f} ms',
                va='center', fontsize=9
            )

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_roofline(
        self,
        roofline_result,
        filename: str = "roofline.png"
    ):
        """
        Plot roofline model with operation points.

        Args:
            roofline_result: RooflineResult object
            filename: Output filename
        """
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            print("WARNING: matplotlib/numpy not available, skipping plot")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        peak_gflops = roofline_result.peak_compute_gflops
        peak_bw = roofline_result.peak_bandwidth_gbps
        ridge = roofline_result.ridge_point

        # Generate roofline
        ai_range = np.logspace(-2, 3, 500)
        memory_roof = ai_range * peak_bw
        compute_roof = np.full_like(ai_range, peak_gflops)
        roofline = np.minimum(memory_roof, compute_roof)

        # Plot roofline
        ax.loglog(ai_range, roofline, 'k-', linewidth=2.5, label='DRAM Roofline')

        # Plot cache rooflines
        cache_styles = {'L1D': '--', 'L2': '-.', 'L3': ':'}
        cache_colors = {'L1D': '#66BB6A', 'L2': '#42A5F5', 'L3': '#AB47BC'}

        for level, bw in roofline_result.cache_rooflines.items():
            cache_roof = np.minimum(ai_range * bw, peak_gflops)
            ax.loglog(
                ai_range, cache_roof,
                linestyle=cache_styles.get(level, '--'),
                color=cache_colors.get(level, 'gray'),
                linewidth=1.5, alpha=0.7,
                label=f'{level} ({bw:.0f} GB/s)'
            )

        # Plot ridge point
        ax.axvline(
            x=ridge, color='red', linestyle=':', alpha=0.5,
            label=f'Ridge Point ({ridge:.1f} F/B)'
        )

        # Plot operation points
        for point in roofline_result.points:
            color = self.layer_colors.get(
                point.name.split('.')[-1].upper(),
                self.colors["other"]
            )

            # Determine marker based on layer type
            if "attention" in point.name.lower():
                marker = 's'
                color = self.colors["attention"]
            elif "mlp" in point.name.lower():
                marker = '^'
                color = self.colors["mlp"]
            elif "layernorm" in point.name.lower():
                marker = 'D'
                color = self.colors["layernorm"]
            elif "embedding" in point.name.lower():
                marker = 'p'
                color = self.colors["info"]
            elif "residual" in point.name.lower():
                marker = 'v'
                color = self.colors["other"]
            else:
                marker = 'o'
                color = self.colors["primary"]

            ax.plot(
                point.arithmetic_intensity,
                point.achieved_gflops,
                marker=marker, markersize=8,
                color=color, alpha=0.8,
                markeredgecolor='black', markeredgewidth=0.5
            )

        # Labels and formatting
        ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=13)
        ax.set_ylabel('Performance (GFLOP/s)', fontsize=13)
        ax.set_title(
            f'Roofline Model: {roofline_result.soc_name}\n'
            f'Peak: {peak_gflops:.1f} GFLOP/s | BW: {peak_bw:.1f} GB/s | '
            f'{roofline_result.dtype.upper()}',
            fontsize=13, fontweight='bold'
        )

        # Custom legend for layer types
        legend_elements = [
            plt.Line2D([0], [0], color='k', linewidth=2.5, label='DRAM Roofline'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=self.colors["attention"],
                       markersize=10, label='Attention'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=self.colors["mlp"],
                       markersize=10, label='MLP'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=self.colors["layernorm"],
                       markersize=10, label='LayerNorm'),
            plt.Line2D([0], [0], marker='v', color='w', markerfacecolor=self.colors["other"],
                       markersize=10, label='Other'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(0.01, 1000)
        ax.set_ylim(0.01, peak_gflops * 2)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_layer_timeline(
        self,
        model_latency,
        filename: str = "layer_timeline.png"
    ):
        """
        Plot layer-by-layer execution timeline.
        Shows compute vs memory bound per layer.
        """
        if not HAS_MATPLOTLIB:
            return

        layers = model_latency.layer_breakdown
        n_layers = len(layers)

        if n_layers == 0:
            return

        fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])

        # Top: Stacked bar chart (compute vs memory cycles)
        x = range(n_layers)
        compute = [l.compute_cycles for l in layers]
        memory = [l.memory_cycles for l in layers]
        names = [l.layer_name.split('.')[-1][:12] for l in layers]

        axes[0].bar(x, compute, label='Compute Cycles',
                    color=self.colors["compute"], alpha=0.8)
        axes[0].bar(x, memory, bottom=compute, label='Memory Cycles',
                    color=self.colors["memory"], alpha=0.8)

        axes[0].set_ylabel('Cycles', fontsize=11)
        axes[0].set_title(
            f'Layer Execution Timeline: {model_latency.model_name}',
            fontsize=13, fontweight='bold'
        )
        axes[0].legend(fontsize=10)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=7)

        # Bottom: Compute/Memory bound indicator
        bound_colors = [
            self.colors["compute"] if l.is_compute_bound
            else self.colors["memory"]
            for l in layers
        ]
        axes[1].bar(x, [1] * n_layers, color=bound_colors, alpha=0.8)
        axes[1].set_yticks([])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        axes[1].set_ylabel('Bound Type', fontsize=11)

        # Legend
        comp_patch = mpatches.Patch(
            color=self.colors["compute"], label='Compute Bound'
        )
        mem_patch = mpatches.Patch(
            color=self.colors["memory"], label='Memory Bound'
        )
        axes[1].legend(handles=[comp_patch, mem_patch], fontsize=10)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_throughput_scaling(
        self,
        sweep_results: list,
        sweep_param: str = "batch_size",
        filename: str = "throughput_scaling.png"
    ):
        """
        Plot throughput scaling across configurations.
        """
        if not HAS_MATPLOTLIB:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        if sweep_param == "batch_size":
            x_vals = [r.batch_size for r in sweep_results]
            x_label = "Batch Size"
        elif sweep_param == "seq_len":
            x_vals = [r.seq_len for r in sweep_results]
            x_label = "Sequence Length"
        else:
            x_vals = list(range(len(sweep_results)))
            x_label = sweep_param

        # Plot 1: Latency
        latencies = [r.latency_ms for r in sweep_results]
        axes[0].plot(x_vals, latencies, 'o-', color=self.colors["primary"],
                     linewidth=2, markersize=8)
        axes[0].set_xlabel(x_label, fontsize=11)
        axes[0].set_ylabel('Latency (ms)', fontsize=11)
        axes[0].set_title('Inference Latency', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Throughput
        throughput = [r.tokens_per_second for r in sweep_results]
        axes[1].plot(x_vals, throughput, 's-', color=self.colors["success"],
                     linewidth=2, markersize=8)
        axes[1].set_xlabel(x_label, fontsize=11)
        axes[1].set_ylabel('Tokens/second', fontsize=11)
        axes[1].set_title('Throughput', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: HW Utilization
        utilization = [r.hw_utilization * 100 for r in sweep_results]
        axes[2].plot(x_vals, utilization, 'D-', color=self.colors["secondary"],
                     linewidth=2, markersize=8)
        axes[2].set_xlabel(x_label, fontsize=11)
        axes[2].set_ylabel('Utilization (%)', fontsize=11)
        axes[2].set_title('Hardware Utilization', fontsize=12, fontweight='bold')
        axes[2].set_ylim(0, 100)
        axes[2].grid(True, alpha=0.3)

        plt.suptitle(
            f'Performance Scaling ({sweep_param})',
            fontsize=14, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_soc_comparison(
        self,
        comparison_data: List[Dict],
        filename: str = "soc_comparison.png"
    ):
        """
        Plot comparison across different SoC configurations.

        Args:
            comparison_data: List of dicts with keys:
                'soc_name', 'latency_ms', 'gflops', 'power_w', 'efficiency'
        """
        if not HAS_MATPLOTLIB:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        soc_names = [d['soc_name'] for d in comparison_data]
        x_pos = range(len(soc_names))

        bar_colors = [self.colors["primary"], self.colors["success"],
                      self.colors["secondary"], self.colors["attention"],
                      self.colors["danger"]][:len(soc_names)]

        # Latency comparison
        latencies = [d['latency_ms'] for d in comparison_data]
        axes[0, 0].bar(x_pos, latencies, color=bar_colors, edgecolor='white')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(soc_names, rotation=15)
        axes[0, 0].set_ylabel('Latency (ms)')
        axes[0, 0].set_title('Inference Latency', fontweight='bold')
        for i, v in enumerate(latencies):
            axes[0, 0].text(i, v + max(latencies)*0.02, f'{v:.2f}',
                           ha='center', fontsize=9)

        # GFLOP/s comparison
        gflops = [d['gflops'] for d in comparison_data]
        axes[0, 1].bar(x_pos, gflops, color=bar_colors, edgecolor='white')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(soc_names, rotation=15)
        axes[0, 1].set_ylabel('GFLOP/s')
        axes[0, 1].set_title('Achieved Performance', fontweight='bold')
        for i, v in enumerate(gflops):
            axes[0, 1].text(i, v + max(gflops)*0.02, f'{v:.1f}',
                           ha='center', fontsize=9)

        # Power comparison
        power = [d['power_w'] for d in comparison_data]
        axes[1, 0].bar(x_pos, power, color=bar_colors, edgecolor='white')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(soc_names, rotation=15)
        axes[1, 0].set_ylabel('Power (W)')
        axes[1, 0].set_title('Power Consumption', fontweight='bold')
        for i, v in enumerate(power):
            axes[1, 0].text(i, v + max(power)*0.02, f'{v:.2f}',
                           ha='center', fontsize=9)

        # Efficiency (GFLOP/s per Watt)
        efficiency = [d['efficiency'] for d in comparison_data]
        axes[1, 1].bar(x_pos, efficiency, color=bar_colors, edgecolor='white')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(soc_names, rotation=15)
        axes[1, 1].set_ylabel('GFLOP/s per Watt')
        axes[1, 1].set_title('Energy Efficiency', fontweight='bold')
        for i, v in enumerate(efficiency):
            axes[1, 1].text(i, v + max(efficiency)*0.02, f'{v:.1f}',
                           ha='center', fontsize=9)

        plt.suptitle('SoC Configuration Comparison',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_power_breakdown(
        self,
        energy_result,
        filename: str = "power_breakdown.png"
    ):
        """Plot power consumption breakdown."""
        if not HAS_MATPLOTLIB:
            return

        if energy_result.power_breakdown is None:
            return

        pb = energy_result.power_breakdown

        labels = ['Core Dynamic', 'Core Leakage', 'Vector Unit',
                  'Cache', 'DRAM', 'NoC']
        values = [pb.core_dynamic_w, pb.core_leakage_w, pb.vector_unit_w,
                  pb.cache_power_w, pb.dram_power_w, pb.noc_power_w]

        # Filter out zero values
        filtered = [(l, v) for l, v in zip(labels, values) if v > 0.001]
        if not filtered:
            return

        labels, values = zip(*filtered)

        colors = [self.colors["compute"], self.colors["dark"],
                  self.colors["attention"], self.colors["success"],
                  self.colors["memory"], self.colors["info"]][:len(labels)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart
        wedges, texts, autotexts = ax1.pie(
            values, labels=None, autopct='%1.1f%%',
            colors=colors, startangle=90
        )
        ax1.set_title(
            f'Power Breakdown\nTotal: {pb.total_power_w:.3f} W',
            fontsize=12, fontweight='bold'
        )
        ax1.legend(wedges, labels, loc="center left",
                   bbox_to_anchor=(-0.3, 0.5))

        # Bar chart with values
        y_pos = range(len(labels))
        bars = ax2.barh(y_pos, [v * 1000 for v in values],
                        color=colors, edgecolor='white')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Power (mW)')
        ax2.set_title('Power by Component', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()

        for bar, val in zip(bars, values):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     f'{val*1000:.1f} mW', va='center', fontsize=9)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_dtype_comparison(
        self,
        dtype_results: List,
        filename: str = "dtype_comparison.png"
    ):
        """
        Plot comparison across different data types
        (FP32, FP16, INT8).
        """
        if not HAS_MATPLOTLIB:
            return

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        dtypes = [r.dtype for r in dtype_results]
        x_pos = range(len(dtypes))

        dtype_colors = {
            "fp32": self.colors["primary"],
            "fp16": self.colors["success"],
            "int8": self.colors["secondary"],
            "bf16": self.colors["attention"],
        }
        colors = [dtype_colors.get(d, self.colors["other"]) for d in dtypes]

        # Latency
        axes[0].bar(x_pos, [r.latency_ms for r in dtype_results],
                    color=colors, edgecolor='white')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([d.upper() for d in dtypes])
        axes[0].set_ylabel('Latency (ms)')
        axes[0].set_title('Latency by Data Type', fontweight='bold')

        # Throughput
        axes[1].bar(x_pos, [r.tokens_per_second for r in dtype_results],
                    color=colors, edgecolor='white')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([d.upper() for d in dtypes])
        axes[1].set_ylabel('Tokens/second')
        axes[1].set_title('Throughput by Data Type', fontweight='bold')

        # Memory
        axes[2].bar(x_pos, [r.memory_usage_mb for r in dtype_results],
                    color=colors, edgecolor='white')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels([d.upper() for d in dtypes])
        axes[2].set_ylabel('Memory (MB)')
        axes[2].set_title('Memory Usage by Data Type', fontweight='bold')

        plt.suptitle('Data Type Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def generate_text_report(
        self,
        model_latency,
        energy_result=None,
        roofline_result=None,
        filename: str = "report.txt"
    ):
        """Generate a comprehensive text report."""

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("TRANSFORMER INFERENCE PERFORMANCE REPORT\n")
            f.write("RISC-V SoC Performance Modeling Framework\n")
            f.write("=" * 70 + "\n\n")

            # Model info
            f.write("MODEL CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Model        : {model_latency.model_name}\n")
            f.write(f"  Batch Size   : {model_latency.batch_size}\n")
            f.write(f"  Seq Length   : {model_latency.seq_len}\n")
            f.write(f"  Data Type    : {model_latency.dtype}\n\n")

            # Latency results
            f.write("LATENCY RESULTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Total Latency      : {model_latency.total_latency_ms:.4f} ms\n")
            f.write(f"  Total Cycles       : {model_latency.total_cycles:,}\n")
            f.write(f"  Throughput (tok/s)  : {model_latency.throughput_tokens_per_sec:.1f}\n")
            f.write(f"  Throughput (samp/s) : {model_latency.throughput_samples_per_sec:.1f}\n")
            f.write(f"  Total FLOPs        : {model_latency.total_flops:,}\n")
            f.write(f"  Achieved GFLOP/s   : {model_latency.achieved_gflops:.2f}\n")
            f.write(f"  Peak GFLOP/s       : {model_latency.peak_gflops:.2f}\n")
            f.write(f"  HW Utilization     : {model_latency.hardware_utilization:.1%}\n\n")

            # Layer breakdown
            f.write("LAYER BREAKDOWN (Top 15)\n")
            f.write("-" * 70 + "\n")
            f.write(f"  {'Layer':<30} {'Type':<12} {'ms':>10} {'Bound':<8}\n")
            f.write(f"  {'─'*30} {'─'*12} {'─'*10} {'─'*8}\n")

            sorted_layers = sorted(
                model_latency.layer_breakdown,
                key=lambda x: x.total_cycles, reverse=True
            )
            for layer in sorted_layers[:15]:
                bound = "COMP" if layer.is_compute_bound else "MEM"
                f.write(
                    f"  {layer.layer_name:<30} "
                    f"{layer.layer_type:<12} "
                    f"{layer.latency_ms:>10.4f} "
                    f"{bound:<8}\n"
                )

            # Energy results
            if energy_result:
                f.write(f"\nENERGY RESULTS\n")
                f.write("-" * 70 + "\n")
                f.write(f"  Total Energy       : {energy_result.total_energy_mj:.4f} mJ\n")
                f.write(f"  Energy/Token       : {energy_result.energy_per_token_uj:.2f} µJ\n")
                f.write(f"  Average Power      : {energy_result.average_power_w:.3f} W\n")
                f.write(f"  Efficiency         : {energy_result.gflops_per_watt:.2f} GFLOPS/W\n")

            # Roofline results
            if roofline_result:
                f.write(f"\nROOFLINE ANALYSIS\n")
                f.write("-" * 70 + "\n")
                summary = roofline_result.summary()
                for key, val in summary.items():
                    f.write(f"  {key:<25}: {val}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")

        print(f"  Saved: {filepath}")

    def generate_full_dashboard(
        self,
        model_latency,
        energy_result=None,
        roofline_result=None,
        throughput_sweep=None,
        soc_comparison=None,
        dtype_results=None,
        prefix: str = ""
    ):
        """
        Generate all visualizations at once.
        """
        print(f"\nGenerating Dashboard in: {self.output_dir}/")
        print("-" * 50)

        # 1. Latency breakdown
        self.plot_latency_breakdown(
            model_latency,
            filename=f"{prefix}latency_breakdown.png"
        )

        # 2. Layer timeline
        self.plot_layer_timeline(
            model_latency,
            filename=f"{prefix}layer_timeline.png"
        )

        # 3. Roofline
        if roofline_result:
            self.plot_roofline(
                roofline_result,
                filename=f"{prefix}roofline.png"
            )

        # 4. Power breakdown
        if energy_result:
            self.plot_power_breakdown(
                energy_result,
                filename=f"{prefix}power_breakdown.png"
            )

        # 5. Throughput scaling
        if throughput_sweep:
            self.plot_throughput_scaling(
                throughput_sweep,
                filename=f"{prefix}throughput_scaling.png"
            )

        # 6. SoC comparison
        if soc_comparison:
            self.plot_soc_comparison(
                soc_comparison,
                filename=f"{prefix}soc_comparison.png"
            )

        # 7. Data type comparison
        if dtype_results:
            self.plot_dtype_comparison(
                dtype_results,
                filename=f"{prefix}dtype_comparison.png"
            )

        # 8. Text report
        self.generate_text_report(
            model_latency, energy_result, roofline_result,
            filename=f"{prefix}report.txt"
        )

        print("-" * 50)
        print(f"Dashboard complete! Files saved in: {self.output_dir}/\n")


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("Dashboard module loaded successfully")
    print(f"  matplotlib available: {HAS_MATPLOTLIB}")
    print(f"  numpy available: {HAS_NUMPY}")