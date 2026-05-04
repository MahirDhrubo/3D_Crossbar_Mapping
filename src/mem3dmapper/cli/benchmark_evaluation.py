import math
from dataclasses import dataclass

MULTIPLIER_BITS = 32
PRODUCT_BITS = 2 * MULTIPLIER_BITS   # 64 for 32x32 multiplication


class DeviceParams:
    # Energy (fJ)
    E_SET = 23.8
    E_RESET = 0.32
    E_GATE = 0.29

    # Latency (ns)
    T_GATE = 1.1
    T_INIT = 1.0


class GateParams:
    NOR_COUNT: int = 0
    AND_COUNT: int = 0
    NOT_COUNT: int = 0
    COPY_COUNT: int = 0
    TOTAL_AREA: int = 0

    def __init__(self, nor_count, not_count, and_count, copy_count, total_area):
        self.NOR_COUNT = nor_count
        self.NOT_COUNT = not_count
        self.AND_COUNT = and_count
        self.COPY_COUNT = copy_count
        self.TOTAL_AREA = total_area


class MultiplierNorConst(GateParams):
    def __init__(self):
        super().__init__(nor_count=9183, not_count=3429, and_count=0, copy_count=11003, total_area=47 * 512)


class MultiplierConst(GateParams):
    def __init__(self):
        super().__init__(nor_count=5119, not_count=934, and_count=3265, copy_count=7807, total_area=8 * 512)


class AdderNorConst(GateParams):
    def __init__(self):
        # This is your baseline adder, assumed to correspond to 64-bit
        super().__init__(nor_count=624, not_count=201, and_count=0, copy_count=232, total_area=3 * 512)


class AdderConst(GateParams):
    def __init__(self):
        # This is your baseline adder, assumed to correspond to 64-bit
        super().__init__(nor_count=376, not_count=16, and_count=136, copy_count=350, total_area=3 * 128)


class EvaluationCost:
    e_nor: float
    e_and: float
    e_not: float
    e_copy: float
    e_inter: float

    t_gate: float
    t_inter: float

    total_energy: float = 0.0
    total_latency: float = 0.0
    total_area: float = 0.0

    gateParams: GateParams = None
    is_nor_only: bool = False

    def __init__(self, K: int, device_params: DeviceParams, gateCount: GateParams):
        self.e_nor = device_params.E_SET + device_params.E_GATE
        self.e_not = device_params.E_SET + device_params.E_GATE

        self.e_and = device_params.E_RESET + device_params.E_GATE
        self.e_copy = device_params.E_RESET + device_params.E_GATE

        # baseline inter-copy cost for K-bit transfer
        self.e_inter = K * self.e_copy

        self.t_gate = device_params.T_INIT + device_params.T_GATE
        self.t_inter = K * self.t_gate

        self.gateParams = gateCount
        self.is_nor_only = gateCount.AND_COUNT == 0

    def _calculate_energy(self) -> None:
        self.total_energy = (
            self.gateParams.NOR_COUNT * self.e_nor
            + self.gateParams.AND_COUNT * self.e_and
            + self.gateParams.NOT_COUNT * self.e_not
        )

        if self.is_nor_only:
            self.total_energy += self.gateParams.COPY_COUNT * self.e_inter
        else:
            self.total_energy += self.gateParams.COPY_COUNT * self.e_copy

    def _calculate_latency(self) -> None:
        self.total_latency = self.t_gate * (
            self.gateParams.NOR_COUNT
            + self.gateParams.AND_COUNT
            + self.gateParams.NOT_COUNT
        )

        if self.is_nor_only:
            self.total_latency += self.gateParams.COPY_COUNT * self.t_inter
        else:
            self.total_latency += self.gateParams.COPY_COUNT * self.t_gate

    def calculate(self) -> None:
        self._calculate_energy()
        self._calculate_latency()
        self.total_area = self.gateParams.TOTAL_AREA

    def get_results(self):
        return {
            "energy": self.total_energy,
            "latency": self.total_latency,
            "area": self.total_area
        }


# -----------------------------
# Helper functions
# -----------------------------

def ceil_log2(x: int) -> int:
    if x <= 1:
        return 0
    return math.ceil(math.log2(x))


def running_sum_width(num_products_accumulated: int, input_bits: int = MULTIPLIER_BITS) -> int:
    """
    Width of partial sum after accumulating `num_products_accumulated` products.
    Product width = 2 * input_bits.
    Sum width grows by ceil(log2(k)).
    """
    product_bits = 2 * input_bits
    return product_bits + ceil_log2(num_products_accumulated)


def scaled_adder_energy(base_adder_cost: EvaluationCost, width_bits: int, base_bits: int = PRODUCT_BITS) -> float:
    """
    Scale adder energy linearly with width.
    Baseline adder_cost corresponds to base_bits-wide addition.
    """
    return base_adder_cost.total_energy * (width_bits / base_bits)


def scaled_adder_latency(base_adder_cost: EvaluationCost, width_bits: int, base_bits: int = PRODUCT_BITS) -> float:
    """
    Scale adder latency linearly with width.
    """
    return base_adder_cost.total_latency * (width_bits / base_bits)


def scaled_adder_area(base_adder_cost: EvaluationCost, width_bits: int, base_bits: int = PRODUCT_BITS) -> float:
    """
    Scale adder area linearly with width.
    """
    return base_adder_cost.total_area * (width_bits / base_bits)


def inter_move_energy_for_bits(adder_cost: EvaluationCost, width_bits: int) -> float:
    """
    Inter-crossbar movement energy for transferring `width_bits`.
    Uses per-bit copy cost.
    """
    return width_bits * adder_cost.e_copy


def inter_move_latency_for_bits(adder_cost: EvaluationCost, width_bits: int) -> float:
    """
    Inter-crossbar movement latency for transferring `width_bits`.
    Uses per-bit gate latency.
    """
    return width_bits * adder_cost.t_gate


# -----------------------------
# Revised MVM models
# -----------------------------

def evaluate_MVM_cost_3D(dimension: int, multiplier_cost: EvaluationCost, adder_cost: EvaluationCost):
    if multiplier_cost.is_nor_only != adder_cost.is_nor_only and multiplier_cost.is_nor_only is True:
        raise ValueError("Incompatible gate configurations")

    if dimension <= 0:
        return 0.0, 0.0, 0.0

    multiplier_cost.calculate()
    adder_cost.calculate()

    # Multiplications: still one per vector element
    total_energy = dimension * multiplier_cost.total_energy
    total_latency = dimension * multiplier_cost.total_latency
    total_area = dimension * multiplier_cost.total_area

    # Additions: width grows with the accumulated partial sum
    # Sequential accumulation model:
    # step k means summing k-th product into running sum
    for k in range(2, dimension + 1):
        w = running_sum_width(k, MULTIPLIER_BITS)

        total_energy += scaled_adder_energy(adder_cost, w)
        total_latency += scaled_adder_latency(adder_cost, w)
        total_area += scaled_adder_area(adder_cost, w)

    # 3D movement model:
    # your original model used:
    # if dimension <= 11: k = 0 else k = ceil((dimension - 11)/10)
    # and movement count = dimension + k + 1
    # keep that structure, but movement width now grows with partial-sum width
    if dimension <= 11:
        spill_groups = 0
    else:
        spill_groups = (dimension - 11 + 9) // 10

    move_count = dimension + spill_groups + 1

    # Use final accumulator width for transferred partial sums
    final_width = running_sum_width(dimension, MULTIPLIER_BITS)

    total_energy += move_count * inter_move_energy_for_bits(adder_cost, final_width)
    total_latency += move_count * inter_move_latency_for_bits(adder_cost, final_width)

    return total_energy, total_latency, total_area


def evaluate_MVM_cost_2D(dimension: int, multiplier_cost: EvaluationCost, adder_cost: EvaluationCost):
    if multiplier_cost.is_nor_only != adder_cost.is_nor_only and multiplier_cost.is_nor_only is False:
        raise ValueError("Incompatible gate configurations")

    if dimension <= 0:
        return 0.0, 0.0, 0.0

    multiplier_cost.calculate()
    adder_cost.calculate()

    # Multiplications
    total_energy = dimension * multiplier_cost.total_energy
    total_latency = dimension * multiplier_cost.total_latency
    total_area = dimension * multiplier_cost.total_area

    # Additions with growing width
    for k in range(2, dimension + 1):
        w = running_sum_width(k, MULTIPLIER_BITS)

        total_energy += scaled_adder_energy(adder_cost, w)
        total_latency += scaled_adder_latency(adder_cost, w)
        total_area += scaled_adder_area(adder_cost, w)

    # 2D movement model:
    # keep your "2 * dimension" transfer structure,
    # but move width-sensitive partial sums instead of fixed 64-bit values
    final_width = running_sum_width(dimension, MULTIPLIER_BITS)
    move_count = 2 * dimension

    total_energy += move_count * inter_move_energy_for_bits(adder_cost, final_width)
    total_latency += move_count * inter_move_latency_for_bits(adder_cost, final_width)

    return total_energy, total_latency, total_area


# ...existing code...
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv

def plot_bar_chart(norm_energy_3d, norm_latency_3d, norm_area_3d, norm_energy_2d, norm_latency_2d, norm_area_2d):
    x = np.arange(len(norm_energy_3d))
    width = 0.35

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)

    # hatch patterns and colors (choose contrasting patterns)
    hatch_3d = "///"
    hatch_2d = "\\\\\\"

    # font settings
    title_fs = 16
    label_fs = 13
    xtick_fs = 12
    ytick_fs = 12
    legend_fs = 11
    xtick_rotation = 45
    xtick_ha = 'right'
    benchmark_label_weight = 'bold'
    title_weight = 'bold'
    label_weight = 'bold'

    ax = axes[0]
    b1 = ax.bar(x - width/2, norm_energy_3d, width, label='Our Framework',
                color='#76f597', edgecolor='black', linewidth=0.5, hatch=hatch_3d)
    b2 = ax.bar(x + width/2, norm_energy_2d, width, label='AUTO',
                color='#ff7f0e', edgecolor='black', linewidth=0.5, hatch=hatch_2d)
    ax.set_ylabel('Normalized Energy', fontsize=label_fs, fontweight=label_weight)
    ax.set_title('Normalized Energy per Benchmark', fontsize=title_fs, fontweight=title_weight)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=xtick_rotation, ha=xtick_ha, fontsize=xtick_fs, fontweight=benchmark_label_weight)
    ax.tick_params(axis='y', labelsize=ytick_fs)
    ax.legend(fontsize=legend_fs)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    ax2 = axes[1]
    b3 = ax2.bar(x - width/2, norm_latency_3d, width, label='Our Framework',
                 color='#76f597', edgecolor='black', linewidth=0.5, hatch=hatch_3d)
    b4 = ax2.bar(x + width/2, norm_latency_2d, width, label='AUTO',
                 color='#ff7f0e', edgecolor='black', linewidth=0.5, hatch=hatch_2d)
    ax2.set_ylabel('Normalized Latency', fontsize=label_fs, fontweight=label_weight)
    ax2.set_title('Normalized Latency per Benchmark', fontsize=title_fs, fontweight=title_weight)
    ax2.set_xticks(x)
    ax2.set_xticklabels(benchmarks, rotation=xtick_rotation, ha=xtick_ha, fontsize=xtick_fs, fontweight=benchmark_label_weight)
    ax2.tick_params(axis='y', labelsize=ytick_fs)
    ax2.legend(fontsize=legend_fs)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    ax3 = axes[2]
    b5 = ax3.bar(x - width/2, norm_area_3d, width, label='Our Framework',
                 color='#76f597', edgecolor='black', linewidth=0.5, hatch=hatch_3d)
    b6 = ax3.bar(x + width/2, norm_area_2d, width, label='AUTO',
                 color='#ff7f0e', edgecolor='black', linewidth=0.5, hatch=hatch_2d)
    ax3.set_ylabel('Normalized Area', fontsize=label_fs, fontweight=label_weight)
    ax3.set_title('Normalized Area per Benchmark', fontsize=title_fs, fontweight=title_weight)
    ax3.set_xticks(x)
    ax3.set_xticklabels(benchmarks, rotation=xtick_rotation, ha=xtick_ha, fontsize=xtick_fs, fontweight=benchmark_label_weight)
    ax3.tick_params(axis='y', labelsize=ytick_fs)
    ax3.legend(fontsize=legend_fs)

    # increase tick label weight for y-axis as well
    for a in axes:
        for label in a.get_yticklabels():
            label.set_fontsize(ytick_fs)
            label.set_fontweight('normal')

    outdir = Path("data/plots")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath_svg = outdir / "mvm_costs_normalized.svg"
    outpath_pdf = outdir / "mvm_costs_normalized.pdf"
    # save as vector formats
    fig.savefig(outpath_svg, dpi=300, format="svg")
    fig.savefig(outpath_pdf, dpi=300, format="pdf")
    print(f"Saved normalized bar chart to {outpath_svg} and {outpath_pdf}")

    csv_path = outdir / "mvm_costs_normalized.csv"
    with csv_path.open("w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "benchmark", "dimension",
            "energy_3d", "energy_2d", "norm_energy_3d", "norm_energy_2d",
            "latency_3d", "latency_2d", "norm_latency_3d", "norm_latency_2d",
            "area_3d", "area_2d", "norm_area_3d", "norm_area_2d"
        ])
        for i, name in enumerate(benchmarks):
            writer.writerow([
                name,
                dimensions[i],
                energies_3d[i], energies_2d[i], float(norm_energy_3d[i]), float(norm_energy_2d[i]),
                latencies_3d[i], latencies_2d[i], float(norm_latency_3d[i]), float(norm_latency_2d[i]),
                areas_3d[i], areas_2d[i], float(norm_area_3d[i]), float(norm_area_2d[i])
            ])
    print(f"Wrote CSV summary to {csv_path}")

    # also print a small table to stdout
    for i, name in enumerate(benchmarks):
        print(f"{name}: energy 3D={energies_3d[i]:.3e}, 2D={energies_2d[i]:.3e} | latency 3D={latencies_3d[i]:.3e}, 2D={latencies_2d[i]:.3e}")


def plot_normalized_comparison_and_save(
    norm_energy_3d, norm_latency_3d, norm_area_3d,
    norm_energy_2d, norm_latency_2d, norm_area_2d,
    benchmarks,
    dimensions,
    energies_3d=None, energies_2d=None,
    latencies_3d=None, latencies_2d=None,
    areas_3d=None, areas_2d=None,
    outdir="data/plots"
):
    """
    Saves multiple comparison plots as PDF and writes CSV summaries.

    Required inputs:
        norm_energy_3d, norm_latency_3d, norm_area_3d
        norm_energy_2d, norm_latency_2d, norm_area_2d
        benchmarks
        dimensions

    Optional raw values:
        energies_3d, energies_2d,
        latencies_3d, latencies_2d,
        areas_3d, areas_2d
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # convert to numpy
    norm_energy_3d = np.asarray(norm_energy_3d, dtype=float)
    norm_latency_3d = np.asarray(norm_latency_3d, dtype=float)
    norm_area_3d = np.asarray(norm_area_3d, dtype=float)

    norm_energy_2d = np.asarray(norm_energy_2d, dtype=float)
    norm_latency_2d = np.asarray(norm_latency_2d, dtype=float)
    norm_area_2d = np.asarray(norm_area_2d, dtype=float)

    x = np.arange(len(benchmarks))

    # font settings
    title_fs = 16
    label_fs = 13
    xtick_fs = 11
    ytick_fs = 11
    legend_fs = 11
    xtick_rotation = 45
    xtick_ha = "right"

    def _save_metric_csv(metric_name, v3d, v2d):
        csv_path = outdir / f"{metric_name.lower()}_normalized_comparison.csv"
        percent_diff = (v3d - v2d) / v2d * 100.0
        abs_diff = v3d - v2d

        with csv_path.open("w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow([
                "benchmark",
                "dimension",
                f"{metric_name.lower()}_3d",
                f"{metric_name.lower()}_2d",
                "absolute_difference_3d_minus_2d",
                "percent_difference_3d_vs_2d"
            ])
            for i, b in enumerate(benchmarks):
                writer.writerow([
                    b,
                    dimensions[i],
                    float(v3d[i]),
                    float(v2d[i]),
                    float(abs_diff[i]),
                    float(percent_diff[i])
                ])
        print(f"Wrote {metric_name} CSV to {csv_path}")

    def _plot_zoomed_line(metric_name, v3d, v2d):
        fig, ax = plt.subplots(figsize=(14, 5))
        combined = np.concatenate([v3d, v2d])
        spread = combined.max() - combined.min()
        margin = 0.1 * spread if spread > 0 else 1e-6

        ax.plot(x, v3d, marker='o', label='Our Framework')
        ax.plot(x, v2d, marker='s', label='AUTO')

        ax.set_ylim(combined.min() - margin, combined.max() + margin)
        ax.set_xlabel("Benchmark", fontsize=label_fs, fontweight='bold')
        ax.set_ylabel(f"Normalized {metric_name}", fontsize=label_fs, fontweight='bold')
        ax.set_title(f"{metric_name}: Zoomed Comparison", fontsize=title_fs, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=xtick_rotation, ha=xtick_ha, fontsize=xtick_fs, fontweight='bold')
        ax.tick_params(axis='y', labelsize=ytick_fs)
        ax.legend(fontsize=legend_fs)
        ax.grid(True, linestyle='--', alpha=0.4)

        outpath = outdir / f"{metric_name.lower()}_zoomed_line.pdf"
        fig.savefig(outpath, dpi=300, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {metric_name} zoomed line plot to {outpath}")

    def _plot_percent_diff(metric_name, v3d, v2d):
        fig, ax = plt.subplots(figsize=(14, 5))
        percent_diff = (v3d - v2d) / v2d * 100.0

        ax.plot(x, percent_diff, marker='o')
        ax.set_xlabel("Benchmark", fontsize=label_fs, fontweight='bold')
        ax.set_ylabel("Difference (%)", fontsize=label_fs, fontweight='bold')
        ax.set_title(f"{metric_name}: Percentage Difference (3D vs 2D)", fontsize=title_fs, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=xtick_rotation, ha=xtick_ha, fontsize=xtick_fs, fontweight='bold')
        ax.tick_params(axis='y', labelsize=ytick_fs)
        ax.grid(True, linestyle='--', alpha=0.4)

        outpath = outdir / f"{metric_name.lower()}_percent_difference.pdf"
        fig.savefig(outpath, dpi=300, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {metric_name} percent difference plot to {outpath}")

    def _plot_absolute_diff(metric_name, v3d, v2d):
        fig, ax = plt.subplots(figsize=(14, 5))
        abs_diff = v3d - v2d

        ax.plot(x, abs_diff, marker='o')
        ax.set_xlabel("Benchmark", fontsize=label_fs, fontweight='bold')
        ax.set_ylabel("Absolute Difference", fontsize=label_fs, fontweight='bold')
        ax.set_title(f"{metric_name}: Absolute Difference (3D - 2D)", fontsize=title_fs, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=xtick_rotation, ha=xtick_ha, fontsize=xtick_fs, fontweight='bold')
        ax.tick_params(axis='y', labelsize=ytick_fs)
        ax.grid(True, linestyle='--', alpha=0.4)

        outpath = outdir / f"{metric_name.lower()}_absolute_difference.pdf"
        fig.savefig(outpath, dpi=300, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {metric_name} absolute difference plot to {outpath}")

    def _plot_mean_range(metric_name, v3d, v2d):
        fig, ax = plt.subplots(figsize=(7, 5))

        mean3d = np.mean(v3d)
        mean2d = np.mean(v2d)

        err3d = [[mean3d - np.min(v3d)], [np.max(v3d) - mean3d]]
        err2d = [[mean2d - np.min(v2d)], [np.max(v2d) - mean2d]]

        ax.errorbar([0], [mean3d], yerr=err3d, fmt='o', label='Our Framework')
        ax.errorbar([1], [mean2d], yerr=err2d, fmt='o', label='AUTO')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Our Framework", "AUTO"], fontsize=xtick_fs, fontweight='bold')
        ax.set_ylabel(f"Normalized {metric_name}", fontsize=label_fs, fontweight='bold')
        ax.set_title(f"{metric_name}: Mean ± Range", fontsize=title_fs, fontweight='bold')
        ax.tick_params(axis='y', labelsize=ytick_fs)
        ax.legend(fontsize=legend_fs)
        ax.grid(True, linestyle='--', alpha=0.4)

        outpath = outdir / f"{metric_name.lower()}_mean_range.pdf"
        fig.savefig(outpath, dpi=300, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {metric_name} mean-range plot to {outpath}")

        print(f"\n--- {metric_name} Summary ---")
        print(f"Our Framework -> mean={mean3d:.8f}, min={np.min(v3d):.8f}, max={np.max(v3d):.8f}")
        print(f"AUTO          -> mean={mean2d:.8f}, min={np.min(v2d):.8f}, max={np.max(v2d):.8f}")

    # master csv with everything
    master_csv = outdir / "all_normalized_comparisons.csv"
    with master_csv.open("w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "benchmark", "dimension",
            "norm_energy_3d", "norm_energy_2d",
            "norm_latency_3d", "norm_latency_2d",
            "norm_area_3d", "norm_area_2d",
            "energy_percent_diff", "latency_percent_diff", "area_percent_diff",
            "energy_abs_diff", "latency_abs_diff", "area_abs_diff",
            "energy_3d_raw", "energy_2d_raw",
            "latency_3d_raw", "latency_2d_raw",
            "area_3d_raw", "area_2d_raw"
        ])

        for i, b in enumerate(benchmarks):
            writer.writerow([
                b,
                dimensions[i],
                float(norm_energy_3d[i]), float(norm_energy_2d[i]),
                float(norm_latency_3d[i]), float(norm_latency_2d[i]),
                float(norm_area_3d[i]), float(norm_area_2d[i]),
                float((norm_energy_3d[i] - norm_energy_2d[i]) / norm_energy_2d[i] * 100.0),
                float((norm_latency_3d[i] - norm_latency_2d[i]) / norm_latency_2d[i] * 100.0),
                float((norm_area_3d[i] - norm_area_2d[i]) / norm_area_2d[i] * 100.0),
                float(norm_energy_3d[i] - norm_energy_2d[i]),
                float(norm_latency_3d[i] - norm_latency_2d[i]),
                float(norm_area_3d[i] - norm_area_2d[i]),
                None if energies_3d is None else energies_3d[i],
                None if energies_2d is None else energies_2d[i],
                None if latencies_3d is None else latencies_3d[i],
                None if latencies_2d is None else latencies_2d[i],
                None if areas_3d is None else areas_3d[i],
                None if areas_2d is None else areas_2d[i],
            ])
    print(f"Wrote master CSV to {master_csv}")

    # per-metric csv
    _save_metric_csv("Energy", norm_energy_3d, norm_energy_2d)
    _save_metric_csv("Latency", norm_latency_3d, norm_latency_2d)
    _save_metric_csv("Area", norm_area_3d, norm_area_2d)

    # plots
    _plot_zoomed_line("Energy", norm_energy_3d, norm_energy_2d)
    _plot_percent_diff("Energy", norm_energy_3d, norm_energy_2d)
    _plot_absolute_diff("Energy", norm_energy_3d, norm_energy_2d)
    _plot_mean_range("Energy", norm_energy_3d, norm_energy_2d)

    _plot_zoomed_line("Latency", norm_latency_3d, norm_latency_2d)
    _plot_percent_diff("Latency", norm_latency_3d, norm_latency_2d)
    _plot_absolute_diff("Latency", norm_latency_3d, norm_latency_2d)
    _plot_mean_range("Latency", norm_latency_3d, norm_latency_2d)

    _plot_zoomed_line("Area", norm_area_3d, norm_area_2d)
    _plot_percent_diff("Area", norm_area_3d, norm_area_2d)
    _plot_absolute_diff("Area", norm_area_3d, norm_area_2d)
    _plot_mean_range("Area", norm_area_3d, norm_area_2d)

if __name__ == "__main__":
    benchmarks = ['eris1176', 'cegb2919', 'raefsky1', 'fxm3_6 ', 'Na5', 'Ex5', 'fp', 'ex40', 'benzene', 'bcsstk33', 'graham1', 'net25', 'bundle1', 'Si10H16', 'Goodwin_040']
    dimensions = [1176, 2919, 3242, 5026, 5832, 6545, 7548, 7740, 8219, 8738, 9035, 9520, 10581, 17077, 17922]
    # benchmarks = ['a', 'b']
    # dimensions = [20, 41]
    multiplier_cost = EvaluationCost(K=4, device_params=DeviceParams(), gateCount=MultiplierConst())
    adder_cost = EvaluationCost(K=4, device_params=DeviceParams(), gateCount=AdderConst())
    
    multiplier_nor_cost = EvaluationCost(K=4, device_params=DeviceParams(), gateCount=MultiplierNorConst())
    adder_nor_cost = EvaluationCost(K=4, device_params=DeviceParams(), gateCount=AdderNorConst())

    multiplier_cost.calculate()
    adder_cost.calculate()
    multiplier_nor_cost.calculate()
    adder_nor_cost.calculate()

    print(multiplier_cost.get_results())
    print(adder_cost.get_results())
    print(multiplier_nor_cost.get_results())
    print(adder_nor_cost.get_results())


    energies_3d = []
    latencies_3d = []
    energies_2d = []
    latencies_2d = []
    areas_3d = []
    areas_2d = []

    for dimension in dimensions:
        e3, l3, a3 = evaluate_MVM_cost_3D(dimension, multiplier_cost, adder_cost)
        e2, l2, a2 = evaluate_MVM_cost_2D(dimension, multiplier_nor_cost, adder_nor_cost)
        energies_3d.append(e3)
        latencies_3d.append(l3)
        areas_3d.append(a3)
        energies_2d.append(e2)
        latencies_2d.append(l2)
        areas_2d.append(a2)

    # normalize per-benchmark (per index) so max(3D,2D) -> 1.0
    n = len(dimensions)
    norm_energy_3d = np.zeros(n)
    norm_energy_2d = np.zeros(n)
    norm_latency_3d = np.zeros(n)
    norm_latency_2d = np.zeros(n)
    norm_area_3d = np.zeros(n)
    norm_area_2d = np.zeros(n)

    for i in range(n):
        max_e = max(energies_3d[i], energies_2d[i], 1.0)
        norm_energy_3d[i] = energies_3d[i] / max_e
        norm_energy_2d[i] = energies_2d[i] / max_e

        max_l = max(latencies_3d[i], latencies_2d[i], 1.0)
        norm_latency_3d[i] = latencies_3d[i] / max_l
        norm_latency_2d[i] = latencies_2d[i] / max_l

        max_a = max(areas_3d[i], areas_2d[i], 1.0)
        norm_area_3d[i] = areas_3d[i] / max_a
        norm_area_2d[i] = areas_2d[i] / max_a
    
    print(norm_latency_2d)
    print(norm_area_3d)

    # plotting

    plot_bar_chart(norm_energy_3d, norm_latency_3d, norm_area_3d, norm_energy_2d, norm_latency_2d, norm_area_2d)
    # plot_normalized_comparison_and_save(norm_energy_3d, norm_latency_3d, norm_area_3d, norm_energy_2d, norm_latency_2d, norm_area_2d, benchmarks=benchmarks, dimensions=dimensions, energies_3d=energies_3d, energies_2d=energies_2d, latencies_3d=latencies_3d, latencies_2d=latencies_2d, areas_3d=areas_3d, areas_2d=areas_2d)
