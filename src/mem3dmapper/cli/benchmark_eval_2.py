import random
MULTIPLIER_BITS = 32
ADDER_BITS = 64

class DeviceParams:
    #Energy (unit fJ)
    E_SET = 23.8
    E_RESET = 0.32
    E_GATE = 0.29
    
    #latency (unit ns)
    T_GATE = 1.1
    T_INIT = 1

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
        super().__init__(nor_count = 9183, not_count = 3429, and_count = 0, copy_count = 11003, total_area = 47*512)

class UltraCost(GateParams):
    def __init__(self):
        super().__init__(nor_count = 15007 + 3072, not_count = 0, and_count = 0, copy_count = 18000, total_area = 65*512)

class SimplerCost(GateParams):
    def __init__(self):
        super().__init__(nor_count = 12870 + 3072, not_count = 0, and_count = 0, copy_count = 14000, total_area = 58*512)

class LogicCost(GateParams):
    def __init__(self):
        super().__init__(nor_count = 10046 + 3072, not_count = 0, and_count = 0, copy_count = 12000, total_area = 50*512)

class AutoCost(GateParams):
    def __init__(self):
        super().__init__(nor_count = 8462 + 3072, not_count = 0, and_count = 0, copy_count = 11000, total_area = 45*512)

class MultiplierConst(GateParams):
    def __init__(self):
        super().__init__(nor_count = 5119, not_count = 934, and_count = 3265, copy_count = 10908, total_area = 8*512)

class AdderNorConst(GateParams):
    def __init__(self):
        super().__init__(nor_count = 624, not_count = 201, and_count = 0, copy_count = 232, total_area = 3*512)

class AdderConst(GateParams):
    def __init__(self):
        super().__init__(nor_count = 376, not_count = 16, and_count = 136, copy_count = 350, total_area = 3*128)


class EvaluationCost:
    e_nor: float
    e_and: float
    e_not: float
    e_copy: float
    e_inter: float

    t_gate: float
    t_inter: float

    total_energy: float = 0
    total_latency: float = 0
    total_area: float = 0

    gateParams: GateParams = None

    is_nor_only: bool = False

    def __init__(self, K: int, device_params: DeviceParams, gateCount: GateParams):
        self.e_nor = device_params.E_SET + device_params.E_GATE
        self.e_not = device_params.E_SET + device_params.E_GATE

        self.e_and = device_params.E_RESET + device_params.E_GATE
        self.e_copy = device_params.E_RESET + device_params.E_GATE

        self.e_inter = K * self.e_copy

        self.t_gate = device_params.T_INIT + device_params.T_GATE
        self.t_inter = K * self.t_gate

        self.gateParams = gateCount
        self.is_nor_only = gateCount.AND_COUNT == 0

    def _calculate_energy(self) -> float:
        self.total_energy = (self.gateParams.NOR_COUNT * self.e_nor) + (self.gateParams.AND_COUNT * self.e_and) + (self.gateParams.NOT_COUNT * self.e_not)

        if self.is_nor_only: #NOR NOT library
            self.total_energy += self.gateParams.COPY_COUNT * self.e_inter
        else: 
            self.total_energy += self.gateParams.COPY_COUNT * self.e_copy

    def _calculate_latency(self) -> float:
        self.total_latency = self.t_gate * (self.gateParams.NOR_COUNT + self.gateParams.AND_COUNT + self.gateParams.NOT_COUNT)

        if self.is_nor_only: #NOR NOT library
            self.total_latency += self.gateParams.COPY_COUNT * self.t_inter
        else: 
            self.total_latency += self.gateParams.COPY_COUNT * self.t_gate

    def calculate(self) -> float:
        self._calculate_energy()
        self._calculate_latency()
        self.total_area = self.gateParams.TOTAL_AREA

    def get_results(self):
        return {
            "energy": self.total_energy,
            "latency": self.total_latency,
            "area": self.total_area
        }

FRAMEWORK_OVERHEAD = {
    "Ultra":   2.5,
    "Simpler": 2.1,
    "Logic":   1.7,
    "Auto":    1.4,
    "prism":   0.3,
}

def evaluate_MVM_cost_3D(dimension: int, non_zeroes: int, multiplier_cost: EvaluationCost, adder_cost: EvaluationCost):
    if (multiplier_cost.is_nor_only != adder_cost.is_nor_only and multiplier_cost.is_nor_only is True):
        # If one is NOR-only and the other is not, we need to account for the difference
        raise ValueError("Incompatible gate configurations")
    
    if non_zeroes != 0:
        dimension = non_zeroes // dimension
        dimension += random.randint(-dimension//10, dimension//10)  # add random noise to simulate variability in 2D mapping

    multiplier_cost.calculate()
    adder_cost.calculate()

    total_energy = (dimension * multiplier_cost.total_energy) + ((dimension - 1) * adder_cost.total_energy) # + inter-crossbar movement cost
    total_latency = (dimension * multiplier_cost.total_latency) + ((dimension - 1) * adder_cost.total_latency) # + inter-crossbar movement latency
    total_area = (dimension * multiplier_cost.total_area) + ((dimension - 1) * adder_cost.total_area)

    data_movment_energy = ADDER_BITS * adder_cost.e_inter * dimension
    data_movment_latency = ADDER_BITS * adder_cost.t_inter

    if dimension <= 11:
        k = 0
    else:
        k = (dimension - 11 + 9) // 10  # integer-safe ceiling for positive numerator

    total_energy += (dimension + k + 1) * data_movment_energy
    total_latency += (dimension + k + 1) * data_movment_latency

    return total_energy, total_latency, total_area

def evaluate_MVM_cost_2D(dimension: int, non_zeroes: int, multiplier_cost: EvaluationCost, adder_cost: EvaluationCost, framework_name):
    if (multiplier_cost.is_nor_only != adder_cost.is_nor_only and multiplier_cost.is_nor_only is False):
        # If one is NOR-only and the other is not, we need to account for the difference
        raise ValueError("Incompatible gate configurations")

    if non_zeroes != 0:
        dimension = non_zeroes // dimension
        dimension += random.randint(-dimension//10, dimension//10)  # add random noise to simulate variability in 2D mapping

    multiplier_cost.calculate()
    adder_cost.calculate()

    total_energy = (dimension * multiplier_cost.total_energy) + ((dimension - 1) * adder_cost.total_energy) # + inter-crossbar movement cost
    total_latency = (dimension * multiplier_cost.total_latency) + ((dimension - 1) * adder_cost.total_latency) # + inter-crossbar movement latency
    total_area = (dimension * multiplier_cost.total_area) + ((dimension - 1) * adder_cost.total_area)

    overhead_factor = FRAMEWORK_OVERHEAD.get(framework_name)
    data_movment_energy = ADDER_BITS * adder_cost.e_inter * dimension * overhead_factor
    data_movment_latency = ADDER_BITS * adder_cost.t_inter

    total_energy += (dimension * 2) * data_movment_energy
    total_latency += (dimension * 2) * data_movment_latency


    # print(dimension, copies, (dimension * copies) * data_movment_energy, total_energy)
    # print((dimension * copies) * data_movment_energy / total_energy)

    return total_energy, total_latency, total_area

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
            "norm_energy_3d", "norm_energy_2d",
            "norm_latency_3d", "norm_latency_2d",
            "norm_area_3d", "norm_area_2d"
        ])
        for i, name in enumerate(benchmarks):
            writer.writerow([
                name,
                dimensions[i],
                float(norm_energy_3d[i]), float(norm_energy_2d[i]),
                float(norm_latency_3d[i]), float(norm_latency_2d[i]),
                float(norm_area_3d[i]), float(norm_area_2d[i])
            ])
    print(f"Wrote CSV summary to {csv_path}")

    # also print a small table to stdout
    for i, name in enumerate(benchmarks):
        print(f"{name}: energy 3D={norm_energy_3d[i]:.3e}, 2D={norm_energy_2d[i]:.3e} | latency 3D={norm_latency_3d[i]:.3e}, 2D={norm_latency_2d[i]:.3e}")

def analysis_benchmark_for_naive_2D(benchmarks, dimensions, non_zeroes, multiplier_cost, adder_cost, multiplier_nor_cost, adder_nor_cost):
    energies_3d = []
    latencies_3d = []
    energies_2d = []
    latencies_2d = []
    areas_3d = []
    areas_2d = []

    for dimension, non_zero in zip(dimensions, non_zeroes):
        e3, l3, a3 = evaluate_MVM_cost_3D(dimension, non_zero, multiplier_cost, adder_cost)
        e2, l2, a2 = evaluate_MVM_cost_2D(dimension, non_zero, multiplier_nor_cost, adder_nor_cost)
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

    """
    norm_*_matrix shape: (1 + m, n) where row0 = 3D, rows 1..m = frameworks
    """
    outdir = Path("data/plots")
    n = len(benchmarks)
    m = norm_energy_matrix.shape[0]  # number of frameworks (including 3D)
    x = np.arange(n)
    total_width = 0.85
    bar_w = total_width / m
    offsets = (np.arange(m) - (m - 1) / 2) * bar_w

    # colors and hatches (extend as needed)
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#B279A2"]
    hatches = ["", "///", "\\\\\\", "xxx", "+++","..."]

    title_fs = 16
    label_fs = 13
    xtick_fs = 11
    legend_fs = 11

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), constrained_layout=True)

    # Energy
    ax = axes[0]
    bars = []
    for r in range(m):
        y = norm_energy_matrix[r, :]
        b = ax.bar(x + offsets[r], y, bar_w, label=labels[r], color=colors[r % len(colors)],
                    edgecolor='black', linewidth=0.4, hatch=hatches[r % len(hatches)])
        bars.append(b)
    ax.set_ylabel("Normalized Energy", fontsize=label_fs, fontweight='bold')
    ax.set_title("Normalized Energy per Benchmark", fontsize=title_fs, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=xtick_fs, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(fontsize=legend_fs, ncol=min(4, m))

    # Latency
    ax = axes[1]
    for r in range(m):
        y = norm_latency_matrix[r, :]
        ax.bar(x + offsets[r], y, bar_w, label=labels[r] if r == 0 else None,
                color=colors[r % len(colors)], edgecolor='black', linewidth=0.4, hatch=hatches[r % len(hatches)])
    ax.set_ylabel("Normalized Latency", fontsize=label_fs, fontweight='bold')
    ax.set_title("Normalized Latency per Benchmark", fontsize=title_fs, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=xtick_fs, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Area
    ax = axes[2]
    for r in range(m):
        y = norm_area_matrix[r, :]
        ax.bar(x + offsets[r], y, bar_w, label=labels[r] if r == 0 else None,
                color=colors[r % len(colors)], edgecolor='black', linewidth=0.4, hatch=hatches[r % len(hatches)])
    ax.set_ylabel("Normalized Area", fontsize=label_fs, fontweight='bold')
    ax.set_title("Normalized Area per Benchmark", fontsize=title_fs, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=xtick_fs, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    out_svg = outdir / "mvm_multi_frameworks_normalized.svg"
    out_pdf = outdir / "mvm_multi_frameworks_normalized.pdf"
    fig.savefig(out_svg, format="svg", dpi=300)
    fig.savefig(out_pdf, format="pdf", dpi=300)
    print(f"Saved plots to {out_svg} and {out_pdf}")

import matplotlib.patches as mpatches
# ...existing code...

def plot_multi_frameworks(benchmarks, labels, norm_energy_matrix, norm_latency_matrix, norm_area_matrix):
    """
    norm_*_matrix shape: (m, n) where rows = frameworks (3D is last if you supplied that way)
    """
    outdir = Path("data/plots")
    n = len(benchmarks)
    m = norm_energy_matrix.shape[0]  # number of frameworks
    x = np.arange(n)
    total_width = 0.8
    bar_w = total_width / m
    offsets = (np.arange(m) - (m - 1) / 2) * bar_w

    print(m)

    graph_frac = 0.2
    inner_bar_w = bar_w * (1.0 - graph_frac)
    center_shift = (bar_w - inner_bar_w) / 2.0

    # colors and hatches (extend as needed)
    # colors = ["#7ee3f2", "#f2b26d", "#f2998f", "#72B7B2", "#b6f5a9"]
    # colors = ["#ed3507", "#0e4ee6", "#c28906", "#da07ed", "#07ed1a"]
    hatches = ["//", "-", "xx", "\\\\", ".."]

    title_fs = 16
    label_fs = 18
    xtick_fs = 15
    legend_fs = 22

    # create figure with some bottom margin for centered markers/legends
    fig, axes = plt.subplots(3, 1, figsize=(30, 12), constrained_layout=False)
    # increase bottom to leave room for legends placed below each subplot
    fig.subplots_adjust(hspace=0.45, top=0.99, bottom=0.2, left=0.05, right=0.98)

    # prepare legend handles (patches) once so we can place at bottom-middle for each subplot
    legend_patches = []
    for r in range(m):
        p = mpatches.Patch(facecolor='white',
                           edgecolor='black',
                           hatch=hatches[r % len(hatches)],
                           linewidth=1.5,
                           label=labels[r])
        legend_patches.append(p)

    # Energy
    ax = axes[0]
    for r in range(m):
        y = norm_energy_matrix[r, :]
        ax.bar(x + offsets[r] + center_shift, y, inner_bar_w,
               color='white',
               edgecolor='black',
               linewidth=0.4,
               hatch=hatches[r % len(hatches)])
    ax.set_ylabel("Normalized Energy", fontsize=label_fs, fontweight='bold')
    # ax.set_title("Normalized Energy per Benchmark", fontsize=title_fs, fontweight='bold')
    # ax.set_xticks(x)
    # ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=xtick_fs, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    # place legend (markers) at bottom center of this subplot
    # ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.48), ncol=min(m, 6),
            #   fontsize=legend_fs, frameon=False)

    # Latency
    ax = axes[1]
    for r in range(m):
        y = norm_latency_matrix[r, :]
        ax.bar(x + offsets[r] + center_shift, y, inner_bar_w,
               color='white',
               edgecolor='black',
               linewidth=0.4,
               hatch=hatches[r % len(hatches)])
    ax.set_ylabel("Normalized Latency", fontsize=label_fs, fontweight='bold')
    # ax.set_title("Normalized Latency per Benchmark", fontsize=title_fs, fontweight='bold')
    # ax.set_xticks(x)
    # ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=xtick_fs, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    # ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.48), ncol=min(m, 6),
            #   fontsize=legend_fs, frameon=False)

    # Area
    ax = axes[2]
    for r in range(m):
        y = norm_area_matrix[r, :]
        ax.bar(x + offsets[r] + center_shift, y, inner_bar_w,
               color='white',
               edgecolor='black',
               linewidth=0.4,
               hatch=hatches[r % len(hatches)])
    ax.set_ylabel("Normalized Area", fontsize=label_fs, fontweight='bold')
    # ax.set_title("Normalized Area per Benchmark", fontsize=title_fs, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=xtick_fs, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.9), ncol=min(m, 6),
              fontsize=legend_fs, frameon=False)

    # out_svg = outdir / "mvm_multi_frameworks_normalized.svg"
    # out_pdf = outdir / "mvm_multi_frameworks_normalized.pdf"
    # out_svg = outdir / "LLM_frameworks_normalized.svg"
    # out_pdf = outdir / "LLM_frameworks_normalized.pdf"
    
    out_svg = outdir / "AI_models_normalized.svg"
    out_pdf = outdir / "AI_models_normalized.pdf"

    fig.savefig(out_svg, format="svg", dpi=300)
    fig.savefig(out_pdf, format="pdf", dpi=300)
    print(f"Saved plots to {out_svg} and {out_pdf}")

def compare_multiple_frameworks(
        benchmarks, dimensions, non_zeroes,
        multiplier_cost, adder_cost,
        ultra_cost, simpler_cost, logic_cost, auto_cost, adder_nor_cost):
    energies_3d = []
    latencies_3d = []
    areas_3d = []
    for dimension, non_zero in zip(dimensions, non_zeroes):
        e3, l3, a3 = evaluate_MVM_cost_3D(dimension, non_zero, multiplier_cost, adder_cost)
        energies_3d.append(e3)
        latencies_3d.append(l3)
        areas_3d.append(a3)

    # frameworks to compare for 2D (multiplier models); adder stays adder_nor_cost
    frameworks = ["Ultra", "Simpler", "Logic", "Auto"]
    framework_costs = [ultra_cost, simpler_cost, logic_cost, auto_cost]

    # evaluate each framework across all benchmarks
    energies_2d_all = []
    latencies_2d_all = []
    areas_2d_all = []
    for fw_cost in framework_costs:
        e_list = []
        l_list = []
        a_list = []
        for dimension, non_zero in zip(dimensions, non_zeroes):
            e2, l2, a2 = evaluate_MVM_cost_2D(dimension, non_zero, fw_cost, adder_nor_cost)
            e_list.append(e2)
            l_list.append(l2)
            a_list.append(a2)
        energies_2d_all.append(e_list)
        latencies_2d_all.append(l_list)
        areas_2d_all.append(a_list)

    # stack into matrices: rows = [each 2D framework ..., 3D]
    import numpy as np
    energy_matrix = np.vstack(energies_2d_all + [energies_3d])    # shape (m + 1, n) with 3D last
    latency_matrix = np.vstack(latencies_2d_all + [latencies_3d])
    area_matrix = np.vstack(areas_2d_all + [areas_3d])

    # per-benchmark normalization across all frameworks (including 3D)
    eps = 1e-30
    max_energy_per_bench = np.maximum(energy_matrix.max(axis=0), eps)
    max_latency_per_bench = np.maximum(latency_matrix.max(axis=0), eps)
    max_area_per_bench = np.maximum(area_matrix.max(axis=0), eps)

    norm_energy_matrix = energy_matrix / max_energy_per_bench
    norm_latency_matrix = latency_matrix / max_latency_per_bench
    norm_area_matrix = area_matrix / max_area_per_bench

    # labels: frameworks first, then 3D last
    labels = frameworks + ["3D"]

    # write a CSV with raw + normalized values (rows = benchmarks)
    outdir = Path("data/plots")
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "mvm_costs_multi_frameworks.csv"
    import csv
    with csv_path.open("w", newline="") as cf:
        writer = csv.writer(cf)
        # header
        hdr = ["benchmark", "dimension"]
        for lbl in labels:
            hdr += [f"{lbl}_energy", f"{lbl}_latency", f"{lbl}_area"]
        for lbl in labels:
            hdr += [f"{lbl}_norm_energy", f"{lbl}_norm_latency", f"{lbl}_norm_area"]
        writer.writerow(hdr)

        nbench = len(dimensions)
        rows_count = energy_matrix.shape[0]
        for i in range(nbench):
            row = [benchmarks[i], dimensions[i]]
            # raw values (rows correspond to labels order)
            row += [energy_matrix[r, i] for r in range(rows_count)]
            row += [latency_matrix[r, i] for r in range(rows_count)]
            row += [area_matrix[r, i] for r in range(rows_count)]
            # normalized values
            row += [norm_energy_matrix[r, i] for r in range(rows_count)]
            row += [norm_latency_matrix[r, i] for r in range(rows_count)]
            row += [norm_area_matrix[r, i] for r in range(rows_count)]
            writer.writerow(row)

    print(f"Wrote multi-framework CSV to {csv_path}")

    # call plotting routine (labels list now has 3D last)
    plot_multi_frameworks(benchmarks, labels, norm_energy_matrix, norm_latency_matrix, norm_area_matrix)

if __name__ == "__main__":
    benchmarks = ['eris1176', 'cegb2919', 'raefsky1', 'fxm3_6 ', 'Na5', 'Ex5', 'fp', 'ex40', 'benzene', 'bcsstk33', 'graham1', 'net25', 'bundle1', 'Si10H16', 'Goodwin_040']
    dimensions = [1176, 2919, 3242, 5026, 5832, 6545, 7548, 7740, 8219, 8738, 9035, 9520, 10581, 17077, 17922]
    non_zeroes = [18552, 321543, 293409, 94026, 305630, 295680, 834222, 456188, 242669, 591904, 335472, 401200, 770811, 875923, 561677]
    # benchmarks = ['a', 'b']
    # dimensions = [20, 41]
    K = 4
    multiplier_cost = EvaluationCost(K=K, device_params=DeviceParams(), gateCount=MultiplierConst())
    adder_cost = EvaluationCost(K=K, device_params=DeviceParams(), gateCount=AdderConst())


    ultra_cost = EvaluationCost(K=K, device_params=DeviceParams(), gateCount=UltraCost())
    simpler_cost = EvaluationCost(K=K, device_params=DeviceParams(), gateCount=SimplerCost())
    logic_cost = EvaluationCost(K=K, device_params=DeviceParams(), gateCount=LogicCost())
    auto_cost = EvaluationCost(K=K, device_params=DeviceParams(), gateCount=AutoCost())
    multiplier_nor_cost = EvaluationCost(K=K, device_params=DeviceParams(), gateCount=MultiplierNorConst())

    # multiplier_nor_cost = EvaluationCost(K=4, device_params=DeviceParams(), gateCount=MultiplierNorConst())
    adder_nor_cost = EvaluationCost(K=K, device_params=DeviceParams(), gateCount=AdderNorConst())

    multiplier_cost.calculate()
    adder_cost.calculate()

    print(multiplier_cost.get_results())
    multiplier_nor_cost.calculate()
    print(multiplier_nor_cost.get_results())

    # multiplier_nor_cost.calculate()
    ultra_cost.calculate()
    simpler_cost.calculate()
    logic_cost.calculate()
    auto_cost.calculate()
    adder_nor_cost.calculate()

    # print(multiplier_cost.get_results())
    # print(adder_cost.get_results())

    # # print(multiplier_nor_cost.get_results())
    # print(ultra_cost.get_results())
    # print(simpler_cost.get_results())
    # print(logic_cost.get_results())
    # print(auto_cost.get_results())
    # print(adder_nor_cost.get_results())

    
    compare_multiple_frameworks(
        benchmarks, dimensions, non_zeroes,
        multiplier_cost, adder_cost,
        ultra_cost, simpler_cost, logic_cost, auto_cost, adder_nor_cost)
    # analysis_benchmark_for_naive_2D(benchmarks, dimensions, non_zeroes, multiplier_cost, adder_cost, ultra_cost, adder_nor_cost)

