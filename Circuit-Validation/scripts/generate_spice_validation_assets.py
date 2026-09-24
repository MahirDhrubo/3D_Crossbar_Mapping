from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.titlesize": 10,
})


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ARTIFACT_ROOT
NETLIST_DIR = ARTIFACT_ROOT / "netlists"
RESULTS_DIR = ARTIFACT_ROOT / "results"
GENERATED_DIR = ARTIFACT_ROOT / "generated" / "series_and_validation"
FIG_DIR = ARTIFACT_ROOT / "figures"
TABLE_DIR = ARTIFACT_ROOT / "tables"
NAIVE_TEMPLATE = NETLIST_DIR / "naive_3d_crossbar.cir"
ROUTED_TEMPLATE = NETLIST_DIR / "routed_3d_crossbar.cir"
PLANAR_SUMMARY = RESULTS_DIR / "floatpim_ron_roff_latency_energy_sweep_summary.csv"
PLANAR_VALID = RESULTS_DIR / "floatpim_ron_roff_latency_energy_sweep_valid.csv"
AND_SUMMARY = RESULTS_DIR / "and_floatpim_ron_roff_latency_energy_sweep_summary.csv"
SET_RESET_SUMMARY = RESULTS_DIR / "set_reset_floatpim_ron_roff_latency_energy_summary.csv"
SERIES_CSV = RESULTS_DIR / "series_and_row_parallel_validation.csv"
BEST_POINTS_CSV = RESULTS_DIR / "logic_lowest_energy_best_points.csv"
SWEEP_POINTS_CSV = RESULTS_DIR / "logic_floatpim_sweep_points.csv"

LOGIC_LOW_MAX = 0.1
LOGIC_HIGH_MIN = 0.9
AND_VALIDATION_AMPLITUDE_V = 2.7
AND_VALIDATION_WIDTH_NS = 0.015


@dataclass(frozen=True)
class SeriesCase:
    architecture: str
    row0_a: int
    row0_b: int
    row1_a: int
    row1_b: int
    row0_expected: int
    row1_expected: int
    row0_final_logic: float
    row1_final_logic: float
    row0_latency_ns: float | None
    row1_latency_ns: float | None
    row0_pass: bool
    row1_pass: bool
    deck_path: Path
    csv_path: Path


def xyce_cmd() -> str:
    configured = os.environ.get("XYCE_BIN")
    if configured:
        return configured
    found = shutil.which("Xyce")
    if found:
        return found
    raise FileNotFoundError("Could not find Xyce. Set XYCE_BIN or put Xyce on PATH.")


def set_param(text: str, name: str, value: str) -> str:
    pattern = re.compile(rf"^(\.PARAM\s+{re.escape(name)}=)([^\s]+)", re.MULTILINE | re.IGNORECASE)
    if not pattern.search(text):
        raise ValueError(f"Could not find .PARAM {name}")
    return pattern.sub(rf"\g<1>{value}", text)


def add_print_terms(text: str, terms: list[str]) -> str:
    upper = text.upper()
    missing = [term for term in terms if term.upper() not in upper]
    if not missing:
        return text
    return re.sub(r"^\.END\s*$", "+ " + " ".join(missing) + "\n\n.END", text, flags=re.MULTILINE | re.IGNORECASE)


def logic_param(value: int) -> str:
    return "{WLOGIC1}" if value else "{WLOGIC0}"


def set_all_inits(text: str, planes: list[str], value: str) -> str:
    for plane in planes:
        for row in [0, 1]:
            for column in [0, 1, 2, 3]:
                text = set_param(text, f"INIT_{plane}_R{row}_C{column}", value)
    return text


def configure_common_timing(text: str) -> str:
    pulse_end = 1.1 + AND_VALIDATION_WIDTH_NS
    fall_end = pulse_end + 0.1
    stop = fall_end + 1.8
    text = set_param(text, "T_RISE_START", "1n")
    text = set_param(text, "T_PULSE_START", "1.1ns")
    text = set_param(text, "T_PULSE_END", f"{pulse_end:g}ns")
    text = set_param(text, "T_FALL_END", f"{fall_end:g}ns")
    text = set_param(text, "T_STOP", f"{stop:g}ns")
    return text


def configure_naive_vertical_and(text: str, row0: tuple[int, int], row1: tuple[int, int]) -> str:
    text = set_param(text, "RON", "10k")
    text = set_param(text, "ROFF", "10Meg")
    text = set_all_inits(text, ["L01", "L12", "L23"], "{WLOGIC0}")
    for row, (a, b) in enumerate([row0, row1]):
        text = set_param(text, f"INIT_L01_R{row}_C0", logic_param(a))
        text = set_param(text, f"INIT_L12_R{row}_C0", logic_param(b))
        text = set_param(text, f"INIT_L23_R{row}_C0", "{WLOGIC0}")

    for plane in ["L01", "L12", "L23"]:
        for column in [0, 1, 2, 3]:
            text = set_param(text, f"SEL_{plane}_C{column}", "1" if column == 0 else "0")

    for line in [
        "L0_R0", "L0_R1", "L1_C0", "L1_C1", "L1_C2", "L1_C3",
        "L2_R0", "L2_R1", "L3_C0", "L3_C1", "L3_C2", "L3_C3",
    ]:
        text = set_param(text, f"V_{line}", "0")
        text = set_param(text, f"DRV_EN_{line}", "0")

    for row in [0, 1]:
        text = set_param(text, f"V_L0_R{row}", f"{AND_VALIDATION_AMPLITUDE_V:g}")
        text = set_param(text, f"DRV_EN_L0_R{row}", "1")
    text = set_param(text, "V_L3_C0", "0")
    text = set_param(text, "DRV_EN_L3_C0", "1")
    text = configure_common_timing(text)
    return text


def configure_routed_vertical_and(text: str, row0: tuple[int, int], row1: tuple[int, int]) -> str:
    text = set_param(text, "RON", "10k")
    text = set_param(text, "ROFF", "10Meg")
    text = set_all_inits(text, ["L01", "L12", "L23"], "{WLOGIC0}")
    for row, (a, b) in enumerate([row0, row1]):
        text = set_param(text, f"INIT_L01_R{row}_C0", logic_param(a))
        text = set_param(text, f"INIT_L12_R{row}_C0", logic_param(b))
        text = set_param(text, f"INIT_L23_R{row}_C0", "{WLOGIC0}")

    for plane in ["L01", "L12", "L23"]:
        for column in [0, 1, 2, 3]:
            text = set_param(text, f"ACC_{plane}_C{column}", "0")
    for column in [0, 1, 2, 3]:
        text = set_param(text, f"DIR_L0112_C{column}", "1" if column == 0 else "0")
    text = set_param(text, "ACC_L23_C0", "1")

    for line in [
        "L0_R0", "L0_R1", "L1_C0", "L1_C1", "L1_C2", "L1_C3",
        "L2_R0", "L2_R1", "L3_C0", "L3_C1", "L3_C2", "L3_C3",
    ]:
        text = set_param(text, f"V_{line}", "0")
        text = set_param(text, f"DRV_EN_{line}", "0")

    for row in [0, 1]:
        text = set_param(text, f"V_L0_R{row}", f"{AND_VALIDATION_AMPLITUDE_V:g}")
        text = set_param(text, f"DRV_EN_L0_R{row}", "1")
    text = set_param(text, "V_L3_C0", "0")
    text = set_param(text, "DRV_EN_L3_C0", "1")
    text = configure_common_timing(text)
    return text


def make_series_deck(architecture: str, row0: tuple[int, int], row1: tuple[int, int]) -> Path:
    if architecture == "naive":
        text = NAIVE_TEMPLATE.read_text(encoding="ascii")
        text = configure_naive_vertical_and(text, row0, row1)
    elif architecture == "series_isolated":
        text = ROUTED_TEMPLATE.read_text(encoding="ascii")
        text = configure_routed_vertical_and(text, row0, row1)
    else:
        raise ValueError(architecture)

    text = add_print_terms(text, ["V(x_l23_r0_c0)", "V(x_l23_r1_c0)"])
    directory = GENERATED_DIR / architecture
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"and_r0_{row0[0]}{row0[1]}_r1_{row1[0]}{row1[1]}.cir"
    path.write_text(text, encoding="ascii")
    return path


def find_column(df: pd.DataFrame, target: str) -> str:
    normalized = target.upper()
    for column in df.columns:
        if column.strip().upper() == normalized:
            return column
    raise KeyError(f"Missing {target}. Available columns: {list(df.columns)}")


def threshold_crossing(time: np.ndarray, state: np.ndarray, start: float = 1e-9, threshold: float = 0.5) -> float | None:
    start_index = int(np.searchsorted(time, start, side="left"))
    for i in range(max(1, start_index), len(state)):
        y0 = state[i - 1] - threshold
        y1 = state[i] - threshold
        if y0 == 0:
            return float(time[i - 1])
        if y0 * y1 < 0:
            frac = -y0 / (y1 - y0)
            return float(time[i - 1] + frac * (time[i] - time[i - 1]))
    return None


def logic_passes(logic: float, expected: int) -> bool:
    return logic >= LOGIC_HIGH_MIN if expected else logic <= LOGIC_LOW_MAX


def run_series_case(architecture: str, row0: tuple[int, int], row1: tuple[int, int]) -> SeriesCase:
    deck_path = make_series_deck(architecture, row0, row1)
    subprocess.run(
        [xyce_cmd(), str(deck_path.relative_to(PROJECT_ROOT))],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    csv_path = deck_path.with_suffix(deck_path.suffix + ".csv")
    df = pd.read_csv(csv_path)
    df.columns = [column.strip() for column in df.columns]
    time = pd.to_numeric(df[find_column(df, "TIME")], errors="coerce").to_numpy()
    r0_state = pd.to_numeric(df[find_column(df, "V(X_L23_R0_C0)")], errors="coerce").to_numpy()
    r1_state = pd.to_numeric(df[find_column(df, "V(X_L23_R1_C0)")], errors="coerce").to_numpy()

    r0_expected = int(row0[0] and row0[1])
    r1_expected = int(row1[0] and row1[1])
    r0_logic = 1 - float(r0_state[-1])
    r1_logic = 1 - float(r1_state[-1])
    r0_cross = threshold_crossing(time, r0_state)
    r1_cross = threshold_crossing(time, r1_state)

    return SeriesCase(
        architecture=architecture,
        row0_a=row0[0],
        row0_b=row0[1],
        row1_a=row1[0],
        row1_b=row1[1],
        row0_expected=r0_expected,
        row1_expected=r1_expected,
        row0_final_logic=r0_logic,
        row1_final_logic=r1_logic,
        row0_latency_ns=None if r0_cross is None else (r0_cross - 1e-9) * 1e9,
        row1_latency_ns=None if r1_cross is None else (r1_cross - 1e-9) * 1e9,
        row0_pass=logic_passes(r0_logic, r0_expected),
        row1_pass=logic_passes(r1_logic, r1_expected),
        deck_path=deck_path,
        csv_path=csv_path,
    )


def run_series_validation() -> list[SeriesCase]:
    cases: list[SeriesCase] = []
    combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for architecture in ["naive", "series_isolated"]:
        print(f"Running {architecture} vertical AND cases...")
        for row0 in combos:
            for row1 in combos:
                cases.append(run_series_case(architecture, row0, row1))
    with SERIES_CSV.open("w", newline="", encoding="ascii") as file:
        writer = csv.writer(file)
        writer.writerow([
            "architecture",
            "row0_ab",
            "row1_ab",
            "row0_expected",
            "row1_expected",
            "row0_final_logic",
            "row1_final_logic",
            "row0_latency_ns",
            "row1_latency_ns",
            "row0_pass",
            "row1_pass",
            "both_rows_pass",
            "deck",
            "csv",
        ])
        for case in cases:
            writer.writerow([
                case.architecture,
                f"{case.row0_a}{case.row0_b}",
                f"{case.row1_a}{case.row1_b}",
                case.row0_expected,
                case.row1_expected,
                f"{case.row0_final_logic:.9g}",
                f"{case.row1_final_logic:.9g}",
                "NA" if case.row0_latency_ns is None else f"{case.row0_latency_ns:.9g}",
                "NA" if case.row1_latency_ns is None else f"{case.row1_latency_ns:.9g}",
                int(case.row0_pass),
                int(case.row1_pass),
                int(case.row0_pass and case.row1_pass),
                case.deck_path.relative_to(PROJECT_ROOT),
                case.csv_path.relative_to(PROJECT_ROOT),
            ])
    return cases


def plot_series_heatmap(cases: list[SeriesCase]) -> Path:
    labels = ["00", "01", "10", "11"]
    fig = plt.figure(figsize=(7.0, 2.85), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.42], wspace=0.02)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1], sharey=ax0)
    axes = [ax0, ax1]
    legend_ax = fig.add_subplot(grid[0, 2])
    legend_ax.axis("off")
    styles = {
        0: {"facecolor": "#d73027", "hatch": "xx", "textcolor": "white"},
        1: {"facecolor": "#fee08b", "hatch": "///", "textcolor": "black"},
        2: {"facecolor": "#1a9850", "hatch": "", "textcolor": "white"},
    }
    mark_by_pass = {
        (False, False): "x x",
        (True, False): "✓ x",
        (False, True): "x ✓",
        (True, True): "✓ ✓",
    }
    for ax, architecture, title in zip(
        axes,
        ["naive", "series_isolated"],
        ["Shared column", "Bypass routing"],
    ):
        matrix = np.zeros((4, 4))
        annotations: list[list[str]] = [["" for _ in range(4)] for _ in range(4)]
        for case in cases:
            if case.architecture != architecture:
                continue
            i = labels.index(f"{case.row0_a}{case.row0_b}")
            j = labels.index(f"{case.row1_a}{case.row1_b}")
            matrix[i, j] = int(case.row0_pass) + int(case.row1_pass)
            annotations[i][j] = mark_by_pass[(case.row0_pass, case.row1_pass)]
        for i in range(4):
            for j in range(4):
                correct_rows = int(matrix[i, j])
                style = styles[correct_rows]
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=style["facecolor"],
                    edgecolor="black",
                    hatch=style["hatch"],
                    linewidth=0.8,
                ))
                ax.text(
                    j,
                    i,
                    annotations[i][j],
                    ha="center",
                    va="center",
                    fontsize=18,
                    color=style["textcolor"],
                    fontweight="bold",
                )
        ax.set_title(title, pad=4, fontsize=18)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(3.5, -0.5)
        ax.set_xticks(range(4), labels)
        ax.set_yticks(range(4), labels)
        ax.tick_params(axis="both", labelsize=18)
        ax.set_xlabel("row 1 AB", fontsize=18)
        ax.set_aspect("equal")
    axes[0].set_ylabel("row 0 AB", fontsize=18)
    legend_handles = [
        Patch(facecolor=styles[0]["facecolor"], edgecolor="black", hatch=styles[0]["hatch"], label="Fail"),
        Patch(facecolor=styles[1]["facecolor"], edgecolor="black", hatch=styles[1]["hatch"], label="Partial"),
        Patch(facecolor=styles[2]["facecolor"], edgecolor="black", label="Pass"),
    ]
    legend_ax.legend(handles=legend_handles, loc="center left", frameon=False, fontsize=16, handlelength=1.2)
    path = FIG_DIR / "fig_series_and_truth_table_heatmap.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_series_transient(cases: list[SeriesCase]) -> Path:
    selected = {
        case.architecture: case
        for case in cases
        if (case.row0_a, case.row0_b) == (1, 1) and (case.row1_a, case.row1_b) == (0, 1)
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), sharex=True, sharey=True, layout="constrained")
    for ax, architecture, title in zip(
        axes,
        ["naive", "series_isolated"],
        ["Shared column", "Bypass routing"],
    ):
        case = selected[architecture]
        df = pd.read_csv(case.csv_path)
        df.columns = [column.strip() for column in df.columns]
        time_ns = pd.to_numeric(df[find_column(df, "TIME")], errors="coerce") * 1e9
        row0_logic = 1 - pd.to_numeric(df[find_column(df, "V(X_L23_R0_C0)")], errors="coerce")
        row1_logic = 1 - pd.to_numeric(df[find_column(df, "V(X_L23_R1_C0)")], errors="coerce")
        ax.plot(time_ns, row0_logic, label="row0 AB=11", linewidth=1.9, color="tab:blue", linestyle="-")
        ax.plot(time_ns, row1_logic, label="row1 AB=01", linewidth=1.9, color="tab:orange", linestyle="--")
        ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1.1)
        ax.axvspan(1.1, 1.1 + AND_VALIDATION_WIDTH_NS, facecolor="tab:green", alpha=0.12, edgecolor="none")
        ax.set_title(title, pad=3, fontsize=18)
        ax.set_xlabel("Time (ns)", fontsize=18)
        ax.tick_params(axis="both", labelsize=18)
        ax.set_xlim(0.95, 1.35)
        ax.set_ylim(-0.05, 1.02)
        ax.grid(True, alpha=0.35)
    axes[0].set_ylabel("Logic", fontsize=18)
    axes[0].legend(loc="lower right", fontsize=16, frameon=False)
    path = FIG_DIR / "fig_series_and_transient_comparison.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def combined_sweep_points() -> pd.DataFrame:
    planar = pd.read_csv(PLANAR_SUMMARY)
    and_summary = pd.read_csv(AND_SUMMARY)
    summary = pd.concat([planar, and_summary], ignore_index=True)
    summary.to_csv(SWEEP_POINTS_CSV, index=False)
    return summary


def best_operation_points(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for operation in ["NOR", "NOT", "AND"]:
        op_rows = summary[summary["operation"] == operation].copy()
        fastest = op_rows.sort_values(["avg_switch_latency_ns", "avg_energy_per_gate_j"]).iloc[0]
        lowest = op_rows.sort_values(["avg_energy_per_gate_j", "avg_switch_latency_ns"]).iloc[0]
        for label, row in [("fastest", fastest), ("lowest_energy", lowest)]:
            rows.append({
                "operation": operation,
                "selection": label,
                "amplitude_v": row["amplitude_v"],
                "pulse_width_ns": row["pulse_width_ns"],
                "avg_switch_latency_ns": row["avg_switch_latency_ns"],
                "worst_switch_latency_ns": row["worst_switch_latency_ns"],
                "avg_total_energy_j": row["avg_total_energy_j"],
                "avg_energy_per_gate_j": row["avg_energy_per_gate_j"],
            })
    set_reset = pd.read_csv(SET_RESET_SUMMARY)
    valid_set_reset = set_reset[
        ((set_reset["operation"] == "SET") & (set_reset["final_logic"] >= LOGIC_HIGH_MIN))
        | ((set_reset["operation"] == "RESET") & (set_reset["final_logic"] <= LOGIC_LOW_MAX))
    ].copy()
    for operation in ["SET", "RESET"]:
        op_rows = valid_set_reset[valid_set_reset["operation"] == operation].copy()
        fastest = op_rows.sort_values(["latency_ns", "energy_j"]).iloc[0]
        lowest = op_rows.sort_values(["energy_j", "latency_ns"]).iloc[0]
        for label, row in [("fastest", fastest), ("lowest_energy", lowest)]:
            rows.append({
                "operation": operation,
                "selection": label,
                "amplitude_v": row["amplitude_v"],
                "pulse_width_ns": row["pulse_width_ns"],
                "avg_switch_latency_ns": row["latency_ns"],
                "worst_switch_latency_ns": row["latency_ns"],
                "avg_total_energy_j": row["energy_j"],
                "avg_energy_per_gate_j": row["energy_j"],
            })
    result = pd.DataFrame(rows)
    result.to_csv(BEST_POINTS_CSV, index=False)
    return result


def plot_best_latency_energy(best: pd.DataFrame) -> Path:
    lowest = best[best["selection"] == "lowest_energy"].copy()
    labels = lowest["operation"].tolist()
    x = np.arange(len(labels))
    colors = ["tab:blue", "tab:cyan", "tab:green", "tab:orange", "tab:red"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8), layout="constrained")
    axes[0].bar(x, lowest["avg_switch_latency_ns"], color=colors)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Avg switching latency (ns)")
    axes[0].set_title("Lowest-energy point latency")
    axes[0].grid(axis="y", alpha=0.3)
    energy_fj = lowest["avg_energy_per_gate_j"].to_numpy() * 1e15
    axes[1].bar(x, energy_fj, color=colors)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Avg energy (fJ)")
    axes[1].set_title("Lowest valid energy")
    axes[1].grid(axis="y", alpha=0.3)
    for ax, values in zip(axes, [lowest["avg_switch_latency_ns"].to_numpy(), energy_fj]):
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:.3g}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Best valid operation points with RON=10k, ROFF=10Meg")
    path = FIG_DIR / "fig_lowest_energy_latency.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_sweep_space(summary: pd.DataFrame, best: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7), sharey=True, layout="constrained")
    for ax, operation in zip(axes, ["NOR", "NOT", "AND"]):
        rows = summary[summary["operation"] == operation].copy()
        scatter = ax.scatter(
            rows["avg_switch_latency_ns"],
            rows["avg_energy_per_gate_j"] * 1e15,
            c=rows["amplitude_v"],
            s=np.clip(rows["pulse_width_ns"] * 220, 14, 90),
            cmap="viridis",
            alpha=0.82,
            edgecolors="none",
        )
        chosen = best[(best["operation"] == operation) & (best["selection"] == "lowest_energy")].iloc[0]
        ax.scatter(
            [chosen["avg_switch_latency_ns"]],
            [chosen["avg_energy_per_gate_j"] * 1e15],
            marker="*",
            s=130,
            color="crimson",
            edgecolor="black",
            linewidth=0.4,
            zorder=5,
        )
        ax.set_title(operation)
        ax.set_xlabel("Avg latency (ns)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Avg energy per gate (fJ)")
    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.9, fraction=0.04, pad=0.02)
    cbar.set_label("Pulse amplitude (V)")
    fig.suptitle("Valid pulse sweep points; star marks reported lowest-energy point")
    path = FIG_DIR / "fig_floatpim_valid_sweep_space.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_planar_truth_outputs(best: pd.DataFrame) -> Path:
    valid = pd.read_csv(PLANAR_VALID)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), layout="constrained")
    for ax, operation in zip(axes, ["NOR", "NOT"]):
        row = best[(best["operation"] == operation) & (best["selection"] == "lowest_energy")].iloc[0]
        cases = valid[
            (valid["operation"] == operation)
            & np.isclose(valid["amplitude_v"], row["amplitude_v"])
            & np.isclose(valid["pulse_width_ns"], row["pulse_width_ns"])
        ].copy()
        labels = []
        expected = []
        row0 = []
        row1 = []
        for _, case in cases.iterrows():
            if operation == "NOR":
                labels.append(f"{int(case['A'])}{int(case['B'])}")
            else:
                labels.append(str(int(case["A"])))
            expected.append(float(case["expected_logic"]))
            outputs = [float(value) for value in str(case["final_logic_outputs"]).split()]
            row0.append(outputs[0])
            row1.append(outputs[1])
        x = np.arange(len(labels))
        ax.plot(x, expected, "k--", label="expected", linewidth=1.5)
        ax.scatter(x - 0.06, row0, label="row0", s=36)
        ax.scatter(x + 0.06, row1, label="row1", s=36)
        ax.set_xticks(x, labels)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"{operation} final output logic")
        ax.set_xlabel("input" + (" AB" if operation == "NOR" else " A"))
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Logic = 1 - state")
    axes[1].legend(loc="center right")
    fig.suptitle("Planar row-parallel output agreement at lowest-energy points")
    path = FIG_DIR / "fig_planar_nor_not_truth_outputs.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def write_device_params_table() -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / "tab_device_params.tex"
    path.write_text(
        r"""\begin{tabular}{ll}
\hline
Parameter & Value \\
\hline
$K_\mathrm{on}$ & $-216.2$ \\
$K_\mathrm{off}$ & $0.091$ \\
$V_\mathrm{on}$ & $-1.5\,\mathrm{V}$ \\
$V_\mathrm{off}$ & $0.3\,\mathrm{V}$ \\
$w_\mathrm{on}$ & $0$ \\
$w_\mathrm{off}$ & $3\,\mathrm{nm}$ \\
$\alpha_\mathrm{on}, \alpha_\mathrm{off}$ & $4, 4$ \\
$R_\mathrm{on}$ & $10\,\mathrm{k}\Omega$ \\
$R_\mathrm{off}$ & $10\,\mathrm{M}\Omega$ \\
\hline
\end{tabular}
""",
        encoding="ascii",
    )
    return path


def write_asset_readme(paths: list[Path]) -> Path:
    readme = RESULTS_DIR / "generated_assets_README.md"
    rels = "\n".join(f"- `{path.relative_to(PROJECT_ROOT)}`" for path in paths)
    readme.write_text(
        f"""# SPICE Validation Figure Assets

This folder contains generated support material for the operating-mode and
SPICE-validation discussion.

Generated assets:

{rels}

Notes:

- `fig_series_and_truth_table_heatmap.pdf` compares naive shared-column vertical
  AND execution against the series-isolated routing across all two-row input
  combinations. Each cell annotates final output logic as `row0/row1`.
- `fig_series_and_transient_comparison.pdf` shows the representative concurrent
  case `row0 AB=11`, `row1 AB=01`, where the naive shared-column stack couples
  the rows but the routed version keeps row 1 low.
- `logic_lowest_energy_best_points.csv` reports fastest and lowest-energy
  points for NOR, NOT, AND, and standalone SET/RESET using `RON=10k` and
  `ROFF=10Meg`.
- `fig_lowest_energy_latency.png` reports the selected lowest-energy operation
  points using the same resistance settings.
- `fig_floatpim_valid_sweep_space.png` shows the valid pulse-amplitude/pulse-
  width sweep points used to justify the selected values; the star marks the
  reported lowest-energy point for each operation.
- `fig_planar_nor_not_truth_outputs.png` is built from the 07 planar NOR/NOT
  FloatPIM-resistance sweep CSVs.
- The current generated validation covers vertical AND. It does not provide a
  separate NAND validation figure; keep the paper text to AND unless/until a
  NAND-specific validated deck is added.
""",
        encoding="ascii",
    )
    return readme


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cases = run_series_validation()
    sweep = combined_sweep_points()
    best = best_operation_points(sweep)
    generated = [
        SERIES_CSV,
        SWEEP_POINTS_CSV,
        BEST_POINTS_CSV,
        plot_series_heatmap(cases),
        plot_series_transient(cases),
        plot_best_latency_energy(best),
        plot_sweep_space(sweep, best),
        plot_planar_truth_outputs(best),
        write_device_params_table(),
    ]
    generated.append(write_asset_readme(generated))
    print("Generated:")
    for path in generated:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
