from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ARTIFACT_ROOT / "results" / "switch_nonideal_floatpim_sensitivity_summary.csv"
OUTPUT_PNG = ARTIFACT_ROOT / "figures" / "switch_nonideal_floatpim_sensitivity.png"


def parse_ohms(value: str) -> float:
    text = str(value).strip().lower()
    if text.endswith("meg"):
        return float(text[:-3]) * 1e6
    if text.endswith("k"):
        return float(text[:-1]) * 1e3
    if text.endswith("u"):
        return float(text[:-1]) * 1e-6
    return float(text)


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    df["switch_ron_numeric"] = df["switch_ron_ohm"].map(parse_ohms)
    df["energy_fj"] = df["avg_energy_per_gate_j"] * 1e15
    df["passed"] = df["all_cases_pass"].astype(bool)
    df["latency_numeric"] = pd.to_numeric(df["avg_switch_latency_ns"], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9), sharex=True, layout="constrained")
    colors = {"NOR": "tab:blue", "NOT": "tab:cyan", "AND": "tab:green"}

    for operation, rows in df.groupby("operation", sort=False):
        rows = rows.sort_values("switch_ron_numeric")
        color = colors.get(operation, "tab:gray")
        passed = rows[rows["passed"]]
        failed = rows[~rows["passed"]]

        axes[0].plot(passed["switch_ron_numeric"], passed["energy_fj"], marker="o", label=operation, color=color)
        axes[1].plot(passed["switch_ron_numeric"], passed["latency_numeric"], marker="o", label=operation, color=color)
        axes[2].plot(
            rows["switch_ron_numeric"],
            rows["cases_passed"] / rows["cases_total"],
            marker="o",
            label=operation,
            color=color,
        )
        if not failed.empty:
            axes[0].scatter(failed["switch_ron_numeric"], failed["energy_fj"], marker="x", color=color, s=48)
            axes[1].scatter(failed["switch_ron_numeric"], failed["latency_numeric"], marker="x", color=color, s=48)

    axes[0].set_ylabel("Avg energy/gate (fJ)")
    axes[0].set_title("Energy")
    axes[1].set_ylabel("Avg switching latency (ns)")
    axes[1].set_title("Latency")
    axes[2].set_ylabel("Truth-table pass fraction")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("Correctness")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Switch ON resistance (ohm)")
        ax.grid(True, which="both", alpha=0.3)
    axes[2].legend(loc="lower left")
    fig.suptitle("Non-ideal switch sensitivity with VTEAM RON=10k, ROFF=10Meg")
    fig.savefig(OUTPUT_PNG, dpi=300)
    plt.close(fig)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
