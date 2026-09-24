from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import sweep_and_floatpim_ron_roff as and_sweep
import sweep_nor_not_latency_energy as base


GENERATED_DIR = base.GENERATED_ROOT / "switch_nonideal_floatpim_sensitivity"
SUMMARY_CSV = base.RESULTS_DIR / "switch_nonideal_floatpim_sensitivity_summary.csv"
VALID_CSV = base.RESULTS_DIR / "switch_nonideal_floatpim_sensitivity_valid.csv"

VTEAM_RON = "10k"
VTEAM_ROFF = "10Meg"

PULSES = {
    "NOR": (-2.2, 0.01),
    "NOT": (-2.2, 0.01),
    "AND": (2.7, 0.015),
}

SWITCH_POINTS = [
    ("ideal_ref", "1u", "1e15"),
    ("sw_1ohm", "1", "1e12"),
    ("sw_10ohm", "10", "1e12"),
    ("sw_100ohm", "100", "1e12"),
    ("sw_1kohm", "1k", "1e12"),
    ("sw_10kohm", "10k", "1e12"),
]


@dataclass(frozen=True)
class SwitchResult:
    operation: str
    switch_label: str
    switch_ron: str
    switch_roff: str
    amplitude_v: float
    pulse_width_ns: float
    cases_passed: int
    cases_total: int
    avg_switch_latency_ns: float | None
    worst_switch_latency_ns: float | None
    avg_total_energy_j: float
    avg_energy_per_gate_j: float
    cases: tuple[base.CaseResult, ...]


def set_switch_model(text: str, switch_ron: str, switch_roff: str) -> str:
    pattern = re.compile(
        r"^(\.MODEL\s+IDEAL_BIDIR_SW\s+VSWITCH\()([^)]+)(\))",
        re.MULTILINE | re.IGNORECASE,
    )
    replacement = rf"\g<1>RON={switch_ron} ROFF={switch_roff} VON=2 VOFF=1\g<3>"
    if not pattern.search(text):
        raise ValueError("Could not find IDEAL_BIDIR_SW model")
    return pattern.sub(replacement, text)


def label_value(value: str) -> str:
    return value.replace(".", "p").replace("+", "p").replace("-", "m")


def make_planar_case_deck(
    operation: str,
    amplitude_v: float,
    pulse_width_ns: float,
    switch_label: str,
    switch_ron: str,
    switch_roff: str,
    a: int,
    b: int | None,
) -> Path:
    text = base.TEMPLATE.read_text(encoding="ascii")
    text = base.set_param(text, "RON", VTEAM_RON)
    text = base.set_param(text, "ROFF", VTEAM_ROFF)
    text = set_switch_model(text, switch_ron, switch_roff)
    text = base.configure_l23_operation(text, operation, amplitude_v, a, b)
    text = base.configure_timing(text, pulse_width_ns)
    text = base.append_driver_current_prints(text)

    b_label = "x" if b is None else str(b)
    deck_dir = GENERATED_DIR / switch_label / operation.lower()
    deck_dir.mkdir(parents=True, exist_ok=True)
    deck_path = deck_dir / f"{operation.lower()}_a{a}_b{b_label}.cir"
    deck_path.write_text(text, encoding="ascii")
    return deck_path


def make_and_case_deck(
    amplitude_v: float,
    pulse_width_ns: float,
    switch_label: str,
    switch_ron: str,
    switch_roff: str,
    row0: tuple[int, int],
    row1: tuple[int, int],
) -> Path:
    text = base.TEMPLATE.read_text(encoding="ascii")
    text = and_sweep.configure_routed_and(text, amplitude_v, row0, row1)
    text = set_switch_model(text, switch_ron, switch_roff)
    text = base.configure_timing(text, pulse_width_ns)
    text = and_sweep.append_and_prints(text)

    deck_dir = GENERATED_DIR / switch_label / "and"
    deck_dir.mkdir(parents=True, exist_ok=True)
    deck_path = deck_dir / f"and_r0_{row0[0]}{row0[1]}_r1_{row1[0]}{row1[1]}.cir"
    deck_path.write_text(text, encoding="ascii")
    return deck_path


def run_operation(
    operation: str,
    switch_label: str,
    switch_ron: str,
    switch_roff: str,
) -> SwitchResult:
    amplitude_v, pulse_width_ns = PULSES[operation]
    cases: list[base.CaseResult] = []

    if operation in {"NOR", "NOT"}:
        combinations: list[tuple[int, int | None]]
        combinations = [(0, 0), (0, 1), (1, 0), (1, 1)] if operation == "NOR" else [(0, None), (1, None)]
        for a, b in combinations:
            deck_path = make_planar_case_deck(
                operation,
                amplitude_v,
                pulse_width_ns,
                switch_label,
                switch_ron,
                switch_roff,
                a,
                b,
            )
            cases.append(base.analyze_case(operation, amplitude_v, pulse_width_ns, a, b, deck_path))
            base.cleanup_generated_outputs(deck_path)
    else:
        combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for row0 in combinations:
            for row1 in combinations:
                deck_path = make_and_case_deck(
                    amplitude_v,
                    pulse_width_ns,
                    switch_label,
                    switch_ron,
                    switch_roff,
                    row0,
                    row1,
                )
                cases.append(and_sweep.analyze_case(amplitude_v, pulse_width_ns, row0, row1, deck_path))
                base.cleanup_generated_outputs(deck_path)

    latencies = [latency for case in cases for latency in case.output_latencies_ns]
    return SwitchResult(
        operation=operation,
        switch_label=switch_label,
        switch_ron=switch_ron,
        switch_roff=switch_roff,
        amplitude_v=amplitude_v,
        pulse_width_ns=pulse_width_ns,
        cases_passed=sum(1 for case in cases if case.passed),
        cases_total=len(cases),
        avg_switch_latency_ns=None if not latencies else float(np.mean(latencies)),
        worst_switch_latency_ns=None if not latencies else float(np.max(latencies)),
        avg_total_energy_j=float(np.mean([case.total_energy_j for case in cases])),
        avg_energy_per_gate_j=float(np.mean([case.energy_per_gate_j for case in cases])),
        cases=tuple(cases),
    )


def write_outputs(results: list[SwitchResult]) -> None:
    with SUMMARY_CSV.open("w", newline="", encoding="ascii") as file:
        writer = csv.writer(file)
        writer.writerow([
            "operation",
            "vteam_ron",
            "vteam_roff",
            "switch_label",
            "switch_ron_ohm",
            "switch_roff_ohm",
            "amplitude_v",
            "pulse_width_ns",
            "cases_passed",
            "cases_total",
            "all_cases_pass",
            "avg_switch_latency_ns",
            "worst_switch_latency_ns",
            "avg_total_energy_j",
            "avg_energy_per_gate_j",
        ])
        for result in results:
            writer.writerow([
                result.operation,
                VTEAM_RON,
                VTEAM_ROFF,
                result.switch_label,
                result.switch_ron,
                result.switch_roff,
                f"{result.amplitude_v:.9g}",
                f"{result.pulse_width_ns:.9g}",
                result.cases_passed,
                result.cases_total,
                int(result.cases_passed == result.cases_total),
                "NA" if result.avg_switch_latency_ns is None else f"{result.avg_switch_latency_ns:.9g}",
                "NA" if result.worst_switch_latency_ns is None else f"{result.worst_switch_latency_ns:.9g}",
                f"{result.avg_total_energy_j:.9e}",
                f"{result.avg_energy_per_gate_j:.9e}",
            ])

    with VALID_CSV.open("w", newline="", encoding="ascii") as file:
        writer = csv.writer(file)
        writer.writerow([
            "operation",
            "switch_label",
            "switch_ron_ohm",
            "switch_roff_ohm",
            "amplitude_v",
            "pulse_width_ns",
            "case_index",
            "passed",
            "final_logic_inputs",
            "final_logic_outputs",
            "output_latencies_ns",
            "total_energy_j",
            "energy_per_gate_j",
        ])
        for result in results:
            for index, case in enumerate(result.cases):
                writer.writerow([
                    result.operation,
                    result.switch_label,
                    result.switch_ron,
                    result.switch_roff,
                    f"{result.amplitude_v:.9g}",
                    f"{result.pulse_width_ns:.9g}",
                    index,
                    int(case.passed),
                    " ".join(f"{value:.9g}" for value in case.final_logic_inputs),
                    " ".join(f"{value:.9g}" for value in case.final_logic_outputs),
                    " ".join(f"{value:.9g}" for value in case.output_latencies_ns) or "NA",
                    f"{case.total_energy_j:.9e}",
                    f"{case.energy_per_gate_j:.9e}",
                ])


def main() -> None:
    results: list[SwitchResult] = []
    for switch_label, switch_ron, switch_roff in SWITCH_POINTS:
        print(f"Testing switch {switch_label}: RON={switch_ron}, ROFF={switch_roff}")
        for operation in ["NOR", "NOT", "AND"]:
            result = run_operation(operation, switch_label, switch_ron, switch_roff)
            results.append(result)
            pass_text = f"{result.cases_passed}/{result.cases_total}"
            latency = "NA" if result.avg_switch_latency_ns is None else f"{result.avg_switch_latency_ns:.6g} ns"
            energy = f"{result.avg_energy_per_gate_j * 1e15:.6g} fJ/gate"
            print(f"  {operation}: pass={pass_text}, avg_latency={latency}, avg_energy={energy}")

    write_outputs(results)
    print(SUMMARY_CSV)
    print(VALID_CSV)


if __name__ == "__main__":
    main()
