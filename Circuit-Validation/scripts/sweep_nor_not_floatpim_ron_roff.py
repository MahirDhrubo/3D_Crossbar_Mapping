from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import sweep_nor_not_latency_energy as base


GENERATED_DIR = base.GENERATED_ROOT / "floatpim_ron_roff_latency_energy_sweep"
SUMMARY_CSV = base.RESULTS_DIR / "floatpim_ron_roff_latency_energy_sweep_summary.csv"
VALID_CSV = base.RESULTS_DIR / "floatpim_ron_roff_latency_energy_sweep_valid.csv"


def make_case_deck(
    operation: str,
    amplitude_v: float,
    pulse_width_ns: float,
    a: int,
    b: int | None,
) -> Path:
    text = base.TEMPLATE.read_text(encoding="ascii")
    text = base.set_param(text, "RON", "10k")
    text = base.set_param(text, "ROFF", "10Meg")
    text = base.configure_l23_operation(text, operation, amplitude_v, a, b)
    text = base.configure_timing(text, pulse_width_ns)
    text = base.append_driver_current_prints(text)

    amp_label = f"{amplitude_v:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
    width_label = f"{pulse_width_ns:.3f}".replace(".", "p")
    b_label = "x" if b is None else str(b)
    deck_dir = GENERATED_DIR / operation.lower() / f"v{amp_label}_w{width_label}ns"
    deck_dir.mkdir(parents=True, exist_ok=True)
    deck_path = deck_dir / f"{operation.lower()}_a{a}_b{b_label}.cir"
    deck_path.write_text(text, encoding="ascii")
    return deck_path


def run_sweep_point(operation: str, amplitude_v: float, pulse_width_ns: float) -> base.SweepPoint | None:
    cases: list[base.CaseResult] = []
    combinations: list[tuple[int, int | None]]
    if operation == "NOR":
        combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    else:
        combinations = [(0, None), (1, None)]

    for a, b in combinations:
        deck_path = make_case_deck(operation, amplitude_v, pulse_width_ns, a, b)
        case = base.analyze_case(operation, amplitude_v, pulse_width_ns, a, b, deck_path)
        cases.append(case)
        base.cleanup_generated_outputs(deck_path)

    if not all(case.passed for case in cases):
        return None

    latencies = [latency for case in cases for latency in case.output_latencies_ns]
    return base.SweepPoint(
        operation=operation,
        amplitude_v=amplitude_v,
        pulse_width_ns=pulse_width_ns,
        avg_switch_latency_ns=float(np.mean(latencies)),
        worst_switch_latency_ns=float(np.max(latencies)),
        avg_total_energy_j=float(np.mean([case.total_energy_j for case in cases])),
        avg_energy_per_gate_j=float(np.mean([case.energy_per_gate_j for case in cases])),
        cases=tuple(cases),
    )


def write_outputs(valid_points: list[base.SweepPoint]) -> None:
    with SUMMARY_CSV.open("w", newline="", encoding="ascii") as file:
        writer = csv.writer(file)
        writer.writerow([
            "operation",
            "ron",
            "roff",
            "amplitude_v",
            "pulse_width_ns",
            "avg_switch_latency_ns",
            "worst_switch_latency_ns",
            "avg_total_energy_j",
            "avg_energy_per_gate_j",
        ])
        for point in valid_points:
            writer.writerow([
                point.operation,
                "10k",
                "10Meg",
                f"{point.amplitude_v:.9g}",
                f"{point.pulse_width_ns:.9g}",
                f"{point.avg_switch_latency_ns:.9g}",
                f"{point.worst_switch_latency_ns:.9g}",
                f"{point.avg_total_energy_j:.9e}",
                f"{point.avg_energy_per_gate_j:.9e}",
            ])

    with VALID_CSV.open("w", newline="", encoding="ascii") as file:
        writer = csv.writer(file)
        writer.writerow([
            "operation",
            "ron",
            "roff",
            "amplitude_v",
            "pulse_width_ns",
            "A",
            "B",
            "expected_logic",
            "final_logic_inputs",
            "final_logic_outputs",
            "output_latencies_ns",
            "total_energy_j",
            "energy_per_gate_j",
        ])
        for point in valid_points:
            for case in point.cases:
                writer.writerow([
                    case.operation,
                    "10k",
                    "10Meg",
                    f"{case.amplitude_v:.9g}",
                    f"{case.pulse_width_ns:.9g}",
                    case.a,
                    "NA" if case.b is None else case.b,
                    case.expected,
                    " ".join(f"{value:.9g}" for value in case.final_logic_inputs),
                    " ".join(f"{value:.9g}" for value in case.final_logic_outputs),
                    " ".join(f"{value:.9g}" for value in case.output_latencies_ns) or "NA",
                    f"{case.total_energy_j:.9e}",
                    f"{case.energy_per_gate_j:.9e}",
                ])


def main() -> None:
    amplitudes = [round(v, 2) for v in np.arange(-1.2, -3.01, -0.1)]
    widths = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.5]

    valid_points: list[base.SweepPoint] = []
    for operation in ["NOR", "NOT"]:
        print(f"Sweeping {operation} with FloatPIM RON/ROFF: {len(amplitudes) * len(widths)} pulse points")
        for amplitude_v in amplitudes:
            for pulse_width_ns in widths:
                point = run_sweep_point(operation, amplitude_v, pulse_width_ns)
                if point is not None:
                    valid_points.append(point)

    write_outputs(valid_points)
    print(SUMMARY_CSV)
    print(VALID_CSV)

    for operation in ["NOR", "NOT"]:
        points = [point for point in valid_points if point.operation == operation]
        if not points:
            print(f"{operation}: no valid sweep points")
            continue
        fastest = min(points, key=lambda point: (point.avg_switch_latency_ns, point.avg_energy_per_gate_j))
        lowest_energy = min(points, key=lambda point: (point.avg_energy_per_gate_j, point.avg_switch_latency_ns))
        print(
            f"{operation} fastest valid: VOP={fastest.amplitude_v:g} V "
            f"width={fastest.pulse_width_ns:g} ns "
            f"avg_latency={fastest.avg_switch_latency_ns:.6g} ns "
            f"worst_latency={fastest.worst_switch_latency_ns:.6g} ns "
            f"avg_energy_per_gate={fastest.avg_energy_per_gate_j:.6e} J "
            f"avg_total_energy={fastest.avg_total_energy_j:.6e} J"
        )
        print(
            f"{operation} lowest-energy valid: VOP={lowest_energy.amplitude_v:g} V "
            f"width={lowest_energy.pulse_width_ns:g} ns "
            f"avg_latency={lowest_energy.avg_switch_latency_ns:.6g} ns "
            f"worst_latency={lowest_energy.worst_switch_latency_ns:.6g} ns "
            f"avg_energy_per_gate={lowest_energy.avg_energy_per_gate_j:.6e} J "
            f"avg_total_energy={lowest_energy.avg_total_energy_j:.6e} J"
        )


if __name__ == "__main__":
    main()
