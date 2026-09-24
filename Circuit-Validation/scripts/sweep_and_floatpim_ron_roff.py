from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

import sweep_nor_not_latency_energy as base


GENERATED_DIR = base.GENERATED_ROOT / "and_floatpim_ron_roff_latency_energy_sweep"
SUMMARY_CSV = base.RESULTS_DIR / "and_floatpim_ron_roff_latency_energy_sweep_summary.csv"
VALID_CSV = base.RESULTS_DIR / "and_floatpim_ron_roff_latency_energy_sweep_valid.csv"


def configure_routed_and(
    text: str,
    amplitude_v: float,
    row0: tuple[int, int],
    row1: tuple[int, int],
) -> str:
    text = base.set_param(text, "RON", "10k")
    text = base.set_param(text, "ROFF", "10Meg")
    text = base.set_all_plane_inits(text, "L01", "{WLOGIC0}")
    text = base.set_all_plane_inits(text, "L12", "{WLOGIC0}")
    text = base.set_all_plane_inits(text, "L23", "{WLOGIC0}")

    for row, (a, b) in enumerate([row0, row1]):
        text = base.set_param(text, f"INIT_L01_R{row}_C0", base.logic_param(a))
        text = base.set_param(text, f"INIT_L12_R{row}_C0", base.logic_param(b))
        text = base.set_param(text, f"INIT_L23_R{row}_C0", "{WLOGIC0}")

    for plane in ["L01", "L12", "L23"]:
        for column in [0, 1, 2, 3]:
            text = base.set_param(text, f"ACC_{plane}_C{column}", "0")
    for column in [0, 1, 2, 3]:
        text = base.set_param(text, f"DIR_L0112_C{column}", "0")
    text = base.set_param(text, "DIR_L0112_C0", "1")
    text = base.set_param(text, "ACC_L23_C0", "1")

    for layer_node in [
        "L0_R0", "L0_R1", "L1_C0", "L1_C1", "L1_C2", "L1_C3",
        "L2_R0", "L2_R1", "L3_C0", "L3_C1", "L3_C2", "L3_C3",
    ]:
        text = base.set_param(text, f"DRV_EN_{layer_node}", "0")
        text = base.set_param(text, f"V_{layer_node}", "0")

    for row in [0, 1]:
        text = base.set_param(text, f"DRV_EN_L0_R{row}", "1")
        text = base.set_param(text, f"V_L0_R{row}", f"{amplitude_v:.9g}")
    text = base.set_param(text, "DRV_EN_L3_C0", "1")
    text = base.set_param(text, "V_L3_C0", "0")
    return text


def append_and_prints(text: str) -> str:
    terms = [
        "V(x_l01_r0_c0)", "V(x_l12_r0_c0)", "V(x_l23_r0_c0)",
        "V(x_l01_r1_c0)", "V(x_l12_r1_c0)", "V(x_l23_r1_c0)",
        "V(drv_l0_r0)", "V(drv_l0_r1)", "V(drv_l3_c0)",
        "I(VDRV_L0_R0)", "I(VDRV_L0_R1)", "I(VDRV_L3_C0)",
    ]
    return add_print_terms(text, terms)


def add_print_terms(text: str, terms: list[str]) -> str:
    upper = text.upper()
    missing = [term for term in terms if term.upper() not in upper]
    if not missing:
        return text
    import re

    return re.sub(r"^\.END\s*$", "+ " + " ".join(missing) + "\n\n.END", text, flags=re.MULTILINE | re.IGNORECASE)


def make_case_deck(
    amplitude_v: float,
    pulse_width_ns: float,
    row0: tuple[int, int],
    row1: tuple[int, int],
) -> Path:
    text = base.TEMPLATE.read_text(encoding="ascii")
    text = configure_routed_and(text, amplitude_v, row0, row1)
    text = base.configure_timing(text, pulse_width_ns)
    text = append_and_prints(text)

    amp_label = f"{amplitude_v:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
    width_label = f"{pulse_width_ns:.3f}".replace(".", "p")
    deck_dir = GENERATED_DIR / f"v{amp_label}_w{width_label}ns"
    deck_dir.mkdir(parents=True, exist_ok=True)
    deck_path = deck_dir / f"and_r0_{row0[0]}{row0[1]}_r1_{row1[0]}{row1[1]}.cir"
    deck_path.write_text(text, encoding="ascii")
    return deck_path


def state_column(df: pd.DataFrame, plane: str, row: int, column: int = 0) -> np.ndarray:
    return pd.to_numeric(df[base.find_column(df, f"V(x_{plane}_r{row}_c{column})")], errors="coerce").to_numpy()


def analyze_case(
    amplitude_v: float,
    pulse_width_ns: float,
    row0: tuple[int, int],
    row1: tuple[int, int],
    deck_path: Path,
) -> base.CaseResult:
    import subprocess

    subprocess.run(
        [base.xyce_cmd(), str(deck_path.relative_to(base.PROJECT_ROOT))],
        cwd=base.PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    csv_path = deck_path.with_suffix(deck_path.suffix + ".csv")
    df = pd.read_csv(csv_path)
    df.columns = [column.strip() for column in df.columns]

    time = pd.to_numeric(df[base.find_column(df, "TIME")], errors="coerce").to_numpy()
    start = base.T_RISE_START_NS * 1e-9
    end = (base.T_PULSE_START_NS + pulse_width_ns + base.T_RAMP_NS) * 1e-9

    input_ok = True
    output_ok = True
    final_inputs: list[float] = []
    final_outputs: list[float] = []
    output_latencies: list[float] = []

    expected_rows = []
    for row, (a, b) in enumerate([row0, row1]):
        expected = int(a and b)
        expected_rows.append(expected)
        for plane, expected_input in [("l01", a), ("l12", b)]:
            state = state_column(df, plane, row)
            final_logic = 1 - float(state[-1])
            final_inputs.append(final_logic)
            input_ok = input_ok and base.logic_passes(final_logic, expected_input)
            input_ok = input_ok and not base.crossed_threshold(time, state, start)

        output_state = state_column(df, "l23", row)
        final_output_logic = 1 - float(output_state[-1])
        final_outputs.append(final_output_logic)
        output_ok = output_ok and base.logic_passes(final_output_logic, expected)
        crossing = base.threshold_crossing(time, output_state, start)
        if expected:
            output_ok = output_ok and crossing is not None
            if crossing is not None:
                output_latencies.append((crossing - start) * 1e9)
        else:
            output_ok = output_ok and crossing is None

    total_power_abs = np.zeros_like(time)
    for source in ["VDRV_L0_R0", "VDRV_L0_R1", "VDRV_L3_C0"]:
        total_power_abs += base.source_power_abs(df, source)
    energy_time, power_window = base.interpolate_window(time, total_power_abs, start, end)
    total_energy = float(np.trapezoid(power_window, energy_time))

    return base.CaseResult(
        operation="AND",
        amplitude_v=amplitude_v,
        pulse_width_ns=pulse_width_ns,
        a=int(f"{row0[0]}{row0[1]}", 2),
        b=int(f"{row1[0]}{row1[1]}", 2),
        expected=-1,
        final_logic_inputs=tuple(final_inputs),
        final_logic_outputs=tuple(final_outputs),
        output_latencies_ns=tuple(output_latencies),
        total_energy_j=total_energy,
        energy_per_gate_j=total_energy / base.ROW_COUNT,
        passed=input_ok and output_ok,
    )


def run_sweep_point(amplitude_v: float, pulse_width_ns: float) -> base.SweepPoint | None:
    cases: list[base.CaseResult] = []
    combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for row0 in combinations:
        for row1 in combinations:
            deck_path = make_case_deck(amplitude_v, pulse_width_ns, row0, row1)
            case = analyze_case(amplitude_v, pulse_width_ns, row0, row1, deck_path)
            cases.append(case)
            base.cleanup_generated_outputs(deck_path)

    if not all(case.passed for case in cases):
        return None

    latencies = [latency for case in cases for latency in case.output_latencies_ns]
    return base.SweepPoint(
        operation="AND",
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
            "row0_ab",
            "row1_ab",
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
                    format(case.a, "02b"),
                    format(case.b or 0, "02b"),
                    " ".join(f"{value:.9g}" for value in case.final_logic_inputs),
                    " ".join(f"{value:.9g}" for value in case.final_logic_outputs),
                    " ".join(f"{value:.9g}" for value in case.output_latencies_ns) or "NA",
                    f"{case.total_energy_j:.9e}",
                    f"{case.energy_per_gate_j:.9e}",
                ])


def main() -> None:
    amplitudes = [round(v, 2) for v in np.arange(1.8, 3.01, 0.1)]
    widths = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.5]
    valid_points: list[base.SweepPoint] = []

    print(f"Sweeping routed AND with FloatPIM RON/ROFF: {len(amplitudes) * len(widths)} pulse points")
    for amplitude_v in amplitudes:
        for pulse_width_ns in widths:
            point = run_sweep_point(amplitude_v, pulse_width_ns)
            if point is not None:
                valid_points.append(point)

    write_outputs(valid_points)
    print(SUMMARY_CSV)
    print(VALID_CSV)

    if not valid_points:
        print("AND: no valid sweep points")
        return
    fastest = min(valid_points, key=lambda point: (point.avg_switch_latency_ns, point.avg_energy_per_gate_j))
    lowest_energy = min(valid_points, key=lambda point: (point.avg_energy_per_gate_j, point.avg_switch_latency_ns))
    for label, point in [("fastest", fastest), ("lowest-energy", lowest_energy)]:
        print(
            f"AND {label} valid: VOP={point.amplitude_v:g} V "
            f"width={point.pulse_width_ns:g} ns "
            f"avg_latency={point.avg_switch_latency_ns:.6g} ns "
            f"worst_latency={point.worst_switch_latency_ns:.6g} ns "
            f"avg_energy_per_gate={point.avg_energy_per_gate_j:.6e} J "
            f"avg_total_energy={point.avg_total_energy_j:.6e} J"
        )


if __name__ == "__main__":
    main()
