from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ARTIFACT_ROOT
NETLIST_DIR = ARTIFACT_ROOT / "netlists"
GENERATED_ROOT = ARTIFACT_ROOT / "generated"
RESULTS_DIR = ARTIFACT_ROOT / "results"
TEMPLATE = NETLIST_DIR / "routed_3d_crossbar.cir"
GENERATED_DIR = GENERATED_ROOT / "latency_energy_sweep"
SUMMARY_CSV = RESULTS_DIR / "latency_energy_sweep_summary.csv"
VALID_CSV = RESULTS_DIR / "latency_energy_sweep_valid.csv"

T_RISE_START_NS = 1.0
T_PULSE_START_NS = 1.1
T_RAMP_NS = 0.1
T_SETTLE_NS = 1.8
ROW_COUNT = 2
LOGIC_LOW_MAX = 0.1
LOGIC_HIGH_MIN = 0.9


@dataclass(frozen=True)
class CaseResult:
    operation: str
    amplitude_v: float
    pulse_width_ns: float
    a: int
    b: int | None
    expected: int
    final_logic_inputs: tuple[float, ...]
    final_logic_outputs: tuple[float, ...]
    output_latencies_ns: tuple[float, ...]
    total_energy_j: float
    energy_per_gate_j: float
    passed: bool


@dataclass(frozen=True)
class SweepPoint:
    operation: str
    amplitude_v: float
    pulse_width_ns: float
    avg_switch_latency_ns: float
    worst_switch_latency_ns: float
    avg_total_energy_j: float
    avg_energy_per_gate_j: float
    cases: tuple[CaseResult, ...]


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


def set_all_plane_inits(text: str, plane: str, value: str) -> str:
    for row in [0, 1]:
        for column in [0, 1, 2, 3]:
            text = set_param(text, f"INIT_{plane}_R{row}_C{column}", value)
    return text


def logic_param(value: int) -> str:
    return "{WLOGIC1}" if value else "{WLOGIC0}"


def configure_timing(text: str, pulse_width_ns: float) -> str:
    t_end = T_PULSE_START_NS + pulse_width_ns
    t_fall = t_end + T_RAMP_NS
    t_stop = t_fall + T_SETTLE_NS
    text = set_param(text, "T_RISE_START", f"{T_RISE_START_NS:g}ns")
    text = set_param(text, "T_PULSE_START", f"{T_PULSE_START_NS:g}ns")
    text = set_param(text, "T_PULSE_END", f"{t_end:g}ns")
    text = set_param(text, "T_FALL_END", f"{t_fall:g}ns")
    text = set_param(text, "T_STOP", f"{t_stop:g}ns")
    return text


def append_driver_current_prints(text: str) -> str:
    measurements = [
        "V(drv_l3_c0)",
        "V(drv_l3_c1)",
        "V(drv_l3_c2)",
        "I(VDRV_L3_C0)",
        "I(VDRV_L3_C1)",
        "I(VDRV_L3_C2)",
    ]
    upper = text.upper()
    missing = [measurement for measurement in measurements if measurement.upper() not in upper]
    if not missing:
        return text
    return re.sub(r"^\.END\s*$", "+ " + " ".join(missing) + "\n\n.END", text, flags=re.MULTILINE | re.IGNORECASE)


def configure_l23_operation(text: str, operation: str, amplitude_v: float, a: int, b: int | None) -> str:
    text = set_all_plane_inits(text, "L01", "{WLOGIC0}")
    text = set_all_plane_inits(text, "L12", "{WLOGIC0}")
    text = set_all_plane_inits(text, "L23", "{WLOGIC0}")

    for row in [0, 1]:
        text = set_param(text, f"INIT_L23_R{row}_C0", logic_param(a))
        text = set_param(text, f"INIT_L23_R{row}_C2", "{WLOGIC1}")
        if operation == "NOR":
            if b is None:
                raise ValueError("NOR requires B")
            text = set_param(text, f"INIT_L23_R{row}_C1", logic_param(b))
        else:
            text = set_param(text, f"INIT_L23_R{row}_C1", "{WLOGIC0}")

    for plane in ["L01", "L12"]:
        for column in [0, 1, 2, 3]:
            text = set_param(text, f"ACC_{plane}_C{column}", "0")
    for column in [0, 1, 2, 3]:
        text = set_param(text, f"DIR_L0112_C{column}", "0")
        text = set_param(text, f"ACC_L23_C{column}", "0")

    text = set_param(text, "ACC_L23_C0", "1")
    text = set_param(text, "ACC_L23_C2", "1")
    if operation == "NOR":
        text = set_param(text, "ACC_L23_C1", "1")

    for layer_node in [
        "L0_R0", "L0_R1", "L1_C0", "L1_C1", "L1_C2", "L1_C3",
        "L2_R0", "L2_R1", "L3_C0", "L3_C1", "L3_C2", "L3_C3",
    ]:
        text = set_param(text, f"DRV_EN_{layer_node}", "0")
        text = set_param(text, f"V_{layer_node}", "0")

    text = set_param(text, "DRV_EN_L3_C0", "1")
    text = set_param(text, "DRV_EN_L3_C2", "1")
    text = set_param(text, "V_L3_C0", f"{amplitude_v:.9g}")
    text = set_param(text, "V_L3_C2", "0")
    if operation == "NOR":
        text = set_param(text, "DRV_EN_L3_C1", "1")
        text = set_param(text, "V_L3_C1", f"{amplitude_v:.9g}")

    return text


def make_case_deck(operation: str, amplitude_v: float, pulse_width_ns: float, a: int, b: int | None) -> Path:
    text = TEMPLATE.read_text(encoding="ascii")
    text = configure_l23_operation(text, operation, amplitude_v, a, b)
    text = configure_timing(text, pulse_width_ns)
    text = append_driver_current_prints(text)

    amp_label = f"{amplitude_v:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
    width_label = f"{pulse_width_ns:.2f}".replace(".", "p")
    b_label = "x" if b is None else str(b)
    deck_dir = GENERATED_DIR / operation.lower() / f"v{amp_label}_w{width_label}ns"
    deck_dir.mkdir(parents=True, exist_ok=True)
    deck_path = deck_dir / f"{operation.lower()}_a{a}_b{b_label}.cir"
    deck_path.write_text(text, encoding="ascii")
    return deck_path


def find_column(dataframe: pd.DataFrame, target: str) -> str:
    normalized = target.strip().upper()
    for column in dataframe.columns:
        if column.strip().upper() == normalized:
            return column
    raise KeyError(f"Could not find {target!r}; columns={list(dataframe.columns)}")


def threshold_crossing(time: np.ndarray, state: np.ndarray, start: float, threshold: float = 0.5) -> float | None:
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


def crossed_threshold(time: np.ndarray, state: np.ndarray, start: float) -> bool:
    return threshold_crossing(time, state, start) is not None


def logic_passes(value: float, expected: int) -> bool:
    if expected:
        return value >= LOGIC_HIGH_MIN
    return value <= LOGIC_LOW_MAX


def interpolate_window(time: np.ndarray, values: np.ndarray, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    mask = (time > start) & (time < end)
    window_time = np.concatenate(([start], time[mask], [end]))
    window_values = np.concatenate((
        [np.interp(start, time, values)],
        values[mask],
        [np.interp(end, time, values)],
    ))
    return window_time, window_values


def state_column(df: pd.DataFrame, row: int, column: int) -> np.ndarray:
    return pd.to_numeric(df[find_column(df, f"V(x_l23_r{row}_c{column})")], errors="coerce").to_numpy()


def source_power_abs(df: pd.DataFrame, source: str) -> np.ndarray:
    node = source.removeprefix("VDRV_").lower()
    voltage = pd.to_numeric(df[find_column(df, f"V(drv_{node})")], errors="coerce").to_numpy()
    current = pd.to_numeric(df[find_column(df, f"I({source})")], errors="coerce").to_numpy()
    return np.abs(voltage * current)


def analyze_case(
    operation: str,
    amplitude_v: float,
    pulse_width_ns: float,
    a: int,
    b: int | None,
    deck_path: Path,
) -> CaseResult:
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
    start = T_RISE_START_NS * 1e-9
    end = (T_PULSE_START_NS + pulse_width_ns + T_RAMP_NS) * 1e-9

    expected = int(not (a or b)) if operation == "NOR" else int(not a)
    output_latencies: list[float] = []
    output_ok = True
    input_ok = True
    final_inputs: list[float] = []
    final_outputs: list[float] = []

    for row in [0, 1]:
        input_states = [state_column(df, row, 0)]
        input_expected = [a]
        if operation == "NOR":
            input_states.append(state_column(df, row, 1))
            input_expected.append(0 if b is None else b)

        for state, expected_input in zip(input_states, input_expected):
            final_logic = 1 - float(state[-1])
            final_inputs.append(final_logic)
            input_ok = input_ok and logic_passes(final_logic, expected_input)
            input_ok = input_ok and not crossed_threshold(time, state, start)

        output_state = state_column(df, row, 2)
        final_output_logic = 1 - float(output_state[-1])
        final_outputs.append(final_output_logic)
        output_ok = output_ok and logic_passes(final_output_logic, expected)

        crossing = threshold_crossing(time, output_state, start)
        if expected == 1:
            output_ok = output_ok and crossing is None
        else:
            output_ok = output_ok and crossing is not None
            if crossing is not None:
                output_latencies.append((crossing - start) * 1e9)

    total_power_abs = np.zeros_like(time)
    for source in ["VDRV_L3_C0", "VDRV_L3_C1", "VDRV_L3_C2"]:
        if source in [column.strip().upper()[2:-1] for column in df.columns if column.strip().upper().startswith("I(")]:
            total_power_abs += source_power_abs(df, source)

    energy_time, power_window = interpolate_window(time, total_power_abs, start, end)
    total_energy = float(np.trapezoid(power_window, energy_time))
    passed = input_ok and output_ok

    return CaseResult(
        operation=operation,
        amplitude_v=amplitude_v,
        pulse_width_ns=pulse_width_ns,
        a=a,
        b=b,
        expected=expected,
        final_logic_inputs=tuple(final_inputs),
        final_logic_outputs=tuple(final_outputs),
        output_latencies_ns=tuple(output_latencies),
        total_energy_j=total_energy,
        energy_per_gate_j=total_energy / ROW_COUNT,
        passed=passed,
    )


def cleanup_generated_outputs(deck_path: Path) -> None:
    for suffix in [".csv", ".prn"]:
        generated = deck_path.with_suffix(deck_path.suffix + suffix)
        if generated.exists():
            generated.unlink()


def run_sweep_point(operation: str, amplitude_v: float, pulse_width_ns: float) -> SweepPoint | None:
    cases: list[CaseResult] = []
    combinations: list[tuple[int, int | None]]
    if operation == "NOR":
        combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    else:
        combinations = [(0, None), (1, None)]

    for a, b in combinations:
        deck_path = make_case_deck(operation, amplitude_v, pulse_width_ns, a, b)
        case = analyze_case(operation, amplitude_v, pulse_width_ns, a, b, deck_path)
        cases.append(case)
        cleanup_generated_outputs(deck_path)

    if not all(case.passed for case in cases):
        return None

    latencies = [latency for case in cases for latency in case.output_latencies_ns]
    return SweepPoint(
        operation=operation,
        amplitude_v=amplitude_v,
        pulse_width_ns=pulse_width_ns,
        avg_switch_latency_ns=float(np.mean(latencies)),
        worst_switch_latency_ns=float(np.max(latencies)),
        avg_total_energy_j=float(np.mean([case.total_energy_j for case in cases])),
        avg_energy_per_gate_j=float(np.mean([case.energy_per_gate_j for case in cases])),
        cases=tuple(cases),
    )


def write_outputs(valid_points: list[SweepPoint]) -> None:
    with SUMMARY_CSV.open("w", newline="", encoding="ascii") as file:
        writer = csv.writer(file)
        writer.writerow([
            "operation",
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
    valid_points: list[SweepPoint] = []
    amplitudes = [round(v, 2) for v in np.arange(-0.6, -2.01, -0.1)]
    widths = [round(w, 2) for w in np.arange(0.2, 2.01, 0.1)]

    for operation in ["NOR", "NOT"]:
        print(f"Sweeping {operation}: {len(amplitudes) * len(widths)} pulse points")
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
