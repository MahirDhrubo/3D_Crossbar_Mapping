from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from sweep_set_reset_latency_energy import (
    CIRCUIT_TEMPLATE,
    GENERATED_ROOT,
    LOGIC_HIGH_MIN,
    LOGIC_LOW_MAX,
    PROJECT_ROOT,
    RESULTS_DIR,
    T_PULSE_START_NS,
    T_RAMP_NS,
    T_RISE_START_NS,
    T_SETTLE_NS,
    Point,
    find_column,
    interpolate_window,
    threshold_crossing,
    xyce_cmd,
)


GENERATED_DIR = GENERATED_ROOT / "set_reset_floatpim_ron_roff_sweep"
SUMMARY_CSV = RESULTS_DIR / "set_reset_floatpim_ron_roff_latency_energy_summary.csv"

FLOATPIM_TEMPLATE = (
    CIRCUIT_TEMPLATE
    .replace(".PARAM RON=1k", ".PARAM RON=10k")
    .replace(".PARAM ROFF=300k", ".PARAM ROFF=10Meg")
)


def make_deck(operation: str, amplitude_v: float, pulse_width_ns: float) -> Path:
    if operation == "SET":
        w_init = "{WLOGIC0}"
    elif operation == "RESET":
        w_init = "{WLOGIC1}"
    else:
        raise ValueError(operation)

    t_pulse_end = T_PULSE_START_NS + pulse_width_ns
    t_fall_end = t_pulse_end + T_RAMP_NS
    t_stop = t_fall_end + T_SETTLE_NS
    text = FLOATPIM_TEMPLATE.format(
        operation=f"{operation} with FloatPIM RON/ROFF",
        w_init=w_init,
        vop=f"{amplitude_v:.9g}",
        t_rise_start=T_RISE_START_NS,
        t_pulse_start=T_PULSE_START_NS,
        t_pulse_end=t_pulse_end,
        t_fall_end=t_fall_end,
        t_stop=t_stop,
    )

    amp_label = f"{amplitude_v:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
    width_label = f"{pulse_width_ns:.3f}".replace(".", "p")
    deck_dir = GENERATED_DIR / operation.lower()
    deck_dir.mkdir(parents=True, exist_ok=True)
    deck_path = deck_dir / f"{operation.lower()}_v{amp_label}_w{width_label}ns.cir"
    deck_path.write_text(text, encoding="ascii")
    return deck_path


def analyze_deck(operation: str, amplitude_v: float, pulse_width_ns: float, deck_path: Path) -> Point | None:
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
    drive = pd.to_numeric(df[find_column(df, "V(DRIVE)")], errors="coerce").to_numpy()
    state = pd.to_numeric(df[find_column(df, "V(XSTATE)")], errors="coerce").to_numpy()
    current = pd.to_numeric(df[find_column(df, "I(VDRIVE)")], errors="coerce").to_numpy()

    start = T_RISE_START_NS * 1e-9
    end = (T_PULSE_START_NS + pulse_width_ns + T_RAMP_NS) * 1e-9
    crossing = threshold_crossing(time, state, start)
    if crossing is None:
        return None

    final_state = float(state[-1])
    final_logic = 1 - final_state
    if operation == "SET":
        passed = final_logic >= LOGIC_HIGH_MIN
    else:
        passed = final_logic <= LOGIC_LOW_MAX
    if not passed:
        return None

    energy_time, power_abs = interpolate_window(time, np.abs(drive * current), start, end)
    return Point(
        operation=operation,
        amplitude_v=amplitude_v,
        pulse_width_ns=pulse_width_ns,
        initial_logic=1 - float(state[0]),
        final_logic=final_logic,
        latency_ns=(crossing - start) * 1e9,
        energy_j=float(np.trapezoid(power_abs, energy_time)),
        final_state=final_state,
    )


def run_point(operation: str, amplitude_v: float, pulse_width_ns: float) -> Point | None:
    deck_path = make_deck(operation, amplitude_v, pulse_width_ns)
    try:
        return analyze_deck(operation, amplitude_v, pulse_width_ns, deck_path)
    finally:
        for suffix in [".csv", ".prn"]:
            generated = deck_path.with_suffix(deck_path.suffix + suffix)
            if generated.exists():
                generated.unlink()


def write_summary(points: list[Point]) -> None:
    with SUMMARY_CSV.open("w", newline="", encoding="ascii") as file:
        writer = csv.writer(file)
        writer.writerow([
            "operation",
            "ron",
            "roff",
            "amplitude_v",
            "pulse_width_ns",
            "initial_logic",
            "final_logic",
            "final_state",
            "latency_ns",
            "energy_j",
        ])
        for point in points:
            writer.writerow([
                point.operation,
                "10k",
                "10Meg",
                f"{point.amplitude_v:.9g}",
                f"{point.pulse_width_ns:.9g}",
                f"{point.initial_logic:.9g}",
                f"{point.final_logic:.9g}",
                f"{point.final_state:.9g}",
                f"{point.latency_ns:.9g}",
                f"{point.energy_j:.9e}",
            ])


def main() -> None:
    specs = [
        ("SET", [round(v, 2) for v in np.arange(-1.55, -3.01, -0.05)]),
        ("RESET", [round(v, 2) for v in np.arange(0.35, 3.01, 0.05)]),
    ]
    widths = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]

    valid_points: list[Point] = []
    for operation, amplitudes in specs:
        print(f"Sweeping {operation} with FloatPIM RON/ROFF: {len(amplitudes) * len(widths)} pulse points")
        for amplitude_v in amplitudes:
            for pulse_width_ns in widths:
                point = run_point(operation, amplitude_v, pulse_width_ns)
                if point is not None:
                    valid_points.append(point)

    write_summary(valid_points)
    print(SUMMARY_CSV)

    for operation in ["SET", "RESET"]:
        points = [point for point in valid_points if point.operation == operation]
        if not points:
            print(f"{operation}: no valid pulse points")
            continue
        fastest = min(points, key=lambda point: (point.latency_ns, point.energy_j))
        lowest_energy = min(points, key=lambda point: (point.energy_j, point.latency_ns))
        print(
            f"{operation} fastest valid: VOP={fastest.amplitude_v:g} V "
            f"width={fastest.pulse_width_ns:g} ns "
            f"latency={fastest.latency_ns:.6g} ns "
            f"energy={fastest.energy_j:.6e} J "
            f"final_logic={fastest.final_logic:.6g}"
        )
        print(
            f"{operation} lowest-energy valid: VOP={lowest_energy.amplitude_v:g} V "
            f"width={lowest_energy.pulse_width_ns:g} ns "
            f"latency={lowest_energy.latency_ns:.6g} ns "
            f"energy={lowest_energy.energy_j:.6e} J "
            f"final_logic={lowest_energy.final_logic:.6g}"
        )


if __name__ == "__main__":
    main()
