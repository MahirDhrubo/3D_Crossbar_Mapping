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
GENERATED_ROOT = ARTIFACT_ROOT / "generated"
RESULTS_DIR = ARTIFACT_ROOT / "results"
GENERATED_DIR = GENERATED_ROOT / "set_reset_sweep"
SUMMARY_CSV = RESULTS_DIR / "set_reset_latency_energy_summary.csv"

T_RISE_START_NS = 1.0
T_PULSE_START_NS = 1.02
T_RAMP_NS = 0.02
T_SETTLE_NS = 0.5
LOGIC_LOW_MAX = 0.1
LOGIC_HIGH_MIN = 0.9

CIRCUIT_TEMPLATE = """* Standalone {operation} latency/energy deck for MAGIC VTEAM parameters.

.INCLUDE "netlists/vteam_model.inc"

.PARAM KON=-216.2
.PARAM KOFF=0.091
.PARAM VON=-1.5
.PARAM VOFF=0.3
.PARAM WON=0
.PARAM WOFF=3n
.PARAM ALPHAON=4
.PARAM ALPHAOFF=4
.PARAM RON=1k
.PARAM ROFF=300k

.PARAM WLOGIC0={{WOFF}}
.PARAM WLOGIC1={{WON}}
.PARAM W_INIT={w_init}
.PARAM VOP={vop}
.PARAM T_RISE_START={t_rise_start:g}ns
.PARAM T_PULSE_START={t_pulse_start:g}ns
.PARAM T_PULSE_END={t_pulse_end:g}ns
.PARAM T_FALL_END={t_fall_end:g}ns
.PARAM T_STOP={t_stop:g}ns

VDRIVE drive 0 PWL(
+ 0 0
+ {{T_RISE_START}} 0
+ {{T_PULSE_START}} {{VOP}}
+ {{T_PULSE_END}} {{VOP}}
+ {{T_FALL_END}} 0
+ {{T_STOP}} 0
+)

VSENSE drive mem_p 0
XMEM mem_p 0 xstate VTEAM ron={{RON}} roff={{ROFF}} won={{WON}} woff={{WOFF}} winit={{W_INIT}} von={{VON}} voff={{VOFF}} kon={{KON}} koff={{KOFF}} alphaon={{ALPHAON}} alphaoff={{ALPHAOFF}}

.TRAN 0.1ps {{T_STOP}} 0 0.1ps
.PRINT TRAN FORMAT=CSV V(drive) V(mem_p) V(xstate) I(VDRIVE) I(VSENSE)

.END
"""


@dataclass(frozen=True)
class Point:
    operation: str
    amplitude_v: float
    pulse_width_ns: float
    initial_logic: float
    final_logic: float
    latency_ns: float
    energy_j: float
    final_state: float


def xyce_cmd() -> str:
    configured = os.environ.get("XYCE_BIN")
    if configured:
        return configured
    found = shutil.which("Xyce")
    if found:
        return found
    raise FileNotFoundError("Could not find Xyce. Set XYCE_BIN or put Xyce on PATH.")


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
    text = CIRCUIT_TEMPLATE.format(
        operation=operation,
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


def interpolate_window(time: np.ndarray, values: np.ndarray, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    mask = (time > start) & (time < end)
    window_time = np.concatenate(([start], time[mask], [end]))
    window_values = np.concatenate((
        [np.interp(start, time, values)],
        values[mask],
        [np.interp(end, time, values)],
    ))
    return window_time, window_values


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
    widths = [0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]

    valid_points: list[Point] = []
    for operation, amplitudes in specs:
        print(f"Sweeping {operation}: {len(amplitudes) * len(widths)} pulse points")
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
