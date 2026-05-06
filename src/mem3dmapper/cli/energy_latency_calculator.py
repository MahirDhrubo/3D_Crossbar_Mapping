"""
Generic energy/latency calculator for a MAGIC gate-level computation.

Takes a mapping-result JSON (the format written by the mapper in
cli/main.py: a dict with an "operations_count" entry holding per-gate-type
counts and a total cycle count) and computes the total execution energy
and latency from fixed, confirmed device-level parameters.

Device parameters (fJ per operation, ns per cycle) are confirmed values.
NOT is assumed equal to NOR, and COPY is assumed equal to AND, since a
MAGIC NOT shares NOR's parallel-connection mechanism and a COPY is
implemented as a single-input AND. NOR/NOT operations are SET-initialized;
AND/COPY operations are RESET-initialized. WRITE operations (primary-input
initialization) are excluded from the energy/latency totals.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

E_SET_FJ = 0.113
E_RESET_FJ = 0.364
E_NOR_FJ = 1.29
E_AND_FJ = 0.097
E_NOT_FJ = E_NOR_FJ
E_COPY_FJ = E_AND_FJ
CYCLE_TIME_NS = 0.177

GATE_TYPES: Tuple[str, ...] = ("NOR", "NOT", "AND", "COPY")

EXECUTION_ENERGY_FJ: Dict[str, float] = {
    "NOR": E_NOR_FJ,
    "NOT": E_NOT_FJ,
    "AND": E_AND_FJ,
    "COPY": E_COPY_FJ,
}
INIT_ENERGY_FJ: Dict[str, float] = {
    "NOR": E_SET_FJ,
    "NOT": E_SET_FJ,
    "AND": E_RESET_FJ,
    "COPY": E_RESET_FJ,
}


@dataclass
class EnergyLatencyResult:
    gate_counts: Dict[str, int]
    total_cycles: int
    cycle_time_ns: float
    energy_per_gate_pj: Dict[str, float]
    total_energy_pj: float
    total_latency_ns: float

    @property
    def total_latency_us(self) -> float:
        return self.total_latency_ns / 1000.0


def load_gate_counts(json_path: Union[str, Path]) -> Tuple[Dict[str, int], int]:
    """Load per-gate-type counts and total cycle count from a mapping-result JSON.

    Accepts either the mapper's own output format
    ({"operations_count": {"NOR": .., "NOT": .., "AND": .., "COPY": ..,
    "total": ..}, ...}) or a flat dict using the same keys.
    """
    with open(json_path) as f:
        data = json.load(f)

    counts = data["operations_count"] if "operations_count" in data else data

    gate_counts = {gate: int(counts.get(gate, 0)) for gate in GATE_TYPES}
    total_cycles = int(counts.get("total", sum(gate_counts.values())))
    return gate_counts, total_cycles


def calculate(
    gate_counts: Dict[str, int],
    total_cycles: int,
    cycle_time_ns: float = CYCLE_TIME_NS,
) -> EnergyLatencyResult:
    energy_per_gate_pj = {
        gate: gate_counts.get(gate, 0) * (EXECUTION_ENERGY_FJ[gate] + INIT_ENERGY_FJ[gate]) / 1000.0
        for gate in GATE_TYPES
    }
    total_energy_pj = sum(energy_per_gate_pj.values())
    total_latency_ns = total_cycles * cycle_time_ns

    return EnergyLatencyResult(
        gate_counts=gate_counts,
        total_cycles=total_cycles,
        cycle_time_ns=cycle_time_ns,
        energy_per_gate_pj=energy_per_gate_pj,
        total_energy_pj=total_energy_pj,
        total_latency_ns=total_latency_ns,
    )


def print_result(result: EnergyLatencyResult) -> None:
    print("Gate counts:")
    for gate in GATE_TYPES:
        print(f"  {gate:5s}: {result.gate_counts.get(gate, 0)}")
    print(f"Total cycles : {result.total_cycles}")
    print()
    print("Energy breakdown (pJ):")
    for gate in GATE_TYPES:
        print(f"  {gate:5s}: {result.energy_per_gate_pj[gate]:.4f}")
    print(f"Total energy  : {result.total_energy_pj:.4f} pJ")
    print(f"Total latency : {result.total_latency_ns:.3f} ns  ({result.total_latency_us:.4f} us)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute total energy and latency for a MAGIC gate-level "
        "computation from its mapping-result JSON."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to a mapping-result JSON file (contains per-gate-type operation counts).",
    )
    parser.add_argument(
        "--cycle-time-ns",
        type=float,
        default=CYCLE_TIME_NS,
        help=f"Per-cycle time in ns (default: {CYCLE_TIME_NS}).",
    )
    args = parser.parse_args()

    gate_counts, total_cycles = load_gate_counts(args.json_path)
    result = calculate(gate_counts, total_cycles, args.cycle_time_ns)
    print_result(result)


if __name__ == "__main__":
    main()
