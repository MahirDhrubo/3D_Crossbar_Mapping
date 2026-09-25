"""
Map a BLIF netlist onto a 2D crossbar using the sequential NOR-NOT-only
heuristic (mem3dmapper.mapping.heuristic.map_2D), and write the resulting
execution trace -- per-cycle operations, per-gate-type counts, and peak
live cell count -- to a JSON file, in the same format as cli/main.py's 3D
mapper (consumable by cli/energy_latency_calculator.py).

Only --columns is a real constraint here: it's the fixed row width (the 2D
architectural limit). The mapper starts with a single row and appends more
as needed, so there is no --rows option -- the row count is an output, not
an input.

mem3dmapper.mapping.validator is not run here: its geometry and
concurrent-scheduling checks are written for the 3D crossbar model and do
not apply to a 2D mapping.

Run: python3 map_2d.py <input.blif> <output.json> [options]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mem3dmapper.cli._mapping_io import count_cell_usage, count_peak_live_cells, write_mapping_json
from mem3dmapper.mapping.heuristic import map_2D
from mem3dmapper.mapping.state import MappingState
from mem3dmapper.mapping.types import MappingConfig
from mem3dmapper.netlist.parser import parse_netlist


def run(
    blif_path: Path,
    output_json: Path,
    total_columns: int,
) -> MappingState:
    config = MappingConfig(total_rows=1, total_columns=total_columns)
    netlist = parse_netlist(str(blif_path))
    print(f"{netlist.name}: {len(netlist.gates)} gates")

    state = map_2D(netlist=netlist, config=config)
    if state is None:
        raise RuntimeError(
            f"Mapping failed for grid size {config.total_rows}x{config.total_columns}"
        )

    cell_usage = count_cell_usage(state)
    peak_live_cells = count_peak_live_cells(state)
    print(
        f"Grid size {config.total_rows}x{config.total_columns}  "
        f"total cycles: {state.cycle_count}  cell usage: {cell_usage}  "
        f"peak live cells: {peak_live_cells}"
    )

    # 2D is a sequential, one-gate-per-cycle model: total cycles is the sum
    # of real (non-WRITE) operations, not state.cycle_count (which the
    # concurrent 3D mapper uses to reflect multiple gates sharing a cycle).
    total_ops = sum(1 for op in state.ops if op.type.name != "WRITE")
    write_mapping_json(output_json, state, total_cycles=total_ops, peak_live_cells=peak_live_cells)

    return state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map a BLIF netlist onto a 2D crossbar (sequential NOR-NOT-only "
        "heuristic) and write the execution trace to a JSON file."
    )
    parser.add_argument("blif_path", type=Path, help="Input BLIF netlist file (NOR-NOT-only synthesis).")
    parser.add_argument("output_json", type=Path, help="Path to write the mapping-result JSON file.")
    parser.add_argument("--columns", type=int, default=512, help="Crossbar row width (default: 512).")
    args = parser.parse_args()

    run(args.blif_path, args.output_json, args.columns)


if __name__ == "__main__":
    main()
