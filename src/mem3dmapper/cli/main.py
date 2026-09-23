"""
Map a BLIF netlist onto the 3D crossbar using PRISM's concurrent
level-based scheduler (mem3dmapper.mapping.parallel_mapping), and write
the resulting execution trace -- per-cycle operations, per-gate-type
counts, and peak live cell count -- to a JSON file.

Run: python3 main.py <input.blif> <output.json> [options]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from mem3dmapper.dag.build import build_DAG
from mem3dmapper.dag.visualize import render_graphviz, write_dot
from mem3dmapper.mapping.parallel_mapping import map_netlist
from mem3dmapper.mapping.state import MappingState
from mem3dmapper.mapping.types import Coordinate, MappingConfig, Operation_type
from mem3dmapper.mapping.validator import validate_mapping
from mem3dmapper.netlist.parser import parse_netlist

LARGE_NETLIST_GATE_THRESHOLD = 500


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


def count_cell_usage(state: MappingState) -> int:
    """Number of distinct crossbar cells occupied over the whole mapping."""
    return len({op.location for op in state.ops})


def count_peak_live_cells(state: MappingState) -> int:
    """Peak number of cells simultaneously holding a value with future uses."""
    operations_by_cycle: Dict[int, List] = {}
    for op in state.ops:
        operations_by_cycle.setdefault(op.cycle, []).append(op)

    future_uses = Counter()
    for op in state.ops:
        if op.type == Operation_type.COPY:
            future_uses[op.net] += 1
        elif op.inputs:
            future_uses.update(op.inputs)

    occupancy: Dict[Coordinate, str] = {}
    output_nets = set(state.primary_outputs)
    peak_live_cells = 0

    def live_cell_count() -> int:
        return sum(
            1
            for net in occupancy.values()
            if future_uses.get(net, 0) > 0 or net in output_nets
        )

    for cycle in sorted(operations_by_cycle):
        peak_live_cells = max(peak_live_cells, live_cell_count())

        for op in operations_by_cycle[cycle]:
            if op.type == Operation_type.COPY:
                future_uses[op.net] -= 1
            elif op.inputs:
                future_uses.subtract(op.inputs)

        for op in operations_by_cycle[cycle]:
            occupancy[op.location] = op.net

        peak_live_cells = max(peak_live_cells, live_cell_count())

    return peak_live_cells


def run(
    blif_path: Path,
    output_json: Path,
    config: MappingConfig,
    *,
    validate: bool = True,
    dag_dot: Optional[Path] = None,
    dag_image: Optional[Path] = None,
) -> MappingState:
    netlist = parse_netlist(str(blif_path))
    print(f"{netlist.name}: {len(netlist.gates)} gates")

    if dag_dot is not None:
        if len(netlist.gates) > LARGE_NETLIST_GATE_THRESHOLD:
            print(
                f"Warning: netlist has {len(netlist.gates)} gates "
                f"(> {LARGE_NETLIST_GATE_THRESHOLD}); DAG rendering may be slow/large."
            )
        parents, children = build_DAG(netlist)
        write_dot(dag_dot, netlist, parents, children)
        if dag_image is not None:
            render_graphviz(dag_dot, dag_image, fmt=dag_image.suffix.lstrip(".") or "png")

    state = map_netlist(netlist=netlist, config=config)
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

    if validate:
        validate_mapping(netlist, state)

    operations_count = Counter(op.type.name for op in state.ops)
    operations_count["total"] = state.cycle_count

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(
            {
                "execution": state.ops,
                "operations_count": operations_count,
                "peak_live_cells": peak_live_cells,
            },
            f,
            indent=4,
            cls=EnumEncoder,
        )
    print(f"Wrote {output_json}")

    return state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map a BLIF netlist onto the 3D crossbar (PRISM's concurrent "
        "level-based scheduler) and write the execution trace to a JSON file."
    )
    parser.add_argument("blif_path", type=Path, help="Input BLIF netlist file.")
    parser.add_argument("output_json", type=Path, help="Path to write the mapping-result JSON file.")
    parser.add_argument("--rows", type=int, default=6, help="Crossbar rows / stacked layers (default: 6).")
    parser.add_argument("--columns", type=int, default=512, help="Crossbar columns (default: 512).")
    parser.add_argument("--cluster-window", type=int, default=21, help="Cost-model cluster window (default: 21).")
    parser.add_argument("--alpha", type=float, default=0.6, help="Cost-model weight alpha (default: 0.6).")
    parser.add_argument("--beta", type=float, default=0.4, help="Cost-model weight beta (default: 0.4).")
    parser.add_argument("--gamma", type=float, default=0.0, help="Cost-model weight gamma (default: 0.0).")
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip validating the mapping against the netlist."
    )
    parser.add_argument(
        "--dag-dot", type=Path, default=None, help="Optional path to write the netlist's DAG as a .dot file."
    )
    parser.add_argument(
        "--dag-image",
        type=Path,
        default=None,
        help="Optional path to render the DAG image (requires --dag-dot; "
        "format inferred from the file extension, e.g. .png/.svg).",
    )
    args = parser.parse_args()

    if args.dag_image is not None and args.dag_dot is None:
        parser.error("--dag-image requires --dag-dot")

    config = MappingConfig(
        total_rows=args.rows,
        total_columns=args.columns,
        cluster_window=args.cluster_window,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )

    run(
        args.blif_path,
        args.output_json,
        config,
        validate=not args.skip_validation,
        dag_dot=args.dag_dot,
        dag_image=args.dag_image,
    )


if __name__ == "__main__":
    main()
