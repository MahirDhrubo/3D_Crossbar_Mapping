"""
Map a BLIF netlist onto the 3D crossbar using PRISM's concurrent
level-based scheduler (mem3dmapper.mapping.parallel_mapping), and write
the resulting execution trace -- per-cycle operations, per-gate-type
counts, and peak live cell count -- to a JSON file.

Run: python3 main.py <input.blif> <output.json> [options]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from mem3dmapper.cli._mapping_io import count_cell_usage, count_peak_live_cells, write_mapping_json
from mem3dmapper.dag.build import build_DAG
from mem3dmapper.dag.visualize import render_graphviz, write_dot
from mem3dmapper.mapping.parallel_mapping import map_netlist
from mem3dmapper.mapping.state import MappingState
from mem3dmapper.mapping.types import MappingConfig
from mem3dmapper.mapping.validator import validate_mapping
from mem3dmapper.netlist.parser import parse_netlist

LARGE_NETLIST_GATE_THRESHOLD = 500


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

    write_mapping_json(output_json, state, total_cycles=state.cycle_count, peak_live_cells=peak_live_cells)

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
