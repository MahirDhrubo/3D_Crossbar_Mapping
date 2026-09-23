"""Shared JSON output helpers for the mapping CLIs (main.py, map_2d.py)."""

from __future__ import annotations

import json
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Dict, List

from mem3dmapper.mapping.state import MappingState
from mem3dmapper.mapping.types import Coordinate, Operation_type


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


def write_mapping_json(
    output_json: Path,
    state: MappingState,
    total_cycles: int,
    peak_live_cells: int,
) -> None:
    operations_count = Counter(op.type.name for op in state.ops)
    operations_count["total"] = total_cycles

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
