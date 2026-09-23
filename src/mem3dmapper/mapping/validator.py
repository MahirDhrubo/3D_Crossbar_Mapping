"""Independent validation of a mapped MAGIC execution.

The mapper's final dictionaries are not treated as proof of correctness.  This
module replays the operation log cycle by cycle and checks that the replayed
machine state agrees with the reported :class:`MappingState`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Dict, Iterable, List, Mapping, Sequence

from mem3dmapper.mapping.state import MappingState
from mem3dmapper.mapping.types import Coordinate, Operation, Operation_type
from mem3dmapper.netlist.types import Gate, GateType, Netlist


class MappingValidationError(RuntimeError):
    """Raised when a mapping or its execution schedule is inconsistent."""


_GATE_OPERATION_TYPES = {
    Operation_type.NOT,
    Operation_type.NOR,
    Operation_type.AND,
}

_EXPECTED_OPERATION_TYPE = {
    GateType.NOT: Operation_type.NOT,
    GateType.NOR: Operation_type.NOR,
    GateType.AND: Operation_type.AND,
}


def _error(message: str, *, cycle: int | None = None, index: int | None = None) -> None:
    context = []
    if cycle is not None:
        context.append(f"cycle {cycle}")
    if index is not None:
        context.append(f"operation {index}")
    prefix = f"[{', '.join(context)}] " if context else ""
    raise MappingValidationError(prefix + message)


def _validate_coordinate(
    coordinate: Coordinate,
    state: MappingState,
    *,
    label: str,
    cycle: int | None = None,
    index: int | None = None,
) -> None:
    if (
        not isinstance(coordinate, tuple)
        or len(coordinate) != 2
        or any(type(value) is not int for value in coordinate)
    ):
        _error(f"{label} must be a pair of integer coordinates; got {coordinate!r}.", cycle=cycle, index=index)

    column, row = coordinate
    if not (0 <= column < state.config.total_columns):
        _error(
            f"{label} column {column} is outside [0, {state.config.total_columns - 1}].",
            cycle=cycle,
            index=index,
        )
    if not (0 <= row < state.config.total_rows):
        _error(
            f"{label} row {row} is outside [0, {state.config.total_rows - 1}].",
            cycle=cycle,
            index=index,
        )


def _validate_netlist(netlist: Netlist) -> tuple[Dict[str, Gate], Counter[str]]:
    if len(netlist.primary_inputs) != len(set(netlist.primary_inputs)):
        _error("The netlist contains duplicate primary-input names.")
    if len(netlist.primary_outputs) != len(set(netlist.primary_outputs)):
        _error("The netlist contains duplicate primary-output names.")

    gates_by_output: Dict[str, Gate] = {}
    gate_ids = set()
    for gate in netlist.gates:
        if gate.gid in gate_ids:
            _error(f"Gate id {gate.gid} is not unique.")
        gate_ids.add(gate.gid)

        if gate.output in gates_by_output or gate.output in netlist.primary_inputs:
            _error(f"Net {gate.output!r} has more than one producer.")
        gates_by_output[gate.output] = gate

        if gate.type not in _EXPECTED_OPERATION_TYPE:
            _error(
                f"Gate {gate.gid} uses {gate.type.name}, but the current operation model "
                "can execute only NOT, NOR, and AND."
            )
        if gate.type == GateType.NOT and len(gate.inputs) != 1:
            _error(f"NOT gate {gate.gid} must have exactly one input.")
        if gate.type in {GateType.NOR, GateType.AND} and len(gate.inputs) < 2:
            _error(f"{gate.type.name} gate {gate.gid} must have at least two inputs.")

    defined_nets = set(netlist.primary_inputs) | set(gates_by_output)
    for gate in netlist.gates:
        for input_net in gate.inputs:
            if input_net not in defined_nets:
                _error(f"Gate {gate.gid} consumes undefined net {input_net!r}.")

    for output_net in netlist.primary_outputs:
        if output_net not in defined_nets:
            _error(f"Primary output {output_net!r} has no producer.")

    fanout = Counter(input_net for gate in netlist.gates for input_net in gate.inputs)
    return gates_by_output, fanout


def _validate_operation_metadata(
    operation: Operation,
    state: MappingState,
    *,
    cycle: int,
    index: int,
) -> None:
    if not isinstance(operation.type, Operation_type):
        _error(f"Unknown operation type {operation.type!r}.", cycle=cycle, index=index)
    if not isinstance(operation.net, str) or not operation.net:
        _error("Operation net must be a non-empty string.", cycle=cycle, index=index)
    _validate_coordinate(operation.location, state, label="Destination", cycle=cycle, index=index)

    if operation.type == Operation_type.WRITE:
        if operation.inputs not in (None, ()):
            _error("WRITE cannot declare logic inputs.", cycle=cycle, index=index)
        if operation.src not in (None, []):
            _error("WRITE cannot declare source cells.", cycle=cycle, index=index)
        return

    if operation.type == Operation_type.COPY:
        if operation.inputs not in (None, ()):
            _error("COPY cannot declare logic inputs.", cycle=cycle, index=index)
        if operation.src is None or len(operation.src) != 1:
            _error("COPY must declare exactly one source cell.", cycle=cycle, index=index)
    elif operation.type in _GATE_OPERATION_TYPES:
        if operation.inputs is None:
            _error("A gate operation must record its input nets.", cycle=cycle, index=index)
        if operation.src is None:
            _error("A gate operation must record its input cells.", cycle=cycle, index=index)

    for source_index, source in enumerate(operation.src or []):
        _validate_coordinate(
            source,
            state,
            label=f"Source {source_index}",
            cycle=cycle,
            index=index,
        )


def _validate_parallel_geometry(operation: Operation, *, cycle: int, index: int) -> None:
    sources = operation.src or []
    cells = sources + [operation.location]
    rows = {row for _, row in cells}
    if len(rows) != 1:
        _error(
            f"{operation.type.name} inputs and output must occupy one row; got {cells!r}.",
            cycle=cycle,
            index=index,
        )
    if len(set(cells)) != len(cells):
        _error(
            f"{operation.type.name} inputs and output must use distinct cells.",
            cycle=cycle,
            index=index,
        )


def _validate_series_geometry(operation: Operation, *, cycle: int, index: int) -> None:
    sources = operation.src or []
    source_columns = {column for column, _ in sources}
    output_column, output_row = operation.location
    if source_columns != {output_column}:
        _error(
            f"AND inputs and output must occupy one column; got sources {sources!r} "
            f"and output {operation.location!r}.",
            cycle=cycle,
            index=index,
        )

    input_rows = [row for _, row in sources]
    if len(set(input_rows)) != len(input_rows):
        _error("AND input cells must be distinct.", cycle=cycle, index=index)
    sorted_rows = sorted(input_rows)
    expected_rows = list(range(sorted_rows[0], sorted_rows[0] + len(sorted_rows)))
    if sorted_rows != expected_rows:
        _error(
            f"AND input rows must be contiguous; got {sorted_rows!r}.",
            cycle=cycle,
            index=index,
        )
    if output_row not in {sorted_rows[0] - 1, sorted_rows[-1] + 1}:
        _error(
            "AND output must be immediately above or below its contiguous input window.",
            cycle=cycle,
            index=index,
        )


def _series_footprint(operation: Operation) -> tuple[int, int, int]:
    rows = [row for _, row in (operation.src or [])] + [operation.location[1]]
    return min(rows), max(rows), operation.location[1]


def _validate_concurrent_gate_conflicts(
    indexed_operations: Sequence[tuple[int, Operation]],
    *,
    cycle: int,
) -> None:
    """Validate the paper's blocked-row rules for one execution cycle."""

    for left_position, (_, left) in enumerate(indexed_operations):
        for _, right in indexed_operations[left_position + 1 :]:
            left_parallel = left.type in {Operation_type.NOT, Operation_type.NOR}
            right_parallel = right.type in {Operation_type.NOT, Operation_type.NOR}

            if left_parallel and right_parallel:
                if abs(left.location[1] - right.location[1]) <= 1:
                    _error(
                        f"Concurrent parallel gates {left.net!r} and {right.net!r} use "
                        "the same or adjacent rows.",
                        cycle=cycle,
                    )
                continue

            if left_parallel != right_parallel:
                parallel = left if left_parallel else right
                series = right if left_parallel else left
                series_start, series_end, _ = _series_footprint(series)
                if series_start - 1 <= parallel.location[1] <= series_end + 1:
                    _error(
                        f"Concurrent gates {left.net!r} and {right.net!r} violate the "
                        "parallel/series blocked-row spacing.",
                        cycle=cycle,
                    )
                continue

            left_start, left_end, left_output_row = _series_footprint(left)
            right_start, right_end, right_output_row = _series_footprint(right)
            same_window_and_direction = (
                left_start == right_start
                and left_end == right_end
                and left_output_row == right_output_row
            )
            if same_window_and_direction:
                continue
            if not (left_end + 1 < right_start or right_end + 1 < left_start):
                _error(
                    f"Concurrent series gates {left.net!r} and {right.net!r} have "
                    "overlapping or adjacent row windows without reusable-window alignment.",
                    cycle=cycle,
                )


def _remove_cell(
    occupancy: Dict[Coordinate, str],
    locations: Dict[str, set[Coordinate]],
    coordinate: Coordinate,
) -> None:
    previous_net = occupancy.get(coordinate)
    if previous_net is not None:
        locations[previous_net].discard(coordinate)


def _place_cell(
    occupancy: Dict[Coordinate, str],
    locations: Dict[str, set[Coordinate]],
    coordinate: Coordinate,
    net: str,
) -> None:
    _remove_cell(occupancy, locations, coordinate)
    occupancy[coordinate] = net
    locations[net].add(coordinate)


def _normalise_locations(locations: Mapping[str, Iterable[Coordinate]]) -> Dict[str, set[Coordinate]]:
    return {net: set(cells) for net, cells in locations.items() if cells}


def _validate_reported_state(
    netlist: Netlist,
    state: MappingState,
    occupancy: Dict[Coordinate, str],
    locations: Dict[str, set[Coordinate]],
    write_locations: Dict[str, Coordinate],
    initial_output_locations: Dict[str, Coordinate],
    remaining_uses: Counter[str],
) -> None:
    if state.location_to_net != occupancy:
        _error("state.location_to_net does not match the independently replayed execution.")

    reported_locations = _normalise_locations(state.net_location)
    replayed_locations = _normalise_locations(locations)
    if reported_locations != replayed_locations:
        _error("state.net_location does not match the independently replayed execution.")

    if state.primary_input_locations != write_locations:
        _error("state.primary_input_locations does not match the validated WRITE operations.")

    for output_net in netlist.primary_outputs:
        cells = replayed_locations.get(output_net, set())
        initial_cell = initial_output_locations.get(output_net)
        if initial_cell is None:
            _error(f"Primary output {output_net!r} has no initial creation location.")
        if initial_cell not in cells:
            _error(
                f"Primary output {output_net!r} lost its initial cell {initial_cell!r}."
            )
    if state.primary_output_locations != initial_output_locations:
        _error(
            "state.primary_output_locations must contain only each output's initial creation location."
        )

    if state.primary_outputs != netlist.primary_outputs:
        _error("MappingState.primary_outputs differs from the netlist primary outputs.")

    write_count = sum(operation.type == Operation_type.WRITE for operation in state.ops)
    if state.number_of_writes != write_count:
        _error(
            f"number_of_writes is {state.number_of_writes}, but the schedule contains {write_count} WRITE operations."
        )

    expected_cycle_count = max((operation.cycle for operation in state.ops), default=0)
    if state.cycle_count != expected_cycle_count:
        _error(
            f"cycle_count is {state.cycle_count}, but the last scheduled cycle is {expected_cycle_count}."
        )

    known_nets = set(netlist.primary_inputs) | {gate.output for gate in netlist.gates}
    for net in known_nets | set(state.uses_left):
        expected = remaining_uses.get(net, 0)
        reported = state.uses_left.get(net, 0)
        if reported != expected:
            _error(f"uses_left[{net!r}] is {reported}, expected {expected} after replay.")

    if len(state.tail_x) != state.config.total_rows:
        _error("tail_x length must equal the configured row count.")
    for row, tail in enumerate(state.tail_x):
        if type(tail) is not int or not (0 <= tail <= state.config.total_columns):
            _error(f"tail_x[{row}]={tail!r} is outside the crossbar.")
        occupied_columns = [column for (column, cell_row) in occupancy if cell_row == row]
        if occupied_columns and tail <= max(occupied_columns):
            _error(
                f"tail_x[{row}]={tail} does not cover occupied column {max(occupied_columns)}."
            )

    if len(state.row_cluster) != state.config.total_rows:
        _error("row_cluster length must equal the configured row count.")
    if not isinstance(state.total_cost, (int, float)) or not math.isfinite(state.total_cost):
        _error(f"total_cost must be finite; got {state.total_cost!r}.")


def validate_execution(netlist: Netlist, state: MappingState) -> None:
    """Replay and validate every operation in a mapping schedule.

    Sources are read from the state at the beginning of a cycle.  Destinations
    become visible only after that cycle, so a COPY or gate result cannot be
    consumed in the same cycle that creates it.
    """

    gates_by_output, initial_fanout = _validate_netlist(netlist)
    primary_inputs = set(netlist.primary_inputs)
    primary_outputs = set(netlist.primary_outputs)

    operations_by_cycle: Dict[int, List[tuple[int, Operation]]] = defaultdict(list)
    previous_cycle = -1
    for index, operation in enumerate(state.ops):
        if type(operation.cycle) is not int or operation.cycle < 0:
            _error(f"Operation cycle must be a non-negative integer; got {operation.cycle!r}.", index=index)
        if operation.cycle < previous_cycle:
            _error("Operations are not ordered by nondecreasing cycle.", cycle=operation.cycle, index=index)
        previous_cycle = operation.cycle
        operations_by_cycle[operation.cycle].append((index, operation))

    occupancy: Dict[Coordinate, str] = {}
    locations: Dict[str, set[Coordinate]] = defaultdict(set)
    remaining_uses: Counter[str] = Counter(initial_fanout)
    written_inputs: set[str] = set()
    write_locations: Dict[str, Coordinate] = {}
    reserved_input_cells: set[Coordinate] = set()
    reserved_output_cells: set[Coordinate] = set()
    initial_output_locations: Dict[str, Coordinate] = {}
    executed_outputs: set[str] = set()

    for cycle in sorted(operations_by_cycle):
        indexed_operations = operations_by_cycle[cycle]
        snapshot = dict(occupancy)
        destinations: Dict[Coordinate, int] = {}
        gate_operations: List[tuple[int, Operation]] = []
        copy_operations: List[tuple[int, Operation]] = []

        for index, operation in indexed_operations:
            _validate_operation_metadata(operation, state, cycle=cycle, index=index)

        cycle_destination_cells = {
            operation.location for _, operation in indexed_operations
        }
        participating_nets = {
            snapshot[source]
            for _, operation in indexed_operations
            for source in (operation.src or [])
            if source in snapshot
        }

        for index, operation in indexed_operations:
            if operation.location in destinations:
                _error(
                    f"Destination {operation.location!r} is also written by operation "
                    f"{destinations[operation.location]} in this cycle.",
                    cycle=cycle,
                    index=index,
                )
            destinations[operation.location] = index

            previous_net = snapshot.get(operation.location)
            if previous_net is not None:
                if operation.location in reserved_input_cells:
                    _error(
                        f"Destination {operation.location!r} is a reserved primary-input cell.",
                        cycle=cycle,
                        index=index,
                    )
                if operation.location in reserved_output_cells:
                    _error(
                        f"Destination {operation.location!r} is a retained primary-output cell.",
                        cycle=cycle,
                        index=index,
                    )
                surviving_copies = (
                    locations.get(previous_net, set()) - cycle_destination_cells
                )
                can_replace_live_copy = (
                    previous_net not in participating_nets
                    and bool(surviving_copies)
                )
                if (
                    remaining_uses.get(previous_net, 0) > 0
                    and not can_replace_live_copy
                ):
                    _error(
                        f"Destination {operation.location!r} overwrites live net {previous_net!r} "
                        f"with {remaining_uses[previous_net]} use(s) remaining; replacement "
                        "requires the net to be idle this cycle and another copy to survive.",
                        cycle=cycle,
                        index=index,
                    )
                if previous_net == operation.net:
                    _error(
                        f"Destination {operation.location!r} already contains net {operation.net!r}.",
                        cycle=cycle,
                        index=index,
                    )

            if operation.type == Operation_type.WRITE:
                if operation.net not in primary_inputs:
                    _error("WRITE may initialize only a primary input.", cycle=cycle, index=index)
                if operation.net in written_inputs:
                    _error(
                        f"Primary input {operation.net!r} is written more than once; use COPY for replicas.",
                        cycle=cycle,
                        index=index,
                    )
                written_inputs.add(operation.net)
                write_locations[operation.net] = operation.location
                reserved_input_cells.add(operation.location)
                if operation.net in primary_outputs:
                    initial_output_locations[operation.net] = operation.location
                    reserved_output_cells.add(operation.location)
                continue

            if operation.type == Operation_type.COPY:
                copy_operations.append((index, operation))
                source = operation.src[0]
                if source == operation.location:
                    _error("COPY source and destination must differ.", cycle=cycle, index=index)
                source_net = snapshot.get(source)
                if source_net != operation.net:
                    _error(
                        f"COPY source {source!r} contains {source_net!r}, not {operation.net!r}.",
                        cycle=cycle,
                        index=index,
                    )
                continue

            gate_operations.append((index, operation))
            gate = gates_by_output.get(operation.net)
            if gate is None:
                _error(
                    f"No netlist gate produces {operation.net!r}.",
                    cycle=cycle,
                    index=index,
                )
            if operation.net in executed_outputs:
                _error(f"Gate output {operation.net!r} is executed more than once.", cycle=cycle, index=index)
            # Reserve the output immediately so a duplicate in this same cycle
            # is rejected as well as one in a later cycle.
            executed_outputs.add(operation.net)
            if operation.net in primary_outputs:
                initial_output_locations[operation.net] = operation.location
                reserved_output_cells.add(operation.location)

            expected_type = _EXPECTED_OPERATION_TYPE[gate.type]
            if operation.type != expected_type:
                _error(
                    f"Gate {gate.gid} requires {expected_type.name}, not {operation.type.name}.",
                    cycle=cycle,
                    index=index,
                )
            if tuple(operation.inputs or ()) != tuple(gate.inputs):
                _error(
                    f"Recorded inputs {operation.inputs!r} do not match gate {gate.gid} inputs {gate.inputs!r}.",
                    cycle=cycle,
                    index=index,
                )
            if len(operation.src or []) != len(gate.inputs):
                _error(
                    f"Gate {gate.gid} needs {len(gate.inputs)} input cells, but records "
                    f"{len(operation.src or [])}.",
                    cycle=cycle,
                    index=index,
                )

            actual_input_nets = Counter(snapshot.get(source) for source in operation.src or [])
            expected_input_nets = Counter(gate.inputs)
            if actual_input_nets != expected_input_nets:
                _error(
                    f"Gate {gate.gid} input cells contain {dict(actual_input_nets)!r}, "
                    f"expected {dict(expected_input_nets)!r}.",
                    cycle=cycle,
                    index=index,
                )
            if operation.net in locations and locations[operation.net]:
                _error(
                    f"Gate output {operation.net!r} exists before its producing operation.",
                    cycle=cycle,
                    index=index,
                )

            if operation.type in {Operation_type.NOR}:
                _validate_parallel_geometry(operation, cycle=cycle, index=index)
            elif operation.type in {Operation_type.AND}:
                _validate_series_geometry(operation, cycle=cycle, index=index)

        if len(copy_operations) > 1:
            _error("The current execution model permits only one COPY operation per cycle.", cycle=cycle)
        if copy_operations and gate_operations:
            _error("COPY and logic-gate execution cannot share a cycle.", cycle=cycle)

        all_sources = {
            source
            for _, operation in indexed_operations
            for source in (operation.src or [])
        }
        destination_source_overlap = set(destinations) & all_sources
        if destination_source_overlap:
            _error(
                f"Cells are read and overwritten in the same cycle: {sorted(destination_source_overlap)!r}.",
                cycle=cycle,
            )

        _validate_concurrent_gate_conflicts(gate_operations, cycle=cycle)

        # All reads above used the beginning-of-cycle snapshot. Commit writes now.
        for _, operation in indexed_operations:
            _place_cell(occupancy, locations, operation.location, operation.net)

        for _, operation in gate_operations:
            gate = gates_by_output[operation.net]
            for input_net in gate.inputs:
                remaining_uses[input_net] -= 1
                if remaining_uses[input_net] < 0:
                    _error(
                        f"Net {input_net!r} is consumed more times than its netlist fanout.",
                        cycle=cycle,
                    )

    missing_inputs = primary_inputs - written_inputs
    if missing_inputs:
        _error(f"Primary inputs were never written: {sorted(missing_inputs)!r}.")

    missing_gates = set(gates_by_output) - executed_outputs
    if missing_gates:
        _error(f"Gate outputs were never generated: {sorted(missing_gates)!r}.")

    for output_net in netlist.primary_outputs:
        if not locations.get(output_net):
            _error(f"Primary output {output_net!r} is absent after execution replay.")

    _validate_reported_state(
        netlist,
        state,
        occupancy,
        locations,
        write_locations,
        initial_output_locations,
        remaining_uses,
    )


def validate_mapping(netlist: Netlist, state: MappingState) -> None:
    """Validate the complete mapping, schedule, geometry, and final state."""

    if type(state.config.total_rows) is not int or state.config.total_rows <= 0:
        _error(f"total_rows must be a positive integer; got {state.config.total_rows!r}.")
    if type(state.config.total_columns) is not int or state.config.total_columns <= 0:
        _error(f"total_columns must be a positive integer; got {state.config.total_columns!r}.")
    if type(state.cycle_count) is not int or state.cycle_count < 0:
        _error(f"cycle_count must be a non-negative integer; got {state.cycle_count!r}.")

    validate_execution(netlist, state)


def validate_nor_inv_gates(netlist: Netlist, location: Dict[str, set[Coordinate]]) -> None:
    """Legacy final-state row check retained for callers outside this package.

    Full validation requires :func:`validate_mapping`, because a final cell map
    alone cannot prove execution order, liveness, or concurrent safety.
    """

    for gate in netlist.gates:
        if gate.type not in {GateType.NOR, GateType.NOT}:
            continue
        output_rows = {row for _, row in location.get(gate.output, set())}
        if not output_rows:
            _error(f"Parallel gate output {gate.output!r} has no mapped location.")
        for input_net in gate.inputs:
            input_rows = {row for _, row in location.get(input_net, set())}
            if not output_rows.intersection(input_rows):
                _error(
                    f"Parallel gate {gate.gid} input {input_net!r} and output "
                    f"{gate.output!r} have no common row."
                )
