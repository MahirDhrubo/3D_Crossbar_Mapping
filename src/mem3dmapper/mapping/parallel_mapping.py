
from collections import defaultdict
from collections.abc import Set
from concurrent.futures import Executor, ProcessPoolExecutor
from itertools import repeat
import math
import os
from typing import Dict, List
from mem3dmapper.dag.build import build_DAG
from mem3dmapper.dag.graph_metrics import compute_depths, group_by_levels, hungarian_algo, topo_sort
from mem3dmapper.mapping.cost import and_column_cost, calculate_future_misalignment_nor, nor_row_cost
from mem3dmapper.mapping.heuristic import _build_cluster_id_map, _compute_fanout
from mem3dmapper.mapping.state import MappingState
from mem3dmapper.mapping.types import AndPlacementPlan, Coordinate, MappingConfig, NorInvPlacementPlan
from mem3dmapper.netlist.types import Gate, Netlist, GateType

PROCESSED = 1
NOT_PROCESSED = 0

def _check_parallel_gate_placement(
        state: MappingState,
        gate: Gate,
        blocked_rows: List[int]
) -> NorInvPlacementPlan:
    best_placement = None
    best_cost = float('inf')

    inputs = gate.inputs

    usable_rows = [r for r in range(state.config.total_rows) if r not in blocked_rows]

    for row in usable_rows:
        for output_column in range(state.config.total_columns + 1):
            for in_column1 in range(state.config.total_columns + 1):
                for in_column2 in range(in_column1+1, state.config.total_columns + 1):
                    if in_column1 == output_column or in_column2 == output_column:
                        continue

                    # Check if the columns are free
                    net1 = state.location_to_net.get((in_column1, row))
                    net2 = state.location_to_net.get((in_column2, row))

                    if net1 not in inputs and not state.is_location_free((in_column1, row)):
                        continue
                    if net2 not in inputs and not state.is_location_free((in_column2, row)):
                        continue
                    if not state.is_location_free((output_column, row)):
                        continue
                    
                    # Calculate the cost for this placement
                    missing = 0
                    missing_inputs = []
                    input_columns = []
                    new_columns = 0

                    nets = [net1, net2]
                    for inp in inputs:
                        if (gate.type == GateType.NOR and inp not in nets) or (gate.type == GateType.NOT and state.net_location.get(inp) is None):
                            missing += 1
                            missing_inputs.append(inp)

                    if net1 not in inputs:
                        input_columns.append(in_column1)
                        if net1 is None:
                            new_columns += 1

                    if net2 not in inputs:
                        input_columns.append(in_column2)
                        if net2 is None:
                            new_columns += 1

                    total_cost = nor_row_cost(
                        new_columns=new_columns,
                        copies=missing,
                        cell_replaced=0,
                        future_misalignment_cost=0,
                        alpha=state.config.alpha,
                        beta=state.config.beta,
                        gamma=state.config.gamma
                    )

                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_placement=NorInvPlacementPlan(
                            row=row,
                            placable_nets=missing_inputs,
                            placement_columns=input_columns,
                            output_column=output_column
                        )

    return best_placement, best_cost


def _check_parallel_gate_placement_for_output_range(
        state: MappingState,
        gate: Gate,
        row: int,
        output_column_start: int,
        output_column_end: int
) -> tuple[NorInvPlacementPlan | None, float]:
    """Search one row for a contiguous range of output columns."""

    best_placement = None
    best_cost = float('inf')
    inputs = gate.inputs

    for output_column in range(output_column_start, output_column_end):
        for in_column1 in range(state.config.total_columns + 1):
            for in_column2 in range(in_column1 + 1, state.config.total_columns + 1):
                if in_column1 == output_column or in_column2 == output_column:
                    continue

                net1 = state.location_to_net.get((in_column1, row))
                net2 = state.location_to_net.get((in_column2, row))

                if net1 not in inputs and not state.is_location_free((in_column1, row)):
                    continue
                if net2 not in inputs and not state.is_location_free((in_column2, row)):
                    continue
                if not state.is_location_free((output_column, row)):
                    continue

                missing = 0
                missing_inputs = []
                input_columns = []
                new_columns = 0

                nets = [net1, net2]
                for inp in inputs:
                    if (
                        gate.type == GateType.NOR and inp not in nets
                    ) or (
                        gate.type == GateType.NOT
                        and state.net_location.get(inp) is None
                    ):
                        missing += 1
                        missing_inputs.append(inp)

                if net1 not in inputs:
                    input_columns.append(in_column1)
                    if net1 is None:
                        new_columns += 1

                if net2 not in inputs:
                    input_columns.append(in_column2)
                    if net2 is None:
                        new_columns += 1

                total_cost = nor_row_cost(
                    new_columns=new_columns,
                    copies=missing,
                    cell_replaced=0,
                    future_misalignment_cost=0,
                    alpha=state.config.alpha,
                    beta=state.config.beta,
                    gamma=state.config.gamma
                )

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_placement = NorInvPlacementPlan(
                        row=row,
                        placable_nets=missing_inputs,
                        placement_columns=input_columns,
                        output_column=output_column
                    )

    return best_placement, best_cost


def _check_parallel_gate_placement_for_row(
        state: MappingState,
        gate: Gate,
        row: int
) -> tuple[NorInvPlacementPlan | None, float]:
    """Search every input/output-column combination for one usable row."""

    return _check_parallel_gate_placement_for_output_range(
        state,
        gate,
        row,
        0,
        state.config.total_columns + 1
    )


def _check_parallel_gate_placement_for_ranges(
        state: MappingState,
        gate: Gate,
        search_range: tuple[int, int, int, int, int, int, int]
) -> tuple[NorInvPlacementPlan | None, float, tuple[int, int, int, int] | None]:
    """Search one rectangular chunk of the three nested column loops."""

    (
        row,
        output_start,
        output_end,
        input1_start,
        input1_end,
        input2_start,
        input2_end,
    ) = search_range
    best_placement = None
    best_cost = float('inf')
    best_order = None
    inputs = gate.inputs

    for output_column in range(output_start, output_end):
        for in_column1 in range(input1_start, input1_end):
            for in_column2 in range(
                max(in_column1 + 1, input2_start),
                input2_end
            ):
                if in_column1 == output_column or in_column2 == output_column:
                    continue

                net1 = state.location_to_net.get((in_column1, row))
                net2 = state.location_to_net.get((in_column2, row))

                if net1 not in inputs and not state.is_location_free((in_column1, row)):
                    continue
                if net2 not in inputs and not state.is_location_free((in_column2, row)):
                    continue
                if not state.is_location_free((output_column, row)):
                    continue

                missing = 0
                missing_inputs = []
                input_columns = []
                new_columns = 0
                nets = [net1, net2]

                for inp in inputs:
                    if (
                        gate.type == GateType.NOR and inp not in nets
                    ) or (
                        gate.type == GateType.NOT
                        and state.net_location.get(inp) is None
                    ):
                        missing += 1
                        missing_inputs.append(inp)

                if net1 not in inputs:
                    input_columns.append(in_column1)
                    if net1 is None:
                        new_columns += 1

                if net2 not in inputs:
                    input_columns.append(in_column2)
                    if net2 is None:
                        new_columns += 1

                total_cost = nor_row_cost(
                    new_columns=new_columns,
                    copies=missing,
                    cell_replaced=0,
                    future_misalignment_cost=0,
                    alpha=state.config.alpha,
                    beta=state.config.beta,
                    gamma=state.config.gamma
                )
                candidate_order = (
                    row,
                    output_column,
                    in_column1,
                    in_column2
                )

                if (
                    total_cost < best_cost
                    or (
                        total_cost == best_cost
                        and (best_order is None or candidate_order < best_order)
                    )
                ):
                    best_cost = total_cost
                    best_order = candidate_order
                    best_placement = NorInvPlacementPlan(
                        row=row,
                        placable_nets=missing_inputs,
                        placement_columns=input_columns,
                        output_column=output_column
                    )

    return best_placement, best_cost, best_order


def _split_search_dimension(length: int, parts: int) -> list[tuple[int, int]]:
    """Split ``range(length)`` into ordered, non-empty contiguous ranges."""

    chunk_size = (length + parts - 1) // parts
    return [
        (start, min(start + chunk_size, length))
        for start in range(0, length, chunk_size)
    ]


def _fast_parallel_gate_candidates_for_row(
        state: MappingState,
        gate: Gate,
        row: int
) -> tuple[list[int], list[int]]:
    """Return the only columns needed for an exact minimum-cost search.

    Input-pair cost depends only on whether a cell already contains a requested
    input, reuses a dead cell, or allocates a new cell. At most three earliest
    columns from each category are necessary: two may be used as inputs and one
    may be excluded because it is selected as the output.
    """

    inputs = set(gate.inputs)
    category_columns: Dict[tuple[str, str | None], list[int]] = defaultdict(list)
    output_columns: list[int] = []

    for column in range(state.config.total_columns):
        location = (column, row)
        net = state.location_to_net.get(location)
        is_free = state.is_location_free(location)

        if is_free and len(output_columns) < 3:
            output_columns.append(column)

        if net in inputs:
            category = ("input", net)
        elif is_free and net is None:
            category = ("new", None)
        elif is_free:
            category = ("reusable", None)
        else:
            continue

        if len(category_columns[category]) < 3:
            category_columns[category].append(column)

    input_columns = sorted(
        column
        for columns in category_columns.values()
        for column in columns
    )
    return output_columns, input_columns


def _check_parallel_gate_placement_parallel(
        state: MappingState,
        gate: Gate,
        blocked_rows: List[int],
        *,
        max_workers: int | None = None,
        executor: Executor | None = None
) -> tuple[NorInvPlacementPlan | None, float]:
    """Find the exact original optimum in linear time per usable row.

    ``max_workers`` and ``executor`` remain in the signature for compatibility
    with callers of the former multiprocessing implementation. The reduced
    candidate set is small enough that process startup and state serialization
    would cost more than the search itself.
    """

    del max_workers, executor
    best_placement = None
    best_cost = float('inf')
    best_order = None
    inputs = gate.inputs

    for row in range(state.config.total_rows):
        if row in blocked_rows:
            continue

        output_columns, candidate_input_columns = (
            _fast_parallel_gate_candidates_for_row(state, gate, row)
        )
        if not output_columns or len(candidate_input_columns) < 2:
            continue

        for output_column in output_columns:
            for first_index, in_column1 in enumerate(candidate_input_columns):
                for in_column2 in candidate_input_columns[first_index + 1:]:
                    if in_column1 == output_column or in_column2 == output_column:
                        continue

                    net1 = state.location_to_net.get((in_column1, row))
                    net2 = state.location_to_net.get((in_column2, row))
                    nets = [net1, net2]
                    missing_inputs = [
                        inp
                        for inp in inputs
                        if (
                            gate.type == GateType.NOR and inp not in nets
                        ) or (
                            gate.type == GateType.NOT
                            and state.net_location.get(inp) is None
                        )
                    ]
                    placement_columns = [
                        column
                        for column, net in (
                            (in_column1, net1),
                            (in_column2, net2)
                        )
                        if net not in inputs
                    ]
                    new_columns = sum(
                        state.location_to_net.get((column, row)) is None
                        for column in placement_columns
                    )
                    total_cost = nor_row_cost(
                        new_columns=new_columns,
                        copies=len(missing_inputs),
                        cell_replaced=0,
                        future_misalignment_cost=0,
                        alpha=state.config.alpha,
                        beta=state.config.beta,
                        gamma=state.config.gamma
                    )
                    candidate_order = (
                        row,
                        output_column,
                        in_column1,
                        in_column2
                    )

                    if (
                        total_cost < best_cost
                        or (
                            total_cost == best_cost
                            and (best_order is None or candidate_order < best_order)
                        )
                    ):
                        best_cost = total_cost
                        best_order = candidate_order
                        best_placement = NorInvPlacementPlan(
                            row=row,
                            placable_nets=missing_inputs,
                            placement_columns=placement_columns,
                            output_column=output_column
                        )

    return best_placement, best_cost



def _heuristic_choose_nor_inv_row(
        state: MappingState,
        gate: Gate,
        cluster_id_map: Dict[str, int],
        row_fanout: Dict[str, int],
        blocked_rows: List[int]
) -> NorInvPlacementPlan:
    """
    Choose an appropriate row for placing a NOR gate based on heuristic criteria.
    """
    best_row = -1
    best_placement = None
    best_cost = float('inf')
    best_replacement_cost = float('inf')

    inputs = gate.inputs
    output = gate.output

    usable_rows = [r for r in range(state.config.total_rows) if r not in blocked_rows]

    for row in usable_rows:
        missing = 0
        missing_inputs = []

        for s in inputs:
            if (gate.type == GateType.NOR and not state.has_net_on_row(s, row)) or (gate.type == GateType.NOT and state.net_location.get(s) is None):
                missing += 1
                missing_inputs.append(s)

            # calculating number of columns that can be overwritten
        free_cells: List[Coordinate] = []
        replaceable_cells: List[Coordinate] = []
        # for (x,y), net in state.location_to_net.items():
        for x in range(0, state.tail_x[row]):
            y = row
            net = state.location_to_net.get((x,y))
            if y == row and net not in inputs:
                if state.is_location_free((x,y)):
                    free_cells.append((x,y))

                # this need verification if the replaced net is being used by other ops in the same cycle or not
                elif state.is_net_replacable((x,y)):
                    replaceable_cells.append((x,y))

        new_columns = max(missing + 1 - len(free_cells), 0)

        if new_columns > state.get_remaining_columns_on_row(row) + len(replaceable_cells):
            continue

        required_replacements = max(new_columns - state.get_remaining_columns_on_row(row), 0)
        total_replacement_cost = 0
        selected_replacable_cells: List[Coordinate] = []
        if required_replacements > 0:
            def _cell_fanout(cell: Coordinate) -> int:
                net = state.location_to_net.get(cell)
                return row_fanout.get(net, 0)
            
            sorted_replaceable_cells = sorted(replaceable_cells, key=_cell_fanout)
            selected_replacable_cells = sorted_replaceable_cells[:required_replacements]
            total_replacement_cost = sum(_cell_fanout(cell) for cell in selected_replacable_cells)

        future_misalignment_cost = calculate_future_misalignment_nor(
            net_c=output,
            row=row,
            intended_cluster=state.row_cluster[row],
            net_cluster=cluster_id_map.get(output, 0),
            row_fanout=row_fanout.get(output, 0)
        )

        total_cost = nor_row_cost(
            new_columns=new_columns,
            copies= missing,
            cell_replaced=total_replacement_cost,
            future_misalignment_cost=future_misalignment_cost,
            alpha=state.config.alpha,
            beta=state.config.beta,
            gamma=state.config.gamma
        )

        # print(f"Total cost for placing parallel gate {output} on {gate.type.name} gate at row {row}: {total_cost}")

        if total_cost < best_cost or (total_cost == best_cost and total_replacement_cost < best_replacement_cost):
            best_cost = total_cost
            best_replacement_cost = total_replacement_cost
            
            input_columns = []
            column_count = 0
            total_columns_needed = len(missing_inputs) + 1 # +1 for the output
            for (x,y) in free_cells:
                if column_count < total_columns_needed: # +1 for the output
                    input_columns.append(x)
                    column_count += 1
                else:
                    break
                        
            tail_column = state.tail_x[row]
            for _ in range(column_count, total_columns_needed):
                if tail_column < state.config.total_columns:
                    input_columns.append(tail_column)
                    tail_column += 1
                else:
                    break
                        
            for (x,y) in selected_replacable_cells:
                if column_count < total_columns_needed:
                    input_columns.append(x)
                    column_count += 1
                else:
                    break
            
            best_placement=NorInvPlacementPlan(
                row=row,
                placable_nets=missing_inputs,
                placement_columns=input_columns[:-1],
                output_column=input_columns[-1]
            )

    return best_placement, best_cost

def _map_nor_inv(
        *,
        state: MappingState,
        gate: Gate,
        cluster_id_map: Dict[str, int],
        row_fanout: Dict[str, int],
        blocked_rows: List[int]
):
    """
    Heuristic function to determine the best row for mapping a NOR gate.
    """
    placement, cost = _heuristic_choose_nor_inv_row(
        state,
        gate,
        cluster_id_map,
        row_fanout,
        blocked_rows
    )

    # placement, cost = _check_parallel_gate_placement(
    #     state=state,
    #     gate=gate,
    #     blocked_rows=blocked_rows
    # )

    # placement, cost = _check_parallel_gate_placement_parallel(
    #     state=state,
    #     gate=gate,
    #     blocked_rows=blocked_rows
    # )

    if placement is None:
        return None

    row = placement.row
    last_used_column = -1
    for col, net in zip(placement.placement_columns, placement.placable_nets):
        dest = (col, row)
        src = state.get_any_location_of_net(net)

        if src is None:
            state.write_net(net, dest)
            state.primary_input_locations[net] = dest
        else:
            state.copy_net(net, src, dest)
        last_used_column = max(last_used_column, col)
    
    inputs_locations: List[Coordinate] = []
    for inp in gate.inputs:
        if gate.type == GateType.NOR:
            input_location = state.get_location_of_net_on_row(inp, row)
        else:
            input_location = state.get_any_location_of_net(inp)

        if input_location is not None:
            inputs_locations.append(input_location)

    out_location: Coordinate = (placement.output_column, placement.row)
    last_used_column = max(last_used_column, placement.output_column)

    return out_location, inputs_locations, last_used_column 

def _get_AND_input_sequence(state: MappingState,
                            inputs: List[str],
                            col: int,
                            row_start: int):
    input_length = len(inputs)

    # row -> inputs, col -> input positions
    cost_mat = [[state.copy_cost_at(inputs[i], (col, row_start + c)) for c in range(input_length)] for i in range(input_length)]
    input_order, corresponding_rows, placement_cost = hungarian_algo(cost_mat)

    input_seq = [None] * input_length
    for i, pos in zip(input_order, corresponding_rows):
        input_seq[pos] = inputs[i]

    return input_seq, placement_cost

def _heuristic_AND_placement(
        *,
        state: MappingState,
        inputs: List[str],
        output: str,
        cluster_id_map: Dict[str, int],
        row_fanout: int,
        blocked_rows: List[int]
) -> AndPlacementPlan:
    
    best_placement = None
    best_cost = float("inf")
    copy_cost = float("inf")

    input_length = len(inputs)

    #AND placement window limit
    y_limit = max(0, state.config.total_rows -  input_length)
    for col in range(state.config.total_columns):
        for row_start in range(0, y_limit + 1):
            # check if the current window is blocked
            if any(row in blocked_rows for row in range(row_start, row_start + input_length)):
                continue

            output_cell_candidates = [row_start - 1, row_start + input_length]

            for output_cell_row in output_cell_candidates:
                if not state.is_location_free((col, output_cell_row)):
                    continue

                if output_cell_row in blocked_rows:
                    continue
                
                new_cell = 0

                net = state.location_to_net.get((col, output_cell_row))
                if net in inputs or state.uses_left.get(net, 0) != 0:
                    continue

                if net is None:
                    new_cell += 1

                window_available = True
                for c in range(input_length):
                    # if output == 'n5412' and col == 305:
                    #     print(row_start + c)
                    cell_net = state.location_to_net.get((col, row_start + c))
                    if cell_net not in inputs and not state.is_location_free((col, row_start + c)):
                        window_available = False
                        break
                    if cell_net is None:
                        new_cell += 1

                if not window_available and not state.is_window_replacable(col, row_start, row_start + input_length, inputs):
                    continue

                input_sequence, placement_cost = _get_AND_input_sequence(state, inputs, col, row_start)

                future_misalignment_cost = calculate_future_misalignment_nor(
                    net_c=output,
                    row=output_cell_row,
                    intended_cluster=state.row_cluster[output_cell_row],
                    net_cluster=cluster_id_map.get(output, 0),
                    row_fanout=row_fanout.get(output, 0)
                )

                total_cost = and_column_cost(
                    copies=placement_cost,
                    new_cells=new_cell,
                    future_misalignment_cost=future_misalignment_cost,
                    alpha=state.config.alpha,
                    beta=state.config.beta,
                    gamma=state.config.gamma
                )

                # print(f"Total cost for placing AND gate {output} from row {row_start} and output_cell {output_cell_row} and col: {col}: {total_cost}")

                if total_cost < best_cost:
                    copy_cost = placement_cost
                    best_cost = total_cost
                    best_placement = AndPlacementPlan(
                        column=col,
                        input_row_start=row_start,
                        output_cell_row=output_cell_row,
                        input_sequence=input_sequence
                    )

    return best_placement, copy_cost

def _map_and_gate(
        *,
        state: MappingState,
        placement: AndPlacementPlan
):
    if placement is None:
        return None

    column = placement.column
    input_row_start = placement.input_row_start
    output_cell_row = placement.output_cell_row
    input_sequence = placement.input_sequence 

    # check if the inputs exists,
    # otherwise, copy them to the designated input positions
    for i, inp in enumerate(input_sequence):
        net = state.location_to_net.get((column, input_row_start + i))
        if net != inp:
            src = state.get_any_location_of_net(inp)
            dest = (column, input_row_start + i)

            if src is None:
                state.write_net(inp, dest)
                state.primary_input_locations[inp] = dest
            else:
                state.copy_net(inp, src, dest)

            state.tail_x[input_row_start + i] = max(state.tail_x[input_row_start + i], column + 1)


    out_location = (column, output_cell_row)
    input_locations = [(column, input_row_start + i) for i in range(len(input_sequence))]

    return out_location, input_locations 

def _check_gate_placement(
        state: MappingState,
        gate: Gate,
        row_start: int,
        row_end: int,
        output_cell_row: int,
        columns: list[int],
) :
    if gate.is_parallel():
        raise RuntimeError("This checking and mapping is only for series input sequence gates. Found parallel")
    
    best_placement = None
    best_cost = float("inf")

    if len(gate.inputs) != row_end - row_start + 1:
        return best_placement, best_cost

    inputs = gate.inputs
    input_length = len(inputs)

    # in_use_column = plan.column
    # output_cell_row = plan.output_cell_row

    for col in range(state.config.total_columns):
        if col in columns:
            continue
        
        if not state.is_location_free((col, output_cell_row)):
            continue

        net = state.location_to_net.get((col, output_cell_row))
        if net in inputs or state.uses_left.get(net, 0) != 0:
            continue

        window_available = True
        for c in range(input_length):
            cell_net = state.location_to_net.get((col, row_start + c))
            if cell_net not in inputs and not state.is_location_free((col, row_start + c)):
                window_available = False
                break
        
        if not window_available:
            continue

        input_sequence, placement_cost = _get_AND_input_sequence(state, inputs, col, row_start)

        if placement_cost < best_cost:
            best_cost = placement_cost
            best_placement = AndPlacementPlan(
                column=col,
                input_row_start=row_start,
                output_cell_row=output_cell_row,
                input_sequence=input_sequence
            )

    return best_placement, best_cost


def _map_concurrent_nodes(nodes: List[Gate],
                          state: MappingState,
                          cluster_id_map: Dict[str, int],
                          row_fanout: Dict[str, int],
                          gates_by_id: Dict[int, Gate]):
    # Nodes/Gates which are in same level can be processed in parallel/concurrently
    parallel_gates: Dict[int, int] = {}
    series_gates: Dict[int, int] = {}
    for gate in nodes:
        if gate.is_parallel():
            parallel_gates[gate.gid] = NOT_PROCESSED

        elif gate.is_series():
            series_gates[gate.gid] = NOT_PROCESSED

    total_gates_to_process = len(parallel_gates) + len(series_gates)
    blocked_rows: Set[int] = set()
    try_count:Set[int] = set()
    usable_series_placements: dict[tuple[int, int, int], list[int]] = defaultdict(list) # (row_start, row_end, output_row) -> list of columns used for AND placement in that window, to quickly check for future placements in the same window
    while total_gates_to_process > 0:
        if total_gates_to_process in try_count:
            # if we have already tried to process this many gates without success, it means we are stuck. It means even one gate 
            # at a time is also not possible
            print(try_count)
            raise RuntimeError("Stuck in processing concurrent nodes. No progress can be made for {} gates.".format(total_gates_to_process))

        try_count.add(total_gates_to_process)

        ready_gates: List[Gate] = []
        ready_gates_locations: List[Coordinate] = []
        ready_gates_input_location: Dict[int, List[Coordinate]] = {}
        ready_gates_last_used_column: Dict[int, int] = {}
        for gid, status in parallel_gates.items():
            if status == PROCESSED:
                continue

            gate = gates_by_id.get(gid)
            # If a gate has multiple inputs (more than 1) located on the same row and that
            # row is one of the blocked rows,it should not be processed in the current cycle.
            # Because, Concurrent execution would require copying the inputs from blocked 
            # rows to available rows, increasing the number of COPY cycles.
            common_row, common_count = state.maximum_common_row(gate.inputs)
            if common_row in blocked_rows and common_count > 1:
                continue

            result = _map_nor_inv(
                state=state,
                gate=gate,
                cluster_id_map=cluster_id_map,
                row_fanout=row_fanout,
                blocked_rows=list(blocked_rows)
            )

            if result is not None:
                out_location, input_locations, last_used_column = result
                parallel_gates[gate.gid] = PROCESSED
                total_gates_to_process -= 1
                blocked_rows.update([out_location[1], out_location[1] - 1, out_location[1] + 1]) # block the row for the rest of the gates in this cycle
                ready_gates.append(gate)
                ready_gates_locations.append(out_location)
                ready_gates_input_location[gate.gid] = input_locations
                ready_gates_last_used_column[gate.gid] = last_used_column
            else:
                break
        
        first_gate = True
        for gid, status in series_gates.items():
            if status == PROCESSED:
                continue

            gate = gates_by_id.get(gid)

            # have to check for "AND" gate inputs and decide whether to perform
            # it on any existing AND gate window or a different window.
            # The comparison is on how many copies does it require between them.

            series_placement_plan = None
            placement_cost = float("inf")
            # for plan in uasble_series_placements:
            #     new_placement_plan, cost = _check_gate_placement(state, gate, plan)
            #     if new_placement_plan is not None and cost < placement_cost:
            #         series_placement_plan = new_placement_plan
            #         placement_cost = cost

            # for (row_start, row_end, output_row), columns in usable_series_placements.items():
            #     new_placement_plan, cost = _check_gate_placement(state, gate, row_start, row_end, output_row, columns)
            #     if new_placement_plan is not None and cost < placement_cost:
            #         series_placement_plan = new_placement_plan
            #         placement_cost = cost

            new_placement_plan, cost = _heuristic_AND_placement(
                state=state,
                inputs=gate.inputs,
                output=gate.output,
                cluster_id_map=cluster_id_map,
                row_fanout=row_fanout,
                blocked_rows=blocked_rows
            )

            if new_placement_plan is not None and cost < placement_cost:
                series_placement_plan = new_placement_plan
                placement_cost = cost

            result_for_concurrent = _map_and_gate(
                state=state,
                placement=series_placement_plan
            )

            if result_for_concurrent is not None:
                # usable_series_placements.append(series_placement_plan)
                row_start = series_placement_plan.input_row_start
                row_end = series_placement_plan.input_row_start + len(series_placement_plan.input_sequence) - 1
                output_row = series_placement_plan.output_cell_row
                output_column = series_placement_plan.column
                usable_series_placements[(row_start, row_end, output_row)].append(output_column)
                out_location, input_locations = result_for_concurrent


                series_gates[gate.gid] = PROCESSED
                total_gates_to_process -= 1
                blocked_rows.update([out_location[1], out_location[1] - 1, out_location[1] + 1]) # block the rows for the rest of the gates in this cycle
                for loc in input_locations:
                    blocked_rows.update([loc[1], loc[1] - 1, loc[1] + 1])
                ready_gates.append(gate)
                ready_gates_locations.append(out_location)
                ready_gates_input_location[gate.gid] = input_locations
                ready_gates_last_used_column[gate.gid] = out_location[0]
            

        if ready_gates:
            state.execute_concurrent_nets(ready_gates, ready_gates_locations, ready_gates_input_location)

        for g, loc in zip(ready_gates, ready_gates_locations):
            for inp in g.inputs:
                state.uses_left[inp] -= 1

            if state.row_cluster[loc[1]] is None:
                state.row_cluster[loc[1]] = cluster_id_map.get(g.output, 0)

            if g.output in state.primary_outputs:
                state.primary_output_locations[g.output] = loc
            
            state.tail_x[loc[1]] = max(state.tail_x[loc[1]], ready_gates_last_used_column[g.gid] + 1)


        blocked_rows.clear()
        usable_series_placements.clear()



def map_netlist(netlist: Netlist, config: MappingConfig) -> MappingState:
    if netlist.producers is None or netlist.consumers is None:
        netlist.build_gate_map()

    print("Initiating...")

    parents, children = build_DAG(netlist)
    best_cost = float('inf')
    best_state: MappingState = None
    best_order: List[int] = None
    min_cycle: int = float('inf')
    gates_by_id = {g.gid: g for g in netlist.gates}

    N = 1500
    SEED = 42

    state = MappingState(config=config)
    state.init_params(netlist.primary_outputs)
    level_groups = group_by_levels(children)
    topo_order = topo_sort(children)
    gate_depths = compute_depths(parents, topo_order)
    total_fanout, row_fanout, col_fanout = _compute_fanout(netlist)
    cluster_id_map = _build_cluster_id_map(netlist, gate_depths, cluster_window=state.config.cluster_window)

    state.uses_left = dict(total_fanout)

    # print(level_groups)
    levels = len(level_groups)

    print("Processing Concurrent Nodes..")

    for level in range(levels):
        gids = level_groups[level]
        gates = [gates_by_id[gid] for gid in gids]
        print(f"Processing level {level} with {len(gates)} gates")
        _map_concurrent_nodes(gates, state, cluster_id_map, row_fanout, gates_by_id)

    return state

    # try parallel execution using processes
    # try:
    #     max_workers = min(len(topo_orders), max(1, (os.cpu_count() or 1)))
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
    #         futures = {ex.submit(_process_topo_order, topo, netlist, config): idx for idx, topo in enumerate(topo_orders, start=1)}
    #         for fut in concurrent.futures.as_completed(futures):
    #             res = fut.result()
    #             if res is None:
    #                 continue
    #             state, topo_order = res
    #             # guard in case MappingState isn't fully picklable / transmitted as expected
    #             try:
    #                 if state.cycle_count < min_cycle or (state.cycle_count == min_cycle and state.total_cost < best_cost):
    #                     min_cycle = state.cycle_count
    #                     best_cost = state.total_cost
    #                     best_state = state
    #                     best_order = topo_order
    #             except Exception:
    #                 # If state cannot be inspected reliably, skip it
    #                 continue

    # except Exception:
    #     # fallback to sequential loop if parallel execution fails (pickling issues, etc.)
    #     # for i in range(1):
    #         # topo_order = [0, 3, 2, 28, 4, 5, 6, 1, 30, 7, 9, 8, 29, 10, 33, 34, 31, 12, 32, 35, 11, 13, 16, 15, 36, 17, 21, 14, 18, 22, 20, 19, 24, 23, 25, 26, 27]
    #     for i, topo_order in enumerate(topo_orders, start=1):
    #         state = MappingState(config=config)
    #         state.init_params(netlist.primary_outputs)
    #         gate_depths = compute_depths(parents, topo_order)
    #         total_fanout, row_fanout, col_fanout = _compute_fanout(netlist)
    #         cluster_id_map = _build_cluster_id_map(netlist, gate_depths, cluster_window=state.config.cluster_window)
    #         state.uses_left = dict(total_fanout)

    #         is_complete = True
    #         try:
    #             for gid in topo_order:
    #                 gate = gates_by_id[gid]
    #                 if gate.type == GateType.NOR or gate.type == GateType.NOT:
    #                     _map_nor_inv(
    #                         state=state,
    #                         gate=gate,
    #                         cluster_id_map=cluster_id_map,
    #                         row_fanout=row_fanout
    #                     )
    #                 elif gate.type == GateType.AND:
    #                     _map_and_gate(
    #                         state=state,
    #                         gate=gate,
    #                         cluster_id_map=cluster_id_map,
    #                         row_fanout=row_fanout.get(gate.output, 0)
    #                     )
    #         except RuntimeError:
    #             is_complete = False

    #         if not is_complete:
    #             if i % 25 == 0:
    #                 print(f"Failed {i} topological orders.")
    #             continue

    #         if i % 25 == 0:
    #             print(f"Completed {i} topological orders. Current order cost: {state.total_cost}, cycles: {state.cycle_count}")

    #         if state.cycle_count < min_cycle or (state.cycle_count == min_cycle and state.total_cost < best_cost):
    #             min_cycle = state.cycle_count
    #             best_cost = state.total_cost
    #             best_state = state
    #             best_order = topo_order

    # return best_state
