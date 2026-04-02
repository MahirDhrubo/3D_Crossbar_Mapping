from typing import List, Dict, Tuple, Optional
from itertools import permutations

from mem3dmapper.netlist.types import Netlist
from mem3dmapper.dag.graph_metrics import *
from mem3dmapper.dag.build import build_DAG
from mem3dmapper.netlist.types import Gate, GateType
from mem3dmapper.mapping.types import Coordinate, MappingConfig, NorInvPlacementPlan, AndPlacementPlan
from mem3dmapper.mapping.state import MappingState
from mem3dmapper.mapping.cost import calculate_future_misalignment_nor, nor_row_cost, and_column_cost

def _compute_fanout(netList: Netlist) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Compute the fanout statistics for each net in the netlist.
    1. total_fanout: total number of gates consumed by the net
    2. row_fanout: number of series gates consumed by the net
    3. col_fanout: number of parallel gates consumed by the net
    
    Args:
        netList (Netlist): The netlist to analyze.
    
    Returns:
        Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
            A tuple containing three dictionaries:
            - total_fanout: mapping from net name to total fanout count
            - row_fanout: mapping from net name to series gate fanout count
            - col_fanout: mapping from net name to parallel gate fanout count
    """

    total_fanout: Dict[str, int] = {}
    row_fanout: Dict[str, int] = {}
    col_fanout: Dict[str, int] = {}

    for (net, gateList) in netList.consumers.items():
        total_fanout[net] = len(gateList)
        row_fanout[net] = sum(1 for g in gateList if g.is_parallel())
        col_fanout[net] = sum(1 for g in gateList if g.is_series())

    for input in netList.primary_inputs:
        total_fanout.setdefault(input, 0)
        row_fanout.setdefault(input, 0)
        col_fanout.setdefault(input, 0)

    return total_fanout, row_fanout, col_fanout

def _build_cluster_id_map(netlist: Netlist, gate_depth: Dict[int, int], cluster_window = 10) -> Dict[int, int]:

    gate_input_depth = compute_gate_input_depths(netlist, gate_depth)

    return {net: depth // cluster_window for net, depth in gate_input_depth.items()}

def _get_input_columns(allowed_cells: List[Coordinate], replaceable_cells: List[Coordinate]) -> List[int]:
    """
    Get the input columns for a NOR gate placement based on allowed and replaceable cells.
    """
    input_columns = []
    for cell in allowed_cells:
        if cell in replaceable_cells:
            input_columns.append(cell[0])
    return input_columns

def _heuristic_choose_nor_inv_row(
        state: MappingState,
        gate: Gate,
        cluster_id_map: Dict[str, int],
        row_fanout: Dict[str, int]
) -> NorInvPlacementPlan:
    """
    Choose an appropriate row for placing a NOR gate based on heuristic criteria.
    """
    best_row = -1
    best_placement = None
    best_cost = float('inf')
    
    inputs = gate.inputs
    output = gate.output

    for row in range(state.config.total_rows):
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
            future_misalignment_cost=future_misalignment_cost
        )

        if total_cost < best_cost:
            best_cost = total_cost
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
        row_fanout: Dict[str, int]
) -> bool:
    """
    Heuristic function to determine the best row for mapping a NOR gate.
    """
    placement, cost = _heuristic_choose_nor_inv_row(
        state,
        gate,
        cluster_id_map,
        row_fanout
    )

    if placement is None:
        raise RuntimeError(f"No valid row found for NOR/INV gate with output {gate.output}")
    
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
        inputs_locations.extend((x,y) for (x,y) in state.net_location.get(inp, set()) if y == row)



    # ensuring all inputs are available on the chosen row
    # if src not found, write the net to (last available column, row). Typically for the primary inputs
    # if found, copy to an available cell on that row
    # for inp in gate.inputs:
    #     if not state.has_net_on_row(inp, row):
    #         src = state.get_any_location_of_net(inp)
    #         dest: Coordinate = None
    #         for (x, y), net in state.location_to_net.items():
    #             if state.is_loc_allowed((x,y)) and y == row and net not in gate.inputs:
    #                 dest = (x, y)
    #                 break

    #         if dest is None:
    #             dest = (state.tail_x[row], row)
    #             state.tail_x[row] += 1

    #         if src is None:
    #             state.write_net(inp, dest)
    #             state.primary_input_locations[inp] = dest

    #         else:
    #             state.copy_net(inp, src, dest)

    #     inputs_locations.extend((x,y) for (x,y) in state.net_location.get(inp, set()) if y == row)

    # output placement
    out_location: Coordinate = (placement.output_column, placement.row)
    last_used_column = max(last_used_column, placement.output_column)
    # for (x, y), net in state.location_to_net.items():
    #     if state.is_loc_allowed((x,y)) and y == row and net not in gate.inputs:
    #         out_location = (x, y)
    #         break
    # if out_location is None:
    #     out_location = (state.tail_x[row], row)
    #     state.tail_x[row] += 1
    
    state.execute_net(
        net=gate.output,
        location=out_location,
        gateType=gate.type,
        inputs=tuple(gate.inputs),
        input_locations=inputs_locations
    )

    for inp in gate.inputs:
        state.uses_left[inp] -= 1
    
    if state.row_cluster[row] is None:
        state.row_cluster[row] = cluster_id_map.get(gate.output, 0)

    if gate.output in state.primary_outputs:
        state.primary_output_locations[gate.output] = out_location
    
    state.tail_x[row] = max(state.tail_x[row], last_used_column + 1)

    state.total_cost += cost


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
) -> AndPlacementPlan:
    
    best_placement = None
    best_cost = float("inf")

    input_length = len(inputs)

    #AND placement window limit
    y_limit = max(0, state.config.total_rows -  input_length)
    for col in range(state.config.total_columns):
        for row_start in range(0, y_limit + 1):
            output_cell_candidates = [row_start - 1, row_start + input_length]

            for output_cell_row in output_cell_candidates:
                if not state.is_location_free((col, output_cell_row)):
                    continue

                net = state.location_to_net.get((col, output_cell_row))
                if net in inputs or state.uses_left.get(net, 0) != 0:
                    continue

                window_available = True
                for c in range(input_length):
                    cell_net = state.location_to_net.get((col, row_start + c))
                    if cell_net not in inputs and (state.uses_left.get(cell_net, 0) != 0 or not state.is_location_free((col, row_start + c)) or net in state.primary_outputs):
                        window_available = False
                        break
                
                if not window_available:
                    continue

                input_sequence, placement_cost = _get_AND_input_sequence(state, inputs, col, row_start)

                future_misalignment_cost = calculate_future_misalignment_nor(
                    net_c=output,
                    row=output_cell_row,
                    intended_cluster=state.row_cluster[output_cell_row],
                    net_cluster=cluster_id_map.get(output, 0),
                    row_fanout=row_fanout
                )

                total_cost = and_column_cost(
                    copies=placement_cost,
                    future_misalignment_cost=future_misalignment_cost
                )

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_placement = AndPlacementPlan(
                        column=col,
                        input_row_start=row_start,
                        output_cell_row=output_cell_row,
                        input_sequence=input_sequence
                    )

    if best_placement is None:
        raise RuntimeError("No valid placement found for AND gate")
    
    return best_placement, best_cost

def _map_and_gate(
        *,
        state: MappingState,
        gate: Gate,
        cluster_id_map: Dict[str, int],
        row_fanout: int
) -> None:
    placement, cost = _heuristic_AND_placement(
        state=state,
        inputs=gate.inputs,
        output=gate.output,
        cluster_id_map=cluster_id_map,
        row_fanout=row_fanout
    )

    column = placement.column
    input_row_start = placement.input_row_start
    output_cell_row = placement.output_cell_row
    input_sequence = placement.input_sequence 

    # check if the inputs exists,
    # otherwise, copy them to the designated input positions
    for i, inp in enumerate(input_sequence):
        net = state.location_to_net.get((column, input_row_start + i))
        if net != inp:
            src = state.get_location_of_net_on_row(inp, input_row_start + i)
            if src is None:
                src = state.get_any_location_of_net(inp)
            dest = (column, input_row_start + i)

            if src is None:
                state.write_net(inp, dest)
                state.primary_input_locations[inp] = dest
            else:
                state.copy_net(inp, src, dest)

            state.tail_x[input_row_start + i] = max(state.tail_x[input_row_start + i], column + 1)


    out_location = (column, output_cell_row)
    state.execute_net(
        net=gate.output,
        location=out_location,
        gateType=gate.type,
        inputs=tuple(gate.inputs),
        input_locations=[(column, input_row_start + i) for i in range(len(input_sequence))]
    )

    if gate.output in state.primary_outputs:
        state.primary_output_locations[gate.output] = out_location

    state.total_cost += cost
    state.tail_x[output_cell_row] = max(state.tail_x[output_cell_row], column + 1)

    if state.row_cluster[output_cell_row] is None:
        state.row_cluster[output_cell_row] = cluster_id_map.get(gate.output, 0)

    for inp in gate.inputs:
        state.uses_left[inp] -= 1    


# def map_netlist(netlist: Netlist) -> MappingState:
#     if netlist.producers is None or netlist.consumers is None:
#         netlist.build_gate_map()

#     parents, children = build_DAG(netlist)
#     #topo_order = topo_sort(children)
#     best_cost = float('inf')
#     best_state: MappingState = None
#     best_order: List[int] = None
#     min_cycle: int = float('inf')
#     gates_by_id: Dict[int, Gate] = {g.gid: g for g in netlist.gates}

#     N = 1500
#     SEED = 42
#     # for i in range(1):
#         # topo_order = [1, 2, 3, 4, 5, 6, 9, 0, 7, 8]
#         # topo_order = [3, 2, 4, 9, 0, 1, 5, 6, 7, 8]
#         # topo_order = [14, 0, 1, 13, 25, 26, 3, 2, 15, 16, 4, 27, 17, 18, 19, 5, 39, 20, 6, 40, 28, 12, 24, 22, 30, 7, 8, 29, 21, 35, 9, 31, 33, 23, 36, 32, 10, 37, 11, 38, 34]
#     for i, topo_order in enumerate(sample_topo_orders(children, n=N, seed=SEED), start=1):
#     # for i, topo_order in enumerate(all_topo_orders(children, limit=N), start=1):
#         config = MappingConfig()
#         state = MappingState(config=config)
#         state.init_params(netlist.primary_outputs)
#         # print(topo_order)
#         is_complete = True
#         gate_depths = compute_depths(parents, topo_order)

#         total_fanout, row_fanout, col_fanout = _compute_fanout(netlist)
#         cluster_id_map = _build_cluster_id_map(netlist, gate_depths, cluster_window=state.config.cluster_window)
#         state.uses_left = dict(total_fanout)

#         is_ok = True

#         for gid in topo_order:
#             gate = gates_by_id[gid]
#             try:
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
                    
#                 else:
#                     raise ValueError(f"Unsupported gate type: {gate.type}")
#             except RuntimeError:
#                 is_complete = False
#                 break
        
#         if not is_complete:
#             if i % 25 == 0:
#                 print(f"Failed {i} topological orders.")
#             continue
        
#         if i % 25 == 0:
#             print(f"Completed {i} topological orders. Current order cost: {state.total_cost}, cycles: {state.cycle_count}")
#         # print(f"(cycle = {state.cycle_count} cost = {state.total_cost}) Topological Order {i}:", topo_order)

#         if state.cycle_count < min_cycle:
#             min_cycle = state.cycle_count
#             best_cost = state.total_cost
#             best_state = state
#             best_order = topo_order
#     # print([gates_by_id.get(gid).output for gid in best_order])
#     print(best_order)
#     return best_state

# ...existing code...
import os
import concurrent.futures
from typing import Any
# ...existing code...

def _process_topo_order(topo_order: List[int], netlist: Netlist, config: MappingConfig) -> Optional[Tuple[MappingState, List[int]]]:
    """
    Worker function to evaluate a single topo_order. Returns (state, topo_order) on success,
    or None if mapping failed for that order.
    """
    state = MappingState(config=config)
    state.init_params(netlist.primary_outputs)

    gate_depths = compute_depths(*build_DAG(netlist)) if False else None  # unused here
    gates_by_id: Dict[int, Gate] = {g.gid: g for g in netlist.gates}

    total_fanout, row_fanout, col_fanout = _compute_fanout(netlist)
    gate_depths = compute_depths(*build_DAG(netlist))  # compute depths once per worker
    cluster_id_map = _build_cluster_id_map(netlist, gate_depths, cluster_window=state.config.cluster_window)
    state.uses_left = dict(total_fanout)

    try:
        for gid in topo_order:
            gate = gates_by_id[gid]
            if gate.type == GateType.NOR or gate.type == GateType.NOT:
                _map_nor_inv(
                    state=state,
                    gate=gate,
                    cluster_id_map=cluster_id_map,
                    row_fanout=row_fanout
                )
            elif gate.type == GateType.AND:
                _map_and_gate(
                    state=state,
                    gate=gate,
                    cluster_id_map=cluster_id_map,
                    row_fanout=row_fanout.get(gate.output, 0)
                )
            else:
                raise ValueError(f"Unsupported gate type: {gate.type}")
    except RuntimeError:
        return None

    return state, topo_order

def map_netlist(netlist: Netlist, config: MappingConfig) -> MappingState:
    if netlist.producers is None or netlist.consumers is None:
        netlist.build_gate_map()

    parents, children = build_DAG(netlist)
    best_cost = float('inf')
    best_state: MappingState = None
    best_order: List[int] = None
    min_cycle: int = float('inf')
    gates_by_id: Dict[int, Gate] = {g.gid: g for g in netlist.gates}

    N = 1500
    SEED = 42

    # collect topo orders up front (sample_topo_orders yields N orders)
    topo_orders = list(sample_topo_orders(children, n=N, seed=SEED))

    # try parallel execution using processes
    try:
        max_workers = min(len(topo_orders), max(1, (os.cpu_count() or 1)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_process_topo_order, topo, netlist, config): idx for idx, topo in enumerate(topo_orders, start=1)}
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                if res is None:
                    continue
                state, topo_order = res
                # guard in case MappingState isn't fully picklable / transmitted as expected
                try:
                    if state.cycle_count < min_cycle or (state.cycle_count == min_cycle and state.total_cost < best_cost):
                        min_cycle = state.cycle_count
                        best_cost = state.total_cost
                        best_state = state
                        best_order = topo_order
                except Exception:
                    # If state cannot be inspected reliably, skip it
                    continue

    except Exception:
        # fallback to sequential loop if parallel execution fails (pickling issues, etc.)
        # for i in range(1):
            # topo_order = [0, 3, 2, 28, 4, 5, 6, 1, 30, 7, 9, 8, 29, 10, 33, 34, 31, 12, 32, 35, 11, 13, 16, 15, 36, 17, 21, 14, 18, 22, 20, 19, 24, 23, 25, 26, 27]
        for i, topo_order in enumerate(topo_orders, start=1):
            state = MappingState(config=config)
            state.init_params(netlist.primary_outputs)
            gate_depths = compute_depths(parents, topo_order)
            total_fanout, row_fanout, col_fanout = _compute_fanout(netlist)
            cluster_id_map = _build_cluster_id_map(netlist, gate_depths, cluster_window=state.config.cluster_window)
            state.uses_left = dict(total_fanout)

            is_complete = True
            try:
                for gid in topo_order:
                    gate = gates_by_id[gid]
                    if gate.type == GateType.NOR or gate.type == GateType.NOT:
                        _map_nor_inv(
                            state=state,
                            gate=gate,
                            cluster_id_map=cluster_id_map,
                            row_fanout=row_fanout
                        )
                    elif gate.type == GateType.AND:
                        _map_and_gate(
                            state=state,
                            gate=gate,
                            cluster_id_map=cluster_id_map,
                            row_fanout=row_fanout.get(gate.output, 0)
                        )
            except RuntimeError:
                is_complete = False

            if not is_complete:
                if i % 25 == 0:
                    print(f"Failed {i} topological orders.")
                continue

            if i % 25 == 0:
                print(f"Completed {i} topological orders. Current order cost: {state.total_cost}, cycles: {state.cycle_count}")

            if state.cycle_count < min_cycle or (state.cycle_count == min_cycle and state.total_cost < best_cost):
                min_cycle = state.cycle_count
                best_cost = state.total_cost
                best_state = state
                best_order = topo_order

    return best_state
# ...existing code...