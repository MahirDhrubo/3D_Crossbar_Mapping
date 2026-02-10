from typing import List, Dict, Tuple, Optional
from itertools import permutations

from mem3dmapper.netlist.types import Netlist
from mem3dmapper.dag.graph_metrics import *
from mem3dmapper.dag.build import build_DAG
from mem3dmapper.netlist.types import Gate, GateType
from mem3dmapper.mapping.types import Coordinate, MappingConfig, Operation, AndPlacementPlant
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

def _heuristic_choose_nor_inv_row(
        state: MappingState,
        inputs: List[str],
        output: str,
        uses_left: Dict[str, int],
        cluster_id_map: Dict[str, int],
        row_fanout: int
) -> int:
    """
    Choose an appropriate row for placing a NOR gate based on heuristic criteria.
    """
    best_row = 0
    best_cost = float('inf')

    for row in range(state.config.total_rows):
        missing = 0
        for s in inputs:
            if not state.has_net_on_row(s, row):
                missing += 1

        # calculateing number of columns that can be overwritten
        dead_cells = 0
        for (x,y), net in state.location_to_net.items():
            if y == row and uses_left.get(net) == 0 and net not in inputs:
                dead_cells += 1

        new_columns = max(missing + 1 - dead_cells, 0)

        if new_columns > state.get_remaining_columns_on_row(row):
            continue

        future_misalignment_cost = calculate_future_misalignment_nor(
            net_c=output,
            row=row,
            intended_cluster=state.row_cluster[row],
            net_cluster=cluster_id_map.get(output, 0),
            row_fanout=row_fanout
        )

        total_cost = nor_row_cost(
            new_columns=new_columns,
            copies=missing,
            future_misalignment_cost=future_misalignment_cost
        )

        if total_cost < best_cost:
            best_cost = total_cost
            best_row = row

    return best_row
    

def _map_nor_inv(
        *,
        state: MappingState,
        gate: Gate,
        uses_left: Dict[str, int],
        cluster_id_map: Dict[str, int],
        row_fanout: int
) -> None:
    """
    Heuristic function to determine the best row for mapping a NOR gate.
    """
    row = _heuristic_choose_nor_inv_row(
        state,
        gate.inputs,
        gate.output,
        uses_left,
        cluster_id_map,
        row_fanout
    )

    # ensuring all inputs are available on the chosen row
    # if src not found, write the net to (last available column, row). Typically for the primary inputs
    # if found, copy to an available cell on that row
    inputs_locations: List[Coordinate] = []
    for inp in gate.inputs:
        if not state.has_net_on_row(inp, row):
            src = state.get_any_location_of_net(inp)
            dest: Coordinate = None
            for (x, y), net in state.location_to_net.items():
                if y == row and uses_left.get(net) == 0 and net not in gate.inputs:
                    dest = (x, y)
                    break

            if dest is None:
                dest = (state.tail_x[row], row)
                state.tail_x[row] += 1

            if src is None:
                state.write_net(inp, dest)
            else:
                state.copy_net(inp, src, dest)

        inputs_locations.extend((x,y) for (x,y) in state.net_location.get(inp, set()) if y == row)

    # output placement
    out_location: Coordinate = None
    for (x, y), net in state.location_to_net.items():
        if y == row and uses_left.get(net) == 0 and net not in gate.inputs:
            out_location = (x, y)
            break
    if out_location is None:
        out_location = (state.tail_x[row], row)
        state.tail_x[row] += 1
    
    state.execute_net(
        net=gate.output,
        location=out_location,
        gateType=gate.type,
        inputs=tuple(gate.inputs),
        input_locations=inputs_locations
    )

    for inp in gate.inputs:
        uses_left[inp] -= 1
    
    if state.row_cluster[row] is None:
        state.row_cluster[row] = cluster_id_map.get(gate.output, 0)

def _heuristic_AND_placement(
        *,
        state: MappingState,
        inputs: List[str],
        output: str,
        uses_left: Dict[str, int],
        cluster_id_map: Dict[str, int],
        row_fanout: int,
) -> AndPlacementPlant:
    
    best_placement = None
    best_cost = float("inf")

    input_length = len(inputs)

    #AND placement window limit
    y_limit = max(0, state.config.total_rows -  input_length)
    for col in range(state.config.total_columns):
        for row_start in range(0, y_limit + 1):
            output_cell_candidates = [row_start - 1, row_start + input_length]

            for output_cell_row in output_cell_candidates:
                if output_cell_row < 0 or output_cell_row > state.config.total_rows:
                    continue

                net = state.location_to_net.get((col, output_cell_row))
                if net in inputs or uses_left[net] != 0:
                    continue

                window_available = True
                for c in range(input_length):
                    cell_net = state.location_to_net.get((col, row_start + c))
                    if cell_net not in inputs and uses_left.get(cell_net, 0) != 0:
                        window_available = False
                        break
                
                if not window_available:
                    continue

                # row -> inputs, col -> input positions
                cost_mat = [[state.copy_cost_at(inputs[i], (col, row_start + c)) for c in range(input_length)] for i in range(input_length)]
                row_ind, col_ind, copy_cost = hungarian_algo(cost_mat)

                future_misalignment_cost = calculate_future_misalignment_nor(
                    net_c=output,
                    row=output_cell_row,
                    intended_cluster=state.row_cluster[output_cell_row],
                    net_cluster=cluster_id_map.get(output, 0),
                    row_fanout=row_fanout
                )

                total_cost = and_column_cost(
                    copies=copy_cost,
                    future_misalignment_cost=future_misalignment_cost
                )

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_placement = AndPlacementPlant(
                        column=col,
                        input_row_start=row_start,
                        output_cell_row=output_cell_row,
                        input_sequence=[inputs[i] for i in row_ind]
                    )

    if best_placement is None:
        raise RuntimeError("No valid placement found for AND gate")
    
    return best_placement

def _map_and_gate(
        *,
        state: MappingState,
        gate: Gate,
        uses_left: Dict[str, int],
        cluster_id_map: Dict[str, int],
        row_fanout: int
) -> None:
    placement = _heuristic_AND_placement(
        state=state,
        inputs=gate.inputs,
        output=gate.output,
        uses_left=uses_left,
        cluster_id_map=cluster_id_map,
        row_fanout=row_fanout
    )

    



def map_netlist(netlist: Netlist) -> List[Operation]:
    config = MappingConfig()
    state = MappingState(config=config)
    state.init_params()

    if netlist.producers is None or netlist.consumers is None:
        netlist.build_gate_map()

    parents, children = build_DAG(netlist)
    topo_order = topo_sort(children)
    gate_depths = compute_depths(parents, topo_order)

    total_fanout, row_fanout, col_fanout = _compute_fanout(netlist)
    cluster_id_map = _build_cluster_id_map(netlist, gate_depths, cluster_window=state.config.cluster_window)
    uses_left: Dict[str, int] = dict(total_fanout)

    gates_by_id: Dict[int, Gate] = {g.gid: g for g in netlist.gates}

    for gid in topo_order:
        gate = gates_by_id[gid]

        if gate.type == GateType.NOR or gate.type == GateType.NOT:
            _map_nor_inv(
                state=state,
                gate=gate,
                uses_left=uses_left,
                cluster_id_map=cluster_id_map,
                row_fanout=row_fanout.get(gate.output, 0)
            )
        elif gate.type == GateType.AND:
            pass
        else:
            raise ValueError(f"Unsupported gate type: {gate.type}") 
    
    return state