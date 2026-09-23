from enum import Enum
from collections import Counter
import json

from mem3dmapper.mapping.state import MappingState
from mem3dmapper.mapping.types import Coordinate, MappingConfig, Operation_type
from mem3dmapper.netlist.parser import parse_netlist
from mem3dmapper.dag.build import build_DAG
from mem3dmapper.dag.visualize import write_dot, render_graphviz
from mem3dmapper.mapping.heuristic import map_2D, map_netlist
from mem3dmapper.mapping.validator import validate_mapping
from mem3dmapper.mapping.parallel_mapping import map_netlist as parallel_map_netlist

file_name = "4bit_adder_nor"

class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)

def mapping_with_arbitrary_rows(start_row, end_row, total_columns, netlist):
    for row in range(start_row, end_row + 1):
        try:
            mapper(total_rows=row, total_columns=total_columns, netlist=netlist)
        except Exception as e:
            print(f"Error occurred while mapping {row}x{total_columns}: {e}")

def mapper(total_rows, total_columns, netlist, config = None):
    if config is None:
        config = MappingConfig(total_rows=total_rows, total_columns=total_columns)

    print(f"Grid size {config.total_rows}x{config.total_columns}")

    # state: MappingState = map_netlist(netlist=netlist, config=config)
    state: MappingState = map_2D(netlist=netlist, config=config)
    if state is None:
        print(f"Mapping failed for grid size {config.total_rows}x{config.total_columns}")
        raise Exception(f"Mapping failed")

    print(f"Grid size {state.config.total_rows}x{state.config.total_columns}  Mapping completed with total cost: {state.total_cost} and total cycles: {state.cycle_count}")
    # validate_mapping(netlist, state)

    execution = state.ops
    operations_count = Counter(op.type.name for op in execution)
    operations_count["total"] = sum(operations_count.values()) - operations_count["WRITE"]
    peak_live_cells = count_peak_live_cells(state)
    json_string = {
        "execution": execution,
        "operations_count": operations_count,
        "peak_live_cells": peak_live_cells
    }
    with open(f"data/json/{file_name}_{state.config.total_rows}x{state.config.total_columns}.json", "w") as f:
        json.dump(json_string, f, indent=4, cls=EnumEncoder)

def run_circuits():
    for bit in range(32, 33):
        file_name = f"multiplier{bit}_nor"
        # file_name = f"adder{bit}"
        netlist = parse_netlist(f"data/netlists/multiplier_nor_netlist/{file_name}.blif")
        print(netlist.name, len(netlist.gates), "gates")

        parents, children = build_DAG(netlist)
        # topo_order = topo_sort(children)
        # print("Topological Order of Gates:", topo_order)


        ### view DAG as DOT file
        if len(netlist.gates) <= 500:
            dot_file = write_dot(f"data/runs/{file_name}.dot", netlist, parents, children)
            render_graphviz(dot_file, f"data/runs/{file_name}.png", fmt="png")

        # mapping_with_arbitrary_rows(start_row=3, end_row=20, total_columns=128)
        mapper(total_rows=4, total_columns=256, netlist=netlist)

def sensitivity_analysis():
    file_name = "adder64"
    netlist = parse_netlist(f"data/netlists/adder_netlist/{file_name}.blif")
    print(netlist.name, len(netlist.gates), "gates")
    
    import numpy as np
    for alpha in np.arange(1, 1.05, 0.1):
        for beta in np.arange(1, 1.05, 0.1):
            for gamma in np.arange(0.5, 3.3, 0.1):
                print(f"Running sensitivity analysis with alpha={alpha}, beta={beta}, gamma={gamma}")
                config = MappingConfig(total_rows=3, total_columns=128, cluster_window=21, alpha=alpha, beta=beta, gamma=gamma)
                state: MappingState = map_netlist(netlist=netlist, config=config)
                if state is None:
                    print(f"Mapping failed for grid size {config.total_rows}x{config.total_columns}")
                    raise Exception(f"Mapping failed")

                print(f"Grid size {state.config.total_rows}x{state.config.total_columns}  Mapping completed with total cost: {state.total_cost} and total cycles: {state.cycle_count}")
                validate_mapping(netlist, state)

def count_cell_usage(state: MappingState):
    # count the number of unique cell used
    
    grid: set[Coordinate] = set()

    for op in state.ops:
        grid.add(op.location)
    
    # print(f"Total number of cell used: {len(grid)}")
    return len(grid)


def count_peak_live_cells(state: MappingState) -> int:
    operations_by_cycle = {}
    for op in state.ops:
        operations_by_cycle.setdefault(op.cycle, []).append(op)

    future_uses = Counter()
    for op in state.ops:
        if op.type == Operation_type.COPY:
            future_uses[op.net] += 1
        elif op.inputs:
            future_uses.update(op.inputs)

    occupancy: dict[Coordinate, str] = {}
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


def concurrent_mapping():
    # bit = 32
    # file_name = f"adder{bit}"
    # file_name = f"multiplier{bit}"
    file_name = "dot_N17_B8_struct"
    # netlist = parse_netlist(f"data/netlists/multiplier_netlist/{file_name}.blif")
    netlist = parse_netlist(f"data/netlists/{file_name}.blif")
    print(netlist.name, len(netlist.gates), "gates")

    config = MappingConfig(total_rows=6, total_columns=512, cluster_window=21, alpha=0.6, beta=0.4, gamma=0)

    if len(netlist.gates) <= 500:
        parents, children = build_DAG(netlist)
        dot_file = write_dot(f"data/runs/{file_name}.dot", netlist, parents, children)
        render_graphviz(dot_file, f"data/runs/{file_name}.png", fmt="png")
    
    state: MappingState = parallel_map_netlist(netlist=netlist, config=config)
    # state: MappingState = map_netlist(netlist=netlist, config=config)
    if state is None:
        print(f"Mapping failed for grid size {config.total_rows}x{config.total_columns}")
        raise Exception(f"Mapping failed")

    cell_usage = count_cell_usage(state)
    peak_live_cells = count_peak_live_cells(state)

    print(f"Grid size {state.config.total_rows}x{state.config.total_columns}  total cycles: {state.cycle_count}  cell usage: {cell_usage}  peak live cells: {peak_live_cells}")
    validate_mapping(netlist, state)

    execution = state.ops
    operations_count = Counter(op.type.name for op in execution)
    operations_count["total"] = state.cycle_count
    json_string = {
        "execution": execution,
        "operations_count": operations_count,
        "peak_live_cells": peak_live_cells
    }

    with open(f"data/temp2/{file_name}_{state.config.total_rows}x{state.config.total_columns}.json", "w") as f:
        json.dump(json_string, f, indent=4, cls=EnumEncoder)

def sensitivity_analysis():
    bit = 32
    # file_name = f"adder{bit}"
    file_name = f"multiplier{bit}_nor"
    netlist = parse_netlist(f"data/netlists/multiplier_nor_netlist/{file_name}.blif")
    print(netlist.name, len(netlist.gates), "gates")
    for x in range(11):
        alpha = x * 0.1
        beta = 1 - alpha
        gamma = 0

        config = MappingConfig(total_rows=6, total_columns=512, cluster_window=21, alpha=alpha, beta=beta, gamma=gamma)
        try:
            state: MappingState = parallel_map_netlist(netlist=netlist, config=config)
        except Exception as e:
            print(f"Error occurred while mapping with alpha={alpha}, beta={beta}, gamma={gamma}: {e}")
        if state is None:
            print(f"Mapping failed for grid size {config.total_rows}x{config.total_columns}")
            continue

        cell_usage = count_cell_usage(state)

        # append the results to a file
        # with open(f"data/temp/sensitivity.json", "a") as f:
        #     json.dump({"file_name": file_name, "rows": state.config.total_rows, "columns": state.config.total_columns, "alpha": alpha, "beta": beta, "gamma": gamma, "total_cycles": state.cycle_count, "cell_usage": cell_usage}, f)
        #     f.write("\n")

        print({"file_name": file_name, "Crossbar": f"{state.config.total_rows}x{state.config.total_columns}", "alpha": alpha, "beta": beta, "gamma": gamma, "total_cycles": state.cycle_count, "cell_usage": cell_usage})
        try:
            validate_mapping(netlist, state)
        except Exception as e:
            print(f"Validation failed for alpha={alpha}, beta={beta}, gamma={gamma}: {e}")

if __name__ == "__main__":
    # run_circuits()
    # sensitivity_analysis()
    concurrent_mapping()
    # sensitivity_analysis()
