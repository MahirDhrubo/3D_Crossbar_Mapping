from enum import Enum
from collections import Counter
import json

from mem3dmapper.mapping.state import MappingState
from mem3dmapper.mapping.types import MappingConfig
from mem3dmapper.netlist.parser import parse_netlist
from mem3dmapper.dag.build import build_DAG
from mem3dmapper.dag.visualize import write_dot, render_graphviz
from mem3dmapper.mapping.heuristic import map_2D, map_netlist
from mem3dmapper.mapping.validator import validate_mapping

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

    state: MappingState = map_netlist(netlist=netlist, config=config)
    # state: MappingState = map_2D(netlist=netlist, config=config)
    if state is None:
        print(f"Mapping failed for grid size {config.total_rows}x{config.total_columns}")
        raise Exception(f"Mapping failed")

    print(f"Grid size {state.config.total_rows}x{state.config.total_columns}  Mapping completed with total cost: {state.total_cost} and total cycles: {state.cycle_count}")
    validate_mapping(netlist, state)

    execution = state.ops
    operations_count = Counter(op.type.name for op in execution)
    operations_count["total"] = sum(operations_count.values()) - operations_count["WRITE"]
    json_string = {
        "execution": execution,
        "operations_count": operations_count
    }
    # with open(f"data/json/{file_name}_{state.config.total_rows}x{state.config.total_columns}.json", "w") as f:
    #     json.dump(json_string, f, indent=4, cls=EnumEncoder)

def run_circuits():
    for bit in range(16, 17):
        file_name = f"multiplier{bit}"
        # file_name = f"adder{bit}"
        netlist = parse_netlist(f"data/netlists/multiplier_netlist/{file_name}.blif")
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
        

if __name__ == "__main__":
    run_circuits()
    # sensitivity_analysis()


