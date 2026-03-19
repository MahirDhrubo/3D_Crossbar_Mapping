from enum import Enum
import json

from mem3dmapper.mapping.state import MappingState
from mem3dmapper.netlist.parser import parse_netlist
from mem3dmapper.dag.build import build_DAG
from mem3dmapper.dag.visualize import write_dot, render_graphviz
from mem3dmapper.mapping.heuristic import map_netlist
from mem3dmapper.mapping.validator import validate_mapping

class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)

netlist = parse_netlist("data/netlists/dot4_4bit_nor.blif")
print(netlist.name, len(netlist.gates), "gates")

parents, children = build_DAG(netlist)
# topo_order = topo_sort(children)
# print("Topological Order of Gates:", topo_order)


### view DAG as DOT file
dot_file = write_dot("data/runs/dot4_4bit_nor.dot", netlist, parents, children)
render_graphviz(dot_file, "data/runs/dot4_4bit_nor.png", fmt="png")

state: MappingState = map_netlist(netlist=netlist)
print(f"Mapping completed with total cost: {state.total_cost} and total cycles: {state.cycle_count}")
validate_mapping(netlist, state)

execution = state.ops

with open("data/json/dot4_4bit_nor.json", "w") as f:
    json.dump(execution, f, indent=4, cls=EnumEncoder)


