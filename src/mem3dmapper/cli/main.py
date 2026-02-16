from enum import Enum
import json
from typing import List

from mem3dmapper.mapping import state
from mem3dmapper.netlist.parser import parse_netlist
from mem3dmapper.netlist.types import Netlist
from mem3dmapper.dag.build import build_DAG
from mem3dmapper.dag.graph_metrics import topo_sort
from mem3dmapper.dag.visualize import write_dot, render_graphviz
from mem3dmapper.mapping.heuristic import map_netlist
from mem3dmapper.mapping.types import Operation
from mem3dmapper.mapping.validator import validate_mapping

class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)

netlist = parse_netlist("data/netlists/2bit_adder.blif")
print(netlist.name, len(netlist.gates), "gates")

parents, children = build_DAG(netlist)
topo_order = topo_sort(children)
print("Topological Order of Gates:", topo_order)

### view DAG as DOT file
dot_file = write_dot("data/runs/2bit_adder.dot", netlist, parents, children)
render_graphviz(dot_file, "data/runs/2bit_adder.png", fmt="png")

state = map_netlist(netlist=netlist)

validate_mapping(netlist, state)

execution = state.ops

with open("data/json/2bit_adder_mapping.json", "w") as f:
    json.dump(execution, f, indent=4, cls=EnumEncoder)


