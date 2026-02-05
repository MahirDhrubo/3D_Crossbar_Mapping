from mem3dmapper.netlist.types import Netlist

def build_DAG(netlist: Netlist):
    parents = {g.gid: set() for g in netlist.gates}
    children = {g.gid: set() for g in netlist.gates}

    for gate in netlist.gates:
        for input in gate.inputs:
            parent = netlist.producers.get(input)

            if parent is not None:
                parents[gate.gid].add(parent.gid)
                children[parent.gid].add(gate.gid)
    
    return parents, children