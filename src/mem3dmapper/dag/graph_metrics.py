from collections import deque
from typing import Dict
from scipy.optimize import linear_sum_assignment

from mem3dmapper.netlist.types import Netlist

def topo_sort(adj: Dict[int, set[int]]):
    """Perform a topological sort on a directed acyclic graph (DAG).

    Args:
        adj (dict): A dictionary representing the adjacency list of the graph,
                    where keys are node identifiers and values are lists of
                    neighboring node identifiers.

    Returns:
        list: A list of nodes in topologically sorted order.
    """
    in_degree = {u: 0 for u in adj}  # Initialize in-degrees of all nodes to 0
    for u in adj:
        for v in adj[u]:
            in_degree[v] += 1  # Compute in-degrees

    # Initialize queue with nodes having in-degree of 0
    queue = deque([u for u in adj if in_degree[u] == 0])
    topo_order = []

    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1  # Decrease in-degree of neighboring nodes
            if in_degree[v] == 0:
                queue.append(v)  # Add to queue if in-degree becomes 0

    if len(topo_order) != len(adj):
        raise ValueError("Graph is not a DAG; topological sort not possible.")

    return topo_order

def compute_depths(parents: Dict[int, set[int]], topo_order: list[int]) -> Dict[int, int]:
    """
    Compute the depth of each gate in a DAG based on its parent gates.
    
    """

    gate_depths = {}

    for gid in topo_order:
        if not parents[gid]:
            gate_depths[gid] = 0
        else:
            gate_depths[gid] = 1 + max(gate_depths[p] for p in parents[gid])

    return gate_depths

def compute_gate_input_depths(netlist: Netlist, gate_depths: Dict[int, int]) -> Dict[str, int]:
    """
    Compute the input depths for each primary input in the netlist.

    """
    input_depths = {}

    for inp in netlist.primary_inputs:
        input_depths[inp] = 0
        consumers = netlist.consumers.get(inp, [])
        if not consumers:
            input_depths[inp] = 0
        else:
            input_depths[inp] = min(gate_depths[g.gid] for g in consumers)

    for gate in netlist.gates:
        input_depths[gate.output] = gate_depths[gate.gid]

    return input_depths

def hungarian_algo(cost):
    """
    Apply the Hungarian algorithm to solve the assignment problem.
    """
    row_index, col_index = linear_sum_assignment(cost)

    # min_cost = cost[row_index, col_index].sum() if cost is numpy
    min_cost = sum(cost[i][j] for i, j in zip(row_index, col_index))   

    return row_index, col_index, min_cost