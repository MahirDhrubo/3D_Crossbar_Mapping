import random
from collections import defaultdict, deque
from typing import Dict, Iterator, List, Optional
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
    print("Performing topological sort")
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

def all_topo_orders(adj: Dict[int, set[int]], limit: Optional[int] = None) -> Iterator[List[int]]:
    """
    Generate all possible topological orders for the DAG `adj`.

    Warning: number of orders can be exponential. Use `limit` to stop early.

    Args:
        adj: adjacency list mapping node -> set(neighbors)
        limit: optional maximum number of orders to produce

    Yields:
        lists representing topological orders (each is a new list).
    """
    n = len(adj)
    # compute initial in-degrees
    in_degree = {u: 0 for u in adj}
    for u in adj:
        for v in adj[u]:
            in_degree[v] += 1

    zeros = [u for u in adj if in_degree[u] == 0]
    count = 0

    def backtrack(order: List[int], zeros_list: List[int]):
        nonlocal count
        if limit is not None and count >= limit:
            return
        if len(order) == n:
            count += 1
            yield list(order)
            return

        # iterate deterministically over available zero-indegree nodes
        for u in sorted(zeros_list):
            order.append(u)

            # prepare next zeros: remove u from current zeros
            next_zeros = [x for x in zeros_list if x != u]

            # decrement in-degrees for all neighbors of u and track them for restoration
            affected = []
            newly_zero = []
            for v in adj[u]:
                in_degree[v] -= 1
                affected.append(v)
                if in_degree[v] == 0:
                    next_zeros.append(v)
                    newly_zero.append(v)

            # recurse
            yield from backtrack(order, next_zeros)

            # backtrack: restore in-degrees for all affected neighbors
            for v in affected:
                in_degree[v] += 1

            order.pop()

            if limit is not None and count >= limit:
                return

    yield from backtrack([], zeros)

def _random_topo_order(adj: Dict[int, set[int]], rng: random.Random) -> List[int]:
    """One randomized Kahn run: pick a zero-indegree node uniformly at random."""
    in_degree = {u: 0 for u in adj}
    for u in adj:
        for v in adj[u]:
            in_degree[v] += 1

    zeros = [u for u in adj if in_degree[u] == 0]
    order: List[int] = []

    while zeros:
        u = rng.choice(zeros)
        zeros.remove(u)
        order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                zeros.append(v)

    if len(order) != len(adj):
        raise ValueError("Graph is not a DAG; topological sort not possible.")
    return order

def sample_topo_orders(adj: Dict[int, set[int]], n: int, seed: Optional[int] = None, max_attempts: int = 10000) -> Iterator[List[int]]:
    """
    Yield up to `n` distinct random topological orders (deterministic with `seed`).
    Stops early if max_attempts are exhausted.
    """
    rng = random.Random(seed)
    seen = set()
    attempts = 0
    while len(seen) < n and attempts < max_attempts:
        attempts += 1
        order = tuple(_random_topo_order(adj, rng))
        if order in seen:
            continue
        seen.add(order)
        yield list(order)

def compute_depths(parents: Dict[int, set[int]], topo_order: list[int]) -> Dict[int, int]:
    """
    Compute the depth of each gate in a DAG based on its parent gates.
    
    """

    gate_depths: Dict[int, int] = {}

    for gid in topo_order:
        pset = parents.get(gid, set())
        if not pset:
            gate_depths[gid] = 0
        else:
            gate_depths[gid] = 1 + max(gate_depths.get(p, 0) for p in pset)

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

# forward level all the nodes of a graph (adj[u] = successor)
def level_order_BFS(adj: Dict[int, set[int]]) -> Dict[int, int]:
    """
    Assign a level to each node in the graph based on its distance from the root.
    """
    levels = {u: -1 for u in adj}
    in_degree = {u: 0 for u in adj}
    queue = deque()
    
    for u in adj:
        for v in adj[u]:
            in_degree[v] += 1

    for u in adj:
        if in_degree[u] == 0:  # If no incoming edges, it's a root
            queue.append(u)
            levels[u] = 0

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            levels[v] = levels[u] + 1
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # If any node still has in_degree > 0, graph contains a cycle
    cyclic_nodes = [u for u, d in in_degree.items() if d > 0]
    if cyclic_nodes:
        raise ValueError(f"Graph contains cycle(s), cyclic nodes: {cyclic_nodes}")

    return levels
            
def group_by_levels(adj: Dict[int, set[int]]) -> Dict[int, List[int]]:
    """
    Group nodes by their levels.
    """
    print("Grouping nodes by levels (BFS)")
    levels = level_order_BFS(adj)
    
    grouped = defaultdict(list)
    for node, level in levels.items():
        grouped[level].append(node)
    return grouped

if __name__ == "__main__":
    # Example usage
    graph = {
        0: set(),
        1: {5, 6},
        2: {0, 9},
        3: {9, 4},
        4: {5, 6},
        5: {7},
        6: {7},
        7: {8},
        8: set(),
        9: {0}
    }
    grouped = group_by_levels(graph)
    print(grouped)