from typing import Set, List, Dict

def nor_row_cost(
        new_columns,
        copies,
        future_misalignment_cost,
        alpha=0.0,
        beta=1.0,
        gamma=1.0
    ) -> float:
    
    return (alpha * new_columns) + (beta * copies) + (gamma * future_misalignment_cost)

def and_column_cost(
        copies,
        future_misalignment_cost,
        beta=1.0,
        gamma=1.0
    ) -> float:
    
    return (beta * copies) + (gamma * future_misalignment_cost)

def calculate_future_misalignment_nor(
        *, 
        net_c: str,
        row: int,
        intended_cluster: int,
        net_cluster: int,
        row_fanout: int,) -> float:
    
    cluster_penalty = 0.0
    if intended_cluster is not None and net_cluster != intended_cluster:
        cluster_penalty = float(row_fanout)
    return cluster_penalty