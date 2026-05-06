from typing import Set, List, Dict

def nor_row_cost(
        new_columns,
        copies,
        cell_replaced,
        future_misalignment_cost,
        alpha,
        beta,
        gamma,
        delta=0.0
    ) -> float:

    return (alpha * copies) + (beta * new_columns) + (gamma * future_misalignment_cost) + (delta * cell_replaced)

def and_column_cost(
        copies,
        new_cells,
        future_misalignment_cost,
        alpha,
        beta,
        gamma
    ) -> float:
    
    return (alpha * copies) + (beta * new_cells) + (gamma * future_misalignment_cost)

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