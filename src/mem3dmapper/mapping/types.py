from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, List, Optional


Coordinate = Tuple[int, int] # (x,y) => (col, row)

class Operation_type(Enum):
    WRITE = auto()
    COPY = auto()
    NOR = auto()
    NOT = auto()
    AND = auto()


@dataclass(frozen=True)
class Operation:
    type: Operation_type
    net: str
    cycle: int
    location: Coordinate
    inputs: Optional[Tuple[str, ...]] = None # Only for NOR, NOT, AND operations

    src: Optional[List[Coordinate]] = None # for input location (except write)

class MappingConfig:
    total_rows: int = 6
    total_columns: int = 256
    cluster_window: int = 256

    alpha: float
    beta: float
    gamma: float

    def __init__(self, total_rows, total_columns, cluster_window = 50, alpha = 1.0, beta = 1.0, gamma = 1.0):
        self.total_rows = total_rows
        self.total_columns = total_columns
        self.cluster_window = cluster_window
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

@dataclass
class AndPlacementPlan:
    column: int
    input_row_start: int
    output_cell_row: int
    input_sequence: List[str]

@dataclass
class NorInvPlacementPlan:
    row: int
    placable_nets: List[str]
    placement_columns: List[int]
    output_column: int