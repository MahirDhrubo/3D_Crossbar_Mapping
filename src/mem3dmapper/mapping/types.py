from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, List, Optional


Coordinate = Tuple[int, int]

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

@dataclass(frozen=True)
class MappingConfig:
    total_rows: int = 4
    total_columns: int = 4


    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0