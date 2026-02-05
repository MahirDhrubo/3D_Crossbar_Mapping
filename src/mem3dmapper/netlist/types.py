from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict

class GateType(Enum):
    AND = auto()
    NOT = auto()
    NOR = auto()

PARALLEL_GATES = {GateType.NOT, GateType.NOR}
SERIES_GATES = {GateType.AND}

@dataclass(frozen=True)
class Gate:
    gid: int
    type: GateType
    inputs: List[str]
    output: str

    def is_parallel(self) -> bool:
        return self.type in PARALLEL_GATES
    
    def is_series(self) -> bool:
        return self.type in SERIES_GATES
    

@dataclass
class Netlist:
    name: str
    gates: List[Gate]
    primary_inputs: List[str]
    primary_outputs: List[str]

    producers: Dict[str, Gate] = None
    consumers: Dict[str, List[Gate]] = None

    def build_gate_map(self):

        self.producers = {}
        self.consumers = {}

        for gate in self.gates:
            self.producers[gate.output] = gate
            for inp in gate.inputs:
                if inp not in self.consumers:
                    self.consumers[inp] = []
                self.consumers[inp].append(gate)
        
