from dataclasses import dataclass, field
from typing import Dict, Tuple, List

from mem3dmapper.mapping.types import Coordinate, Operation, Operation_type, MappingConfig
from mem3dmapper.netlist.types import GateType


@dataclass
class MappingState:
    config: MappingConfig
    cycle_count: int = 0

    net_location: Dict[str, set[Coordinate]] = field(default_factory=dict)
    location_to_net: Dict[Coordinate, str] = field(default_factory=dict)

    ops: list[Operation] = field(default_factory=list)
    number_of_writes: int = 0

    tail_x: List[int] = field(default_factory=list)
    row_cluster: List[int] = field(default_factory=list)

    def init_params(self) -> None:
        self.tail_x = [0] * self.config.total_rows
        self.row_cluster = [0] * self.config.total_rows

    def _increment_cycle(self) -> None:
        self.cycle_count += 1

    def _put_net(self, net: str, location: Coordinate) -> None:
        previous_net = self.location_to_net.get(location)

        if previous_net is not None and previous_net != net:
            self.net_location[previous_net].discard(location)
    
        self.location_to_net[location] = net
        self.net_location.setdefault(net, set()).add(location)

    def get_remaining_columns_on_row(self, row: int) -> int:
        return self.config.total_columns - self.tail_x[row]
    
    def get_any_location_of_net(self, net: str) -> Coordinate:
        #returns none if net not found
        return next(iter(self.net_location.get(net, set())), None)
    
    def get_location_of_net_on_row(self, net: str, row: int) -> Coordinate:
        for location in self.net_location.get(net, set()):
            if location[1] == row:
                return location
        return None

    def has_net_on_row(self, net: str, row: int) -> bool:
        return any(y == row for (x, y) in self.net_location.get(net, set()))
    
    def copy_cost_at(self, net: str, dest: Coordinate) -> int:
        if self.location_to_net.get(dest) == net:
            return 0
        
        # copy from same row -> 2 cycles (not, not)
        # copy from different row -> 3 cycles (not, and, not)
        
        return 2 if self.has_net_on_row(net, dest[1]) else 3


    def copy_net(self, net: str, src: Coordinate, dest: Coordinate) -> None:
        self._put_net(net, dest)
        self._increment_cycle()
        self.ops.append(Operation(
            type=Operation_type.COPY,
            net=net,
            location=dest,
            src=[src],
            cycle=self.cycle_count
        ))

    def write_net(self, net: str, location: Coordinate) -> None:
        self._put_net(net, location)
        self.number_of_writes += 1
        self._increment_cycle()
        self.ops.append(Operation(
            type=Operation_type.WRITE,
            net=net,
            location=location,
            cycle=self.cycle_count
        ))

    def execute_net(self,
                    net: str,
                    location: Coordinate,
                    gateType: GateType,
                    inputs: Tuple[str, ...],
                    input_locations: List[Coordinate]) -> None:
        self._put_net(net, location)

        operation_type = None
        if GateType.NOT == gateType:
            operation_type = Operation_type.NOT
        elif GateType.NOR == gateType:
            operation_type = Operation_type.NOR
        elif GateType.AND == gateType:
            operation_type = Operation_type.AND
        else:
            raise ValueError(f"Unsupported gate type: {gateType}")

        self._increment_cycle()
        self.ops.append(Operation(
            type=operation_type,
            net=net,
            location=location,
            inputs=inputs,
            src=input_locations,
            cycle=self.cycle_count
        ))