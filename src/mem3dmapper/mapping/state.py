from dataclasses import dataclass, field
from typing import Dict, Tuple, List

from mem3dmapper.mapping.types import Coordinate, Operation, Operation_type, MappingConfig
from mem3dmapper.netlist.types import GateType, Gate


@dataclass
class MappingState:
    config: MappingConfig

    primary_outputs: List[str] = field(default_factory=list)
    primary_input_locations: Dict[str, Coordinate] = field(default_factory=dict)
    primary_output_locations: Dict[str, Coordinate] = field(default_factory=dict)
    net_location: Dict[str, set[Coordinate]] = field(default_factory=dict)
    location_to_net: Dict[Coordinate, str] = field(default_factory=dict)
    copy_count: Dict[str, int] = field(default_factory=dict) # copies of nets
    uses_left: Dict[str, int] = field(default_factory=dict) # uses left for each net

    ops: list[Operation] = field(default_factory=list)
    number_of_writes: int = 0
    cycle_count: int = 0
    total_cost: float = 0.0

    tail_x: List[int] = field(default_factory=list)
    row_cluster: List[int] = field(default_factory=list)

    def init_params(self, primary_outputs: List[str]) -> None:
        self.tail_x = [0] * self.config.total_rows
        self.row_cluster = [0] * self.config.total_rows
        self.primary_outputs = primary_outputs

    def _increment_cycle(self, val = 1) -> None:
        self.cycle_count += val

    def _decrement_cycle(self, val = 1) -> None:
        self.cycle_count -= val

    def _put_net(self, net: str, location: Coordinate) -> None:
        previous_net = self.location_to_net.get(location)

        if previous_net is not None and previous_net != net:
            self.net_location[previous_net].discard(location)
    
        self.location_to_net[location] = net
        self.net_location.setdefault(net, set()).add(location)
    
    def _count_copy_cycle(self, src: Coordinate, dest: Coordinate) -> int:
        return 2 if src[1] == dest[1] else 3
    
    def get_new_column_on_row(self, row: int) -> int:
        if (self.tail_x[row] >= self.config.total_columns):
            return -1
        
        self.tail_x[row] += 1
        return self.tail_x[row] - 1

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

    def maximum_common_row(self, nets: List[str]) -> int:
        # find the row that is most common and its count
        row_count = {}
        for net in nets:
            locations = self.net_location.get(net)
            if locations:
                row_set = set()
                for location in locations:
                    row = location[1]
                    if row in row_set:
                        continue
                    row_set.add(row)
                    row_count[row] = row_count.get(row, 0) + 1

        if not row_count:
            return -1, 0

        # Find the row with the maximum count
        max_row = max(row_count, key=row_count.get)
        return max_row, row_count[max_row]

    def has_net_on_row(self, net: str, row: int) -> bool:
        for col in range(self.tail_x[row]):
            if self.location_to_net.get((col, row)) == net:
                return True
        return False

    def copy_cost_at(self, net: str, dest: Coordinate) -> int:
        if self.location_to_net.get(dest) == net:
            return 0
        if dest in self.primary_input_locations.values() or dest in self.primary_output_locations.values():
            return 1000 # prohibit copying to primary input/output locations
        
        # if self.net_location.get(net) is None:
        #     return 0

        # copy from same row -> 2 cycles (not, not)
        # copy from different row -> 3 cycles (not, and, not)
        
        #return 2 if self.has_net_on_row(net, dest[1]) else 3
        return 1


    def copy_net(self, net: str, src: Coordinate, dest: Coordinate) -> None:
        self._put_net(net, dest)

        # self._increment_cycle(self._count_copy_cycle(src, dest))
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
        #self._increment_cycle()
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

    def execute_concurrent_nets(self,
                                gates: List[Gate],
                                locations: List[Coordinate],
                                input_locations_by_gate: Dict[int, List[Coordinate]]) -> None:
        for gate, location in zip(gates, locations):
            self.execute_net(gate.output, location, gate.type, tuple(gate.inputs), input_locations_by_gate.get(gate.gid))
            self._decrement_cycle()
        
        self._increment_cycle()
            

    def is_forbidden(self, location: Coordinate) -> bool:
        # Check if the location is forbidden for placement
        if not (0 <= location[0] < self.config.total_columns and 0 <= location[1] < self.config.total_rows):
            return True

        if location in self.primary_input_locations.values():
            return True

        if location in self.primary_output_locations.values():
            return True

        return False

    def is_location_free(self, location: Coordinate) -> bool:
        if self.is_forbidden(location):
            return False

        if self.uses_left.get(self.location_to_net.get(location), 0) != 0:
            return False

        return True

    def is_net_replacable(self, location: Coordinate) -> bool:
        if self.is_forbidden(location):
            return False

        # A net is replacable if it has multiple copies
        net = self.location_to_net.get(location)
        return len(self.net_location.get(net, set())) > 1
    
    def is_window_replacable(self, col, row_start, row_end, excluded_nets) -> bool:
        copy_count = {}
        for row in range(row_start, row_end + 1):
            net = self.location_to_net.get((col, row))
            if self.is_forbidden((col, row)):
                return False

            copy_count[net] = len(self.net_location.get(net, set()))
        
        is_replacable = True
        for row in range(row_start, row_end + 1):
            net = self.location_to_net.get((col, row))

            if net in excluded_nets:
                continue
            
            if copy_count[net] <= 1:
                is_replacable = False
                break
            else:
                copy_count[net] -= 1

        return is_replacable
