from typing import Dict, List, Tuple

from mem3dmapper.mapping.state import MappingState
from mem3dmapper.mapping.types import Coordinate
from mem3dmapper.netlist.types import Netlist

class MappingValidationError(RuntimeError):
    """Custom exception for mapping validation errors."""
    pass

def validate_mapping(netlist: Netlist, state: MappingState) -> None:
    """
    Validate the mapping of nets to locations.

    Args:
        netlist (Netlist): The netlist containing primary inputs and outputs.
        location (Dict[str, set[Tuple[int, int]]]): Mapping of nets to their locations.

    Raises:
        MappingValidationError: If any primary input or output net is not mapped.
    """
    location = state.net_location
    for net in netlist.primary_inputs:
        continue
        if net not in location or not location[net]:
            raise MappingValidationError(f"Primary input net '{net}' is not mapped to any location.")

    for net in netlist.primary_outputs:
        if net not in location or not location[net]:
            raise MappingValidationError(f"Primary output net '{net}' is not mapped to any location.")
        
    
    #validate_nor_inv_gates(netlist, location)


# validate if gate inputs exists
        

def validate_nor_inv_gates(netlist:Netlist, location: Dict[str, set[Coordinate]]) -> None:
    for gate in netlist.gates:
        if gate.is_parallel():
            out_rows = {y for (x, y) in location.get(gate.output, set())}
            if not out_rows or len(out_rows) == 0:
                raise MappingValidationError(f"NOR/INV gate output net '{gate.output}' is not mapped to any location.")
            for inp in gate.inputs:
                inp_rows = {y for (x, y) in location.get(inp, set())}
                if not inp_rows or len(inp_rows) == 0:
                    raise MappingValidationError(f"NOR/INV gate input net '{inp}' is not mapped to any location.")
                
                # naively checking for at least one common row, meaning both input and output nets are mapped to at least one common row
                # might need more sophisticated checks later
                # for example,if the output net is generated for row 2 is actually
                # by the inputs on the same row 2
                common = list(set(out_rows).intersection(set(inp_rows)))
                if len(common) == 0:
                    raise MappingValidationError(f"NOR/INV gate '{gate.gid}' input net '{inp}' and output net '{gate.output}' are not mapped to any common row.")