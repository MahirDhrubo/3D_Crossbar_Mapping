from typing import List, Dict, Optional

from .types import Netlist, Gate, GateType

class BlifParserError(Exception):
    pass

def _strip_comments(line: str) -> str:
    if '#' in line:
        line = line[:line.index('#')]
    return line.strip()

def _parse_pin_mapping(tokens: List[str]) -> Dict[str, str]:
    pin_map: Dict[str, str] = {}

    for token in tokens:
        if '=' not in token:
            raise BlifParserError(f"Invalid pin mapping. Expected 'pin=net', got '{token}'")
        pin, net = token.split('=', 1)
        pin_map[pin] = net.split('$')[-1].split('_')[-1]

    return pin_map

def parse_blif(blif_lines: List[str]) -> Netlist:
    name: Optional[str] = None
    gates: List[Gate] = []
    primary_inputs: List[str] = []
    primary_outputs: List[str] = []

    gid_counter = 0
    
    for line in blif_lines:
        line = _strip_comments(line)

        if not line:
            continue

        elif line.startswith('.model'):
            tokens = line.split()
            if len(tokens) != 2:
                raise BlifParserError(f"Invalid .model line: '{line}'")
            name = tokens[1]

        elif line.startswith('.inputs'):
            tokens = line.split()[1:]
            primary_inputs.extend(tokens)

        elif line.startswith('.outputs'):
            tokens = line.split()[1:]
            primary_outputs.extend(tokens)
        
        elif line.startswith('.names'):
            continue

        elif line.startswith('.subckt'):
            tokens = line.split()[1:]
            if len(tokens) < 2:
                raise BlifParserError(f"Invalid .subckt line: '{line}'")
            
            gate = tokens[0]
            pin_mapping = _parse_pin_mapping(tokens[1:])

            if gate == 'NOT':
                if 'A' not in pin_mapping or 'Y' not in pin_mapping:
                    raise BlifParserError(f"Missing A/Y pins: '{line}'")
                
                gates.append(Gate(gid=gid_counter, type=GateType.NOT, inputs=[pin_mapping['A']], output=pin_mapping['Y']))

            elif gate == 'AND2':
                if 'A' not in pin_mapping or 'B' not in pin_mapping or 'Y' not in pin_mapping:
                    raise BlifParserError(f"Missing A/B/Y pins: '{line}'")
                
                gates.append(Gate(gid=gid_counter, type=GateType.AND, inputs=[pin_mapping['A'], pin_mapping['B']], output=pin_mapping['Y']))

            elif gate == 'NOR2':
                if 'A' not in pin_mapping or 'B' not in pin_mapping or 'Y' not in pin_mapping:
                    raise BlifParserError(f"Missing A/B/Y pins: '{line}'")
                
                gates.append(Gate(gid=gid_counter, type=GateType.NOR, inputs=[pin_mapping['A'], pin_mapping['B']], output=pin_mapping['Y']))

            else:
                raise BlifParserError(f"Unsupported gate type: '{gate}'")
            
            gid_counter += 1

        elif line.startswith('.end'):
            break

        else:
            if line.startswith('.'):
                raise BlifParserError(f"Unsupported BLIF directive: '{line}'")
        
    if name is None:
        raise BlifParserError("BLIF file missing .model")
    
    if not primary_inputs:
        raise BlifParserError("BLIF file missing .inputs")
    
    if not primary_outputs:
        raise BlifParserError("BLIF file missing .outputs")
    
    netlist = Netlist(name=name, gates=gates, primary_inputs=primary_inputs, primary_outputs=primary_outputs)
    netlist.build_gate_map()

    return netlist

def read_blif_file(file_path: str) -> Netlist:
    with open(file_path, 'r') as f:
        blif_lines = f.readlines()
    
    return parse_blif(blif_lines)