from pathlib import Path

from .blif_parser import read_blif_file
from .types import Netlist

def parse_netlist(path: str) -> Netlist:
    ext = Path(path).suffix.lower()

    if ext in {'.blif', 'blf'}:
        return read_blif_file(path)
    
    else:
        raise ValueError(f"Unsupported netlist file extension: '{ext}'")