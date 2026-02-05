from pathlib import Path
from typing import Dict, List, Set

from mem3dmapper.netlist.types import GateType, Netlist


def _gate_label(netlist: Netlist, gid: int) -> str:
    g = netlist.gates_by_id[gid] if hasattr(netlist, "gates_by_id") else None
    if g is None:
        # fallback: search (ok for small circuits)
        for gg in netlist.gates:
            if gg.gid == gid:
                g = gg
                break
    if g is None:
        return f"g{gid}"

    if g.type == GateType.NOT:
        return f"g{gid}: NOT\\n{g.inputs[0]} → {g.output}"
    else:
        return f"g{gid}: {g.type.name}\\n{g.inputs[0]}, {g.inputs[1]} → {g.output}"

def dag_to_dot(
    netlist: Netlist,
    preds: Dict[int, Set[int]],
    succs: Dict[int, Set[int]],
    *,
    rankdir: str = "LR",
    include_isolated: bool = True,
) -> str:
    """
    Returns a Graphviz DOT string for the gate-level DAG.

    Nodes are gates (gid). Edges are dependencies.
    Node labels include type + nets for readability.

    rankdir: "LR" (left-to-right) or "TB" (top-to-bottom)
    """
    # Collect all nodes
    nodes = set(preds.keys()) | set(succs.keys())
    if not include_isolated:
        # keep only nodes that have at least one edge
        nodes = {n for n in nodes if preds.get(n) or succs.get(n)}

    lines: List[str] = []
    lines.append('digraph mem3d_dag {')
    lines.append(f'  rankdir={rankdir};')
    lines.append('  node [shape=box, fontsize=10];')
    lines.append('  edge [fontsize=9];')

    # Nodes
    for gid in sorted(nodes):
        label = _gate_label(netlist, gid).replace('"', '\\"')
        lines.append(f'  g{gid} [label="{label}"];')

    # Edges
    for u in sorted(nodes):
        for v in sorted(succs.get(u, set())):
            if v not in nodes:
                continue
            lines.append(f'  g{u} -> g{v};')

    lines.append('}')
    return "\n".join(lines)


def write_dot(
    path: str | Path,
    netlist: Netlist,
    preds: Dict[int, Set[int]],
    succs: Dict[int, Set[int]],
    *,
    rankdir: str = "LR",
) -> Path:
    """
    Write DOT file to `path` and return the Path.
    """
    p = Path(path)
    dot = dag_to_dot(netlist, preds, succs, rankdir=rankdir)
    p.write_text(dot, encoding="utf-8")
    return p


def render_graphviz(
    dot_path: str | Path,
    out_path: str | Path,
    *,
    fmt: str = "png",
) -> Path:
    """
    Render a DOT file using graphviz (dot) if available.
    Requires `graphviz` installed in WSL: sudo apt-get install graphviz

    fmt: "png" or "svg"
    """
    import subprocess

    dot_path = Path(dot_path)
    out_path = Path(out_path)

    cmd = ["dot", f"-T{fmt}", str(dot_path), "-o", str(out_path)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("Graphviz 'dot' not found. Install with: sudo apt-get install graphviz") from e
    return out_path
