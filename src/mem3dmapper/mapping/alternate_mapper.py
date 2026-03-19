from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Iterable
from collections import defaultdict
import math
import heapq

from mem3dmapper.netlist.types import Gate, GateType, Netlist
from mem3dmapper.dag.build import build_DAG


# ============================================================
# Runtime structures
# ============================================================

Coord = Tuple[int, int]  # (row, col), 0-based


@dataclass(frozen=True)
class Op:
    kind: str                 # WRITE, COPY, GATE
    gate_gid: Optional[int]   # for GATE only, but may also annotate prep ops
    net: Optional[str] = None
    src: Optional[Coord] = None
    dst: Optional[Coord] = None
    gate_type: Optional[GateType] = None
    gate_inputs: Optional[Tuple[Coord, ...]] = None
    gate_output: Optional[Coord] = None


@dataclass
class Cell:
    net: Optional[str] = None
    fixed_pi: bool = False
    fixed_po: bool = False


@dataclass
class State:
    rows: int
    cols: int
    grid: Dict[Coord, Cell]
    net_locs: Dict[str, Set[Coord]]
    executed: Set[int]
    ready: Set[int]
    remaining_uses: Dict[str, int]
    copies_so_far: int
    ops: List[Op] = field(default_factory=list)

    def clone(self) -> "State":
        return State(
            rows=self.rows,
            cols=self.cols,
            grid={k: Cell(v.net, v.fixed_pi, v.fixed_po) for k, v in self.grid.items()},
            net_locs={n: set(v) for n, v in self.net_locs.items()},
            executed=set(self.executed),
            ready=set(self.ready),
            remaining_uses=dict(self.remaining_uses),
            copies_so_far=self.copies_so_far,
            ops=list(self.ops),
        )


@dataclass
class TokenPlacement:
    token_idx: int
    net: str
    cell: Coord
    mode: str  # EXISTING / WRITE / COPY


@dataclass
class Realization:
    copy_cost: int
    prep_ops: List[Op]
    input_cells: List[Coord]
    output_cell: Coord
    score_tiebreak: int = 0  # lower is better


@dataclass
class SolveResult:
    success: bool
    best_state: Optional[State]
    best_copy_count: Optional[int]


# ============================================================
# Helpers
# ============================================================

def all_cells(rows: int, cols: int) -> Iterable[Coord]:
    for r in range(rows):
        for c in range(cols):
            yield (r, c)


def init_state(netlist: Netlist, rows: int, cols: int, parents: Dict[int, Set[int]]) -> State:
    grid = {(r, c): Cell() for r in range(rows) for c in range(cols)}
    net_locs: Dict[str, Set[Coord]] = defaultdict(set)

    remaining_uses = defaultdict(int)
    for g in netlist.gates:
        for inp in g.inputs:
            remaining_uses[inp] += 1

    ready = {g.gid for g in netlist.gates if len(parents[g.gid]) == 0}

    return State(
        rows=rows,
        cols=cols,
        grid=grid,
        net_locs=net_locs,
        executed=set(),
        ready=ready,
        remaining_uses=dict(remaining_uses),
        copies_so_far=0,
        ops=[],
    )


def gate_map(netlist: Netlist) -> Dict[int, Gate]:
    return {g.gid: g for g in netlist.gates}


def is_unwritten_pi(state: State, netlist: Netlist, net: str) -> bool:
    return net in netlist.primary_inputs and len(state.net_locs.get(net, set())) == 0


def is_available_net(state: State, net: str) -> bool:
    return len(state.net_locs.get(net, set())) > 0


def is_fixed_cell(state: State, cell: Coord) -> bool:
    c = state.grid[cell]
    return c.fixed_pi or c.fixed_po


def can_overwrite_cell(state: State, cell: Coord, protected: Set[Coord]) -> bool:
    if cell in protected:
        return False

    c = state.grid[cell]
    if c.fixed_pi or c.fixed_po:
        return False

    if c.net is None:
        return True

    old_net = c.net
    copies = len(state.net_locs.get(old_net, set()))
    rem = state.remaining_uses.get(old_net, 0)

    # Allowed if dead, or another surviving copy remains.
    if rem == 0:
        return True
    return copies >= 2


def remove_net_from_cell(state: State, cell: Coord):
    old_net = state.grid[cell].net
    if old_net is not None:
        if cell in state.net_locs.get(old_net, set()):
            state.net_locs[old_net].remove(cell)
            if not state.net_locs[old_net]:
                del state.net_locs[old_net]
    state.grid[cell].net = None


def place_net_in_cell(state: State, cell: Coord, net: str, fix_pi: bool = False, fix_po: bool = False):
    # caller must ensure writable / legal
    remove_net_from_cell(state, cell)
    state.grid[cell].net = net
    if fix_pi:
        state.grid[cell].fixed_pi = True
    if fix_po:
        state.grid[cell].fixed_po = True
    state.net_locs.setdefault(net, set()).add(cell)


def existing_source_for_net(state: State, net: str) -> Optional[Coord]:
    locs = sorted(state.net_locs.get(net, set()))
    return locs[0] if locs else None


def max_bipartite_matching(tokens: List[Tuple[int, str]], candidate_cells: List[Coord], state: State):
    """
    tokens: [(token_idx, net), ...]
    candidate_cells: cells that can satisfy token for free iff cell currently holds net
    Returns dict token_idx -> cell
    """
    adj: Dict[int, List[Coord]] = {}
    for tidx, net in tokens:
        matches = []
        for cell in candidate_cells:
            if state.grid[cell].net == net:
                matches.append(cell)
        adj[tidx] = matches

    match_to_token: Dict[Coord, int] = {}

    def dfs(tidx: int, seen: Set[Coord]) -> bool:
        for cell in adj[tidx]:
            if cell in seen:
                continue
            seen.add(cell)
            if cell not in match_to_token or dfs(match_to_token[cell], seen):
                match_to_token[cell] = tidx
                return True
        return False

    for tidx, _ in tokens:
        dfs(tidx, set())

    token_to_cell = {tidx: cell for cell, tidx in match_to_token.items()}
    return token_to_cell


def consumer_alignment_score(netlist: Netlist, net: str, cell: Coord) -> int:
    """
    Lower is better.
    Small future-aware tie-break:
    - NOR consumers like row alignment
    - AND consumers like column alignment
    """
    r, c = cell
    score = 0
    for g in netlist.consumers.get(net, []):
        if g.type == GateType.NOR:
            score += r
        elif g.type == GateType.AND:
            score += c
        else:
            score += 0
    return score


def state_signature(state: State) -> Tuple:
    """
    Dedup signature.
    """
    net_pos = []
    for net in sorted(state.net_locs.keys()):
        locs = tuple(sorted(state.net_locs[net]))
        net_pos.append((net, locs))
    fixed = tuple(sorted(
        (cell, state.grid[cell].fixed_pi, state.grid[cell].fixed_po)
        for cell in state.grid
        if state.grid[cell].fixed_pi or state.grid[cell].fixed_po
    ))
    return (
        tuple(sorted(state.executed)),
        tuple(net_pos),
        fixed,
    )


# ============================================================
# Local solver
# ============================================================

def local_realizations_for_gate(
    state: State,
    gate: Gate,
    netlist: Netlist,
    keep_top_k: int = 6,
) -> List[Realization]:
    if gate.type == GateType.NOT:
        reals = solve_not_gate(state, gate, netlist)
    elif gate.type == GateType.NOR:
        reals = solve_nor_gate(state, gate, netlist)
    elif gate.type == GateType.AND:
        reals = solve_and_gate(state, gate, netlist)
    else:
        reals = []

    reals.sort(key=lambda x: (x.copy_cost, x.score_tiebreak))
    return reals[:keep_top_k]


def solve_not_gate(state: State, gate: Gate, netlist: Netlist) -> List[Realization]:
    assert len(gate.inputs) == 1
    inp = gate.inputs[0]
    out = gate.output

    results: List[Realization] = []

    existing_srcs = sorted(state.net_locs.get(inp, set()))
    inp_unwritten_pi = is_unwritten_pi(state, netlist, inp)

    if not existing_srcs and not inp_unwritten_pi:
        return []

    for dst in all_cells(state.rows, state.cols):
        # destination must be writable eventually
        if not can_overwrite_cell(state, dst, protected=set()):
            continue

        if existing_srcs:
            for src in existing_srcs:
                if src == dst:
                    continue
                prep_ops: List[Op] = []
                score = consumer_alignment_score(netlist, out, dst)
                results.append(
                    Realization(
                        copy_cost=0,
                        prep_ops=prep_ops,
                        input_cells=[src],
                        output_cell=dst,
                        score_tiebreak=score,
                    )
                )
        else:
            # unwritten PI: choose any writable src != dst and write it there
            for src in all_cells(state.rows, state.cols):
                if src == dst:
                    continue
                if not can_overwrite_cell(state, src, protected=set()):
                    continue
                prep_ops = [Op(kind="WRITE", gate_gid=gate.gid, net=inp, dst=src)]
                score = consumer_alignment_score(netlist, out, dst)
                results.append(
                    Realization(
                        copy_cost=0,
                        prep_ops=prep_ops,
                        input_cells=[src],
                        output_cell=dst,
                        score_tiebreak=score,
                    )
                )

    return results


def solve_nor_gate(state: State, gate: Gate, netlist: Netlist) -> List[Realization]:
    tokens = list(enumerate(gate.inputs))
    out = gate.output
    results: List[Realization] = []

    for r in range(state.rows):
        row_cells = [(r, c) for c in range(state.cols)]

        # free exact matches
        matched = max_bipartite_matching(tokens, row_cells, state)
        used_input_cells = set(matched.values())

        unmatched = [(tidx, net) for tidx, net in tokens if tidx not in matched]

        # candidate writable cells on row for placing missing inputs
        writable_row_cells = [cell for cell in row_cells if cell not in used_input_cells and can_overwrite_cell(state, cell, protected=used_input_cells)]

        # need one per unmatched input + one output
        if len(writable_row_cells) < len(unmatched) + 1:
            continue

        prep_ops: List[Op] = []
        copy_cost = 0
        input_cells: List[Optional[Coord]] = [None] * len(tokens)

        for tidx, cell in matched.items():
            input_cells[tidx] = cell

        free_cells = list(writable_row_cells)

        feasible = True
        for tidx, net in unmatched:
            cell = free_cells.pop(0)
            if is_unwritten_pi(state, netlist, net):
                prep_ops.append(Op(kind="WRITE", gate_gid=gate.gid, net=net, dst=cell))
            else:
                src = existing_source_for_net(state, net)
                if src is None:
                    feasible = False
                    break
                prep_ops.append(Op(kind="COPY", gate_gid=gate.gid, net=net, src=src, dst=cell))
                copy_cost += 1
            input_cells[tidx] = cell

        if not feasible:
            continue

        # output on same row, different from all input cells
        input_set = set(input_cells)
        output_candidates = [cell for cell in free_cells if cell not in input_set]
        for out_cell in output_candidates:
            score = consumer_alignment_score(netlist, out, out_cell)
            results.append(
                Realization(
                    copy_cost=copy_cost,
                    prep_ops=list(prep_ops),
                    input_cells=list(input_cells),  # type: ignore
                    output_cell=out_cell,
                    score_tiebreak=score,
                )
            )

    return results


def solve_and_gate(state: State, gate: Gate, netlist: Netlist) -> List[Realization]:
    tokens = list(enumerate(gate.inputs))
    k = len(tokens)
    out = gate.output
    results: List[Realization] = []

    if k > state.rows:
        return []

    for c in range(state.cols):
        for start in range(state.rows - k + 1):
            seg = [(start + i, c) for i in range(k)]
            matched = max_bipartite_matching(tokens, seg, state)
            used_input_cells = set(matched.values())
            unmatched = [(tidx, net) for tidx, net in tokens if tidx not in matched]

            # remaining segment cells to fill
            remaining_seg_cells = [cell for cell in seg if cell not in used_input_cells]

            # must be writable because they will be written/copied into before gate
            if not all(can_overwrite_cell(state, cell, protected=used_input_cells) for cell in remaining_seg_cells):
                continue

            output_candidates = []
            if start - 1 >= 0:
                output_candidates.append((start - 1, c))
            if start + k < state.rows:
                output_candidates.append((start + k, c))

            output_candidates = [oc for oc in output_candidates if can_overwrite_cell(state, oc, protected=used_input_cells)]

            if not output_candidates:
                continue

            prep_ops: List[Op] = []
            copy_cost = 0
            input_cells: List[Optional[Coord]] = [None] * k

            for tidx, cell in matched.items():
                input_cells[tidx] = cell

            free_cells = list(remaining_seg_cells)

            feasible = True
            for tidx, net in unmatched:
                cell = free_cells.pop(0)
                if is_unwritten_pi(state, netlist, net):
                    prep_ops.append(Op(kind="WRITE", gate_gid=gate.gid, net=net, dst=cell))
                else:
                    src = existing_source_for_net(state, net)
                    if src is None:
                        feasible = False
                        break
                    prep_ops.append(Op(kind="COPY", gate_gid=gate.gid, net=net, src=src, dst=cell))
                    copy_cost += 1
                input_cells[tidx] = cell

            if not feasible:
                continue

            for out_cell in output_candidates:
                score = consumer_alignment_score(netlist, out, out_cell)
                results.append(
                    Realization(
                        copy_cost=copy_cost,
                        prep_ops=list(prep_ops),
                        input_cells=list(input_cells),  # type: ignore
                        output_cell=out_cell,
                        score_tiebreak=score,
                    )
                )

    return results


# ============================================================
# Apply realization
# ============================================================

def apply_realization(
    state: State,
    gate: Gate,
    realization: Realization,
    netlist: Netlist,
    parents: Dict[int, Set[int]],
    children: Dict[int, Set[int]],
) -> Optional[State]:
    st = state.clone()

    # 1) prep ops: WRITE / COPY
    for op in realization.prep_ops:
        if op.kind == "WRITE":
            assert op.dst is not None and op.net is not None
            if not is_unwritten_pi(st, netlist, op.net):
                return None
            if not can_overwrite_cell(st, op.dst, protected=set()):
                return None
            place_net_in_cell(st, op.dst, op.net, fix_pi=True, fix_po=False)
            st.ops.append(op)

        elif op.kind == "COPY":
            assert op.src is not None and op.dst is not None and op.net is not None
            if st.grid[op.src].net != op.net:
                return None
            if not can_overwrite_cell(st, op.dst, protected=set()):
                return None
            place_net_in_cell(st, op.dst, op.net, fix_pi=False, fix_po=False)
            st.ops.append(op)
            st.copies_so_far += 1

        else:
            return None

    # 2) consume one use for each gate input
    for net in gate.inputs:
        if st.remaining_uses.get(net, 0) <= 0:
            return None
        st.remaining_uses[net] -= 1

    # 3) place gate output
    out_cell = realization.output_cell
    input_cells = tuple(realization.input_cells)

    # output must be different from all input cells
    if out_cell in input_cells:
        return None

    if not can_overwrite_cell(st, out_cell, protected=set(input_cells)):
        return None

    fix_po = gate.output in netlist.primary_outputs
    place_net_in_cell(st, out_cell, gate.output, fix_pi=False, fix_po=fix_po)

    st.ops.append(
        Op(
            kind="GATE",
            gate_gid=gate.gid,
            gate_type=gate.type,
            net=gate.output,
            gate_inputs=input_cells,
            gate_output=out_cell,
        )
    )

    # 4) mark gate executed and update ready
    st.executed.add(gate.gid)
    st.ready.discard(gate.gid)

    for child_gid in children[gate.gid]:
        if child_gid in st.executed:
            continue
        if all(p in st.executed for p in parents[child_gid]):
            st.ready.add(child_gid)

    return st


# ============================================================
# Search
# ============================================================

def gate_branch_priority(state: State, gate: Gate, netlist: Netlist) -> Tuple:
    """
    Smaller tuple = higher priority to expand.
    """
    # Prefer harder gates first: AND, larger fanin, more future impact.
    type_rank = {GateType.AND: 0, GateType.NOR: 1, GateType.NOT: 2}[gate.type]
    fanin_rank = -len(gate.inputs)
    fanout_rank = -len(netlist.consumers.get(gate.output, []))
    is_po = 0 if gate.output in netlist.primary_outputs else 1
    return (type_rank, fanin_rank, fanout_rank, is_po, gate.gid)


def beam_search_mapper(
    netlist: Netlist,
    rows: int,
    cols: int,
    beam_width: int = 50,
    ready_expand_limit: int = 6,
    per_gate_realization_limit: int = 6,
    max_steps: int = 1_000_000,
) -> SolveResult:
    netlist.build_gate_map()
    parents, children = build_DAG(netlist)
    gid_to_gate = gate_map(netlist)

    init = init_state(netlist, rows, cols, parents)

    frontier: List[State] = [init]
    best_complete: Optional[State] = None
    steps = 0

    total_gates = len(netlist.gates)

    while frontier and steps < max_steps:
        steps += 1
        next_frontier: List[State] = []
        seen: Dict[Tuple, int] = {}

        for state in frontier:
            if len(state.executed) == total_gates:
                if best_complete is None or state.copies_so_far < best_complete.copies_so_far:
                    best_complete = state
                continue

            ready_gates = [gid_to_gate[gid] for gid in state.ready]
            ready_gates.sort(key=lambda g: gate_branch_priority(state, g, netlist))
            ready_gates = ready_gates[:ready_expand_limit]

            for gate in ready_gates:
                realizations = local_realizations_for_gate(
                    state, gate, netlist, keep_top_k=per_gate_realization_limit
                )
                for real in realizations:
                    nxt = apply_realization(state, gate, real, netlist, parents, children)
                    if nxt is None:
                        continue

                    sig = state_signature(nxt)
                    prev_best = seen.get(sig)
                    if prev_best is None or nxt.copies_so_far < prev_best:
                        seen[sig] = nxt.copies_so_far
                        next_frontier.append(nxt)

        if not next_frontier:
            break

        # keep best beam_width states by copy count, then by progress
        next_frontier.sort(key=lambda st: (st.copies_so_far, -len(st.executed), len(st.ready)))
        frontier = next_frontier[:beam_width]

        if best_complete is not None:
            # branch-and-bound style prune
            frontier = [st for st in frontier if st.copies_so_far <= best_complete.copies_so_far]

    if best_complete is None:
        return SolveResult(success=False, best_state=None, best_copy_count=None)

    return SolveResult(
        success=True,
        best_state=best_complete,
        best_copy_count=best_complete.copies_so_far,
    )


# ============================================================
# Pretty print
# ============================================================

def print_solution(sol: SolveResult):
    if not sol.success:
        print("INFEASIBLE")
        return

    st = sol.best_state
    print(f"SUCCESS  copy_count={sol.best_copy_count}  total_ops={len(st.ops)}")

    for t, op in enumerate(st.ops):
        if op.kind == "WRITE":
            print(f"{t:4d}: WRITE net={op.net} -> {op.dst}")
        elif op.kind == "COPY":
            print(f"{t:4d}: COPY  net={op.net} {op.src} -> {op.dst}")
        elif op.kind == "GATE":
            print(
                f"{t:4d}: GATE  gid={op.gate_gid} type={op.gate_type.name} "
                f"inputs={op.gate_inputs} output={op.gate_output} outnet={op.net}"
            )


# ============================================================
# Example usage
# ============================================================

def map_netlist_beam_search(netlist: Netlist) -> SolveResult:
    sol = beam_search_mapper(
        netlist,
        rows=4,
        cols=128,
        beam_width=50,
        ready_expand_limit=8,
        per_gate_realization_limit=8,
    )

    print_solution(sol)