"""
Estimated energy/latency for a 32-bit multiplier on SIMPLER, LOGIC, AUTO
(2D-crossbar in-memory mapping papers that only publish an in-memory
operation / "cycle" count, no device-level energy/latency, no
inter-crossbar data-movement cost) versus PRISM (this project's own 3D
mapping result, data/temp2/multiplier32_6x512.json).

None of the three 2D papers report a NOR/NOT split for their reported
cycles, so every reported cycle is conservatively treated as one NOR
operation. Device-level per-operation costs come from the attached gate
characterization table (data/logic_lowest_energy_best_points.csv), using
the "lowest_energy" operating point (not "fastest"), and the
avg_energy_per_gate_j column (already normalized per gate;
avg_total_energy_j is the raw multi-gate testbench reading for NOR/NOT/
AND, identical to avg_energy_per_gate_j for SET/RESET since those are
single-cell operations).

LATENCY: a single flat upper bound (LATENCY_UPPER_BOUND_NS, default
0.1 ns, parameterized) stands in for every operation's own switching
latency. Note this is not a strict upper bound over the lowest_energy
table as given -- AND's own lowest_energy latency is ~0.1085 ns, above
0.1 ns -- so 0.1 ns is a deliberately rounded, not strictly-bounding,
choice; raise --latency-bound if strictness matters more than the round
number.

For the 2D baselines, each reported cycle is 1 gate (SIMPLER's own
Fig.3c: "12 cycles, in accordance with the number of gates" -- no
separate init cycle counted in their convention), and every gate needs
both an init pulse and an execution pulse physically, so each reported
cycle costs TWO bound units: base_latency = total_cycles * 2 * bound.

For PRISM, total_cycles is not "1 cycle = 1 gate" -- it is the mapper's
own actual simulated clock-cycle count, where multiple gates already
share a cycle when executed concurrently (19926 real gate-ops packed
into 15988 cycles). Doubling it here is left OPEN/unresolved: it is not
applied below (latency = total_cycles * bound, not * 2 * bound), on the
assumption that PRISM's own scheduler already places init operations
into their own cycles where physically needed and that this is not
double-counted -- but this has not been confirmed against the
scheduler's actual behavior. Flip PRISM_LATENCY_DOUBLE_INIT below if
that assumption turns out to be wrong.

ENERGY is unaffected by the flat-latency simplification -- it still
uses each operation's own real per-gate energy from the table, plus a
SET or RESET initialization energy added per operation, per gate type.
One convention, used everywhere (2D baselines and PRISM alike):

    NOR / NOT -> RESET-initialized
    AND / COPY -> SET-initialized

This is the OPPOSITE of cli/pim_calculator.py's own comment ("NOR and
NOT require SET initialization... AND and COPY require RESET
initialization") -- flagged here, not silently reconciled; that file
likely needs the same correction if this convention is the right one.

Since every 2D cycle is modeled as a fictional NOR (no real per-paper
gate breakdown exists), only RESET ever applies to the 2D baselines --
SET never appears there. PRISM has real per-gate-type counts (AND/COPY/
NOR/NOT) and uses the same GATE_INIT mapping. COPY is costed identically
to AND's own execution energy (pim_calculator.py's convention for the
execution-energy magnitude, unaffected by the SET/RESET correction
above); NOT is costed using NOR's own gate-execution energy (per this
script's instructions -- NOT's RESET init is still added, only the
execution-energy magnitude is borrowed from NOR). WRITE operations (64,
for the A/B operand bits) are not costed -- there is no WRITE row in the
gate cost table and none was requested.

Inter-crossbar data movement (2D baselines only -- PRISM's 512x6 XZ
footprint is a single physical crossbar, so it has zero movement cost
by construction). This is NOT modeled as "one copy per row-boundary
crossing" (an earlier version of this script did that and undercounted
by ~14x against a live measurement -- see below). Instead it is scaled
directly from a real run of this project's own 2D mapper
(mem3dmapper.mapping.heuristic.map_2D / _map_nor_inv_2D) on
data/netlists/multiplier_nor_netlist/multiplier32_nor.blif
(total_columns=512, matching ROW_CAP elsewhere in this comparison):
that run produced MEASURED_2D_COPIES=11003 COPY operations over
MEASURED_2D_GATES=12612 real NOR/NOT gates (9183 NOR + 3429 NOT),
every one of them empirically confirmed cross-row (i.e. genuinely
inter-crossbar, not same-row alignment) -- see conversation history for
the run log. That gives a measured copies-per-gate rate:

    MEASURED_2D_COPY_RATIO = MEASURED_2D_COPIES / MEASURED_2D_GATES  (~0.872)

For SIMPLER/LOGIC/AUTO, whose own gate-level netlists and mapping code
are not available, each reported cycle count is treated as a proxy
gate count and scaled by this same measured ratio:

    movement_copies = total_cycles * MEASURED_2D_COPY_RATIO

Caveat, not resolved by this script: _map_nor_inv_2D has no
reuse-maximizing ordering heuristic (it samples 1500 random topological
orders and keeps the least-bad); SIMPLER's Strahler/CU ordering and
LOGIC's greedy EIC heuristic exist specifically to minimize this kind
of cross-row copy. Applying our own unoptimized ratio to their reported
cycles is therefore a plausible upper bound on their movement cost, not
a measurement of it -- the true number for their specific algorithms is
unknown without their code, and is bounded below by something closer to
a "one copy per forced row transition" estimate (~1-2 orders of
magnitude smaller; see conversation history).

Each such copy is costed the same way regardless of source: k times a
single NOT operation's own cost (k=K_INTER, parameterized, default 4 --
PRISM's own inter-crossbar communication multiplier, PRISM paper
Sec.8.3). Energy uses NOT's real energy; latency uses the same flat
bound as everything else, not NOT's own switching latency:

    movement_energy_per_copy  = k * NOT_energy_j
    movement_latency_per_copy = k * LATENCY_UPPER_BOUND_NS
    total_movement_cost       = movement_copies * movement_cost_per_copy

Run: python3 multiplier32_2d_baseline_comparison.py [--k 4] [--latency-bound 0.1]
"""

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GATE_COST_CSV = REPO_ROOT / "data" / "logic_lowest_energy_best_points.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "runs" / "multiplier32_2d_baseline_energy_latency.csv"

# ---------------------------------------------------------------------------
# Reported 32-bit multiplication "steps" (in-memory operation / cycle counts)
# ---------------------------------------------------------------------------
# AUTO paper (ICCAD'23), Table IV "Performance Comparison for FiP
# Multiplication", 32-bit column. LOGIC paper (ICCAD'22), Table 1 reports
# the same LOGIC/[8] pair (10046 / 12870).
SIMPLER_CYCLES = 12870  # AUTO/LOGIC's "SIMPLER" row -- actually Haj-Ali'18
                         # ISCAS ("Efficient algorithms..."), not the
                         # Ben-Hur'20 TCAD SIMPLER MAGIC paper, which does
                         # not itself publish a 32-bit multiplier number.
LOGIC_CYCLES = 10046
AUTO_CYCLES = 8462

# LOGIC/AUTO's reported steps cover only the partial-product ADDITION
# stage: partial products themselves (1024 AND terms for 32x32) are
# assumed already generated. Under a NOR/NOT-only MAGIC library,
# AND(a,b) = NOR(NOT a, NOT b) costs 3 ops/bit (2 NOT + 1 NOR) ->
# 1024 * 3 = 3072 cycles, added to LOGIC and AUTO only (as instructed --
# SIMPLER's reported 12870 is used as-is).
PARTIAL_PRODUCT_CYCLES = 3072

ROW_CAP = 512  # cells per crossbar/row; no reuse -> 1 cycle = 1 cell (informational only now)
LATENCY_UPPER_BOUND_NS = 0.1  # flat per-operation latency bound (see docstring)
PRISM_LATENCY_DOUBLE_INIT = False  # see docstring -- left unresolved, default off

# Measured from a live run of this project's own 2D mapper (map_2D /
# _map_nor_inv_2D) on multiplier_nor_netlist/multiplier32_nor.blif,
# total_columns=512. Every COPY op in that run was empirically confirmed
# cross-row. See module docstring for the caveat on applying this ratio
# to SIMPLER/LOGIC/AUTO (plausible upper bound, not a measurement, for
# those three).
MEASURED_2D_COPIES = 11003
MEASURED_2D_GATES = 12612  # 9183 NOR + 3429 NOT
MEASURED_2D_COPY_RATIO = MEASURED_2D_COPIES / MEASURED_2D_GATES

METHODS_2D = {
    "SIMPLER": SIMPLER_CYCLES,
    "LOGIC": LOGIC_CYCLES + PARTIAL_PRODUCT_CYCLES,
    "AUTO": AUTO_CYCLES + PARTIAL_PRODUCT_CYCLES,
}

# Single init convention, used for both the 2D fiction and PRISM's real
# gates (see module docstring -- opposite of pim_calculator.py's comment).
GATE_INIT = {"NOR": "RESET", "NOT": "RESET", "AND": "SET", "COPY": "SET"}

# PRISM: real per-gate-type counts.
# data/temp2/multiplier32_6x512.json -> operations_count.
PRISM_OPS = {"WRITE": 64, "AND": 3265, "COPY": 10608, "NOR": 5119, "NOT": 934}
PRISM_TOTAL_CYCLES = 15988  # the JSON's "total" field (== max cycle index);
                             # the 15998 in chat was a typo for this value.
# COPY's execution energy is costed the same as AND's (pim_calculator.py
# convention); NOT's execution energy is costed using NOR's value (as
# instructed for this script) -- its init type (RESET, from GATE_INIT)
# is unaffected by that substitution.
PRISM_EXEC_ENERGY_SOURCE = {"AND": "AND", "COPY": "AND", "NOR": "NOR", "NOT": "NOR"}


def load_gate_costs(csv_path: Path, selection: str = "lowest_energy"):
    """Returns {operation: (latency_ns, energy_j_per_gate)} for the given
    selection ('lowest_energy' or 'fastest')."""
    costs = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["selection"] != selection:
                continue
            op = row["operation"]
            latency_ns = float(row["avg_switch_latency_ns"])
            energy_j = float(row["avg_energy_per_gate_j"])
            costs[op] = (latency_ns, energy_j)
    missing = {"NOR", "NOT", "AND", "SET", "RESET"} - costs.keys()
    if missing:
        raise ValueError(f"gate cost table missing operations: {missing}")
    return costs


def compute_2d_result(name, total_cycles, gate_costs, row_cap, k_inter, latency_bound_ns):
    # Energy: real NOR execution energy + real init energy (GATE_INIT).
    _, nor_energy = gate_costs["NOR"]
    _, init_energy = gate_costs[GATE_INIT["NOR"]]
    cycle_energy_j = nor_energy + init_energy
    base_energy_j = total_cycles * cycle_energy_j
    # Latency: each cycle = 1 gate = 1 init pulse + 1 exec pulse -> 2 bound units.
    base_latency_ns = total_cycles * 2 * latency_bound_ns

    # Informational only -- no longer used to derive movement (see docstring).
    n_crossbars = -(-total_cycles // row_cap)  # ceil division
    n_boundaries = max(n_crossbars - 1, 0)

    movement_copies = total_cycles * MEASURED_2D_COPY_RATIO
    _, not_energy_j = gate_costs["NOT"]
    move_energy_per_copy_j = k_inter * not_energy_j
    move_latency_per_copy_ns = k_inter * latency_bound_ns
    movement_latency_ns = movement_copies * move_latency_per_copy_ns
    movement_energy_j = movement_copies * move_energy_per_copy_j

    total_latency_ns = base_latency_ns + movement_latency_ns
    total_energy_j = base_energy_j + movement_energy_j

    return {
        "method": name,
        "total_cycles": total_cycles,
        "n_crossbars": n_crossbars,
        "n_boundaries": n_boundaries,
        "movement_copies": movement_copies,
        "base_latency_ns": base_latency_ns,
        "base_energy_j": base_energy_j,
        "movement_latency_ns": movement_latency_ns,
        "movement_energy_j": movement_energy_j,
        "total_latency_ns": total_latency_ns,
        "total_energy_j": total_energy_j,
        "total_latency_us": total_latency_ns / 1000.0,
        "total_energy_pj": total_energy_j * 1e12,
    }


def compute_prism_result(gate_costs, latency_bound_ns):
    total_energy_j = 0.0
    for op, count in PRISM_OPS.items():
        if op == "WRITE":
            continue  # no WRITE row in the gate cost table, not requested
        _, exec_energy = gate_costs[PRISM_EXEC_ENERGY_SOURCE[op]]
        _, init_energy = gate_costs[GATE_INIT[op]]
        total_energy_j += count * (exec_energy + init_energy)

    # Latency: total_cycles already reflects PRISM's own concurrent
    # scheduling (multiple gates can share a cycle). Whether to also
    # double it for a separate init term per cycle is left open -- see
    # docstring and PRISM_LATENCY_DOUBLE_INIT.
    latency_multiplier = 2 if PRISM_LATENCY_DOUBLE_INIT else 1
    total_latency_ns = PRISM_TOTAL_CYCLES * latency_multiplier * latency_bound_ns

    return {
        "method": "PRISM",
        "total_cycles": PRISM_TOTAL_CYCLES,
        "n_crossbars": 1,
        "n_boundaries": 0,
        "movement_copies": 0,
        "base_latency_ns": total_latency_ns,
        "base_energy_j": total_energy_j,
        "movement_latency_ns": 0.0,
        "movement_energy_j": 0.0,
        "total_latency_ns": total_latency_ns,
        "total_energy_j": total_energy_j,
        "total_latency_us": total_latency_ns / 1000.0,
        "total_energy_pj": total_energy_j * 1e12,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gate-costs-csv", type=Path, default=DEFAULT_GATE_COST_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--k", type=float, default=4.0,
                         help="inter-crossbar communication multiplier "
                              "(PRISM's own k, default 4)")
    parser.add_argument("--row-cap", type=int, default=ROW_CAP,
                         help="cells per crossbar/row (default 512; "
                              "informational only, no longer drives the "
                              "movement count -- see docstring)")
    parser.add_argument("--latency-bound", type=float, default=LATENCY_UPPER_BOUND_NS,
                         help="flat per-operation latency bound in ns "
                              "(default 0.1; not a strict upper bound over "
                              "the lowest_energy table -- AND is ~0.1085 ns)")
    args = parser.parse_args()

    gate_costs = load_gate_costs(args.gate_costs_csv)

    results = [
        compute_2d_result(name, cycles, gate_costs, args.row_cap, args.k, args.latency_bound)
        for name, cycles in METHODS_2D.items()
    ]
    results.append(compute_prism_result(gate_costs, args.latency_bound))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"k={args.k}, copy_ratio={MEASURED_2D_COPY_RATIO:.4f}, "
          f"row_cap={args.row_cap}, latency_bound={args.latency_bound}ns, "
          f"PRISM_LATENCY_DOUBLE_INIT={PRISM_LATENCY_DOUBLE_INIT}")
    print(f"{'method':<10}{'cycles':>8}{'copies':>9}"
          f"{'base_lat(ns)':>14}{'move_lat(ns)':>14}{'total_lat(us)':>15}"
          f"{'base_e(pJ)':>13}{'move_e(pJ)':>13}{'total_e(pJ)':>14}")
    for r in results:
        print(f"{r['method']:<10}{r['total_cycles']:>8}{r['movement_copies']:>9.0f}"
              f"{r['base_latency_ns']:>14.2f}"
              f"{r['movement_latency_ns']:>14.2f}{r['total_latency_us']:>15.3f}"
              f"{r['base_energy_j']*1e12:>13.2f}{r['movement_energy_j']*1e12:>13.2f}"
              f"{r['total_energy_pj']:>14.2f}")
    print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()
