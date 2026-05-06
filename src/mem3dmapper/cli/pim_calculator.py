"""
KV-PIM architecture calculator.

Turns PRISM's measured per-kernel numbers into the KV-Block / context-length
/ batch-size tables defined in 3D_MAGIC_KV_PIM_Minimal_System_Architecture
(v4), sections 9, 10, 13.

STATUS: partially real data, partially placeholder. Every TODO below is a
number that needs to come from you / the PRISM tool, not from me.
"""

from dataclasses import dataclass
from math import ceil

# ---------------------------------------------------------------------------
# 1. PRISM device-level parameters (PRISM paper Table 4, confirmed valid).
#    AND/NAND execution latency/energy conservatively assumed == NOR's,
#    per PRISM paper §8.3's own stated conservative assumption.
# ---------------------------------------------------------------------------
E_NOR_fJ = 0.29
E_SET_fJ = 23.8
E_RESET_fJ = 0.32
T_NOR_ns = 1.1
T_SET_RESET_ns = 1.0

# A generic gate-execution cycle (NOR/NOT/AND/NAND, assumed equal cost):
T_GATE_ns = T_NOR_ns
E_GATE_fJ = E_NOR_fJ

# Initialization-aware operation costs:
#   AND and COPY require RESET initialization.
#   NOR and NOT require SET initialization.
T_INITIALIZED_GATE_ns = T_SET_RESET_ns + T_GATE_ns
E_SET_INITIALIZED_GATE_fJ = E_SET_fJ + E_GATE_fJ
E_RESET_INITIALIZED_GATE_fJ = E_RESET_fJ + E_GATE_fJ

T_AND_ns = T_INITIALIZED_GATE_ns
E_AND_fJ = E_RESET_INITIALIZED_GATE_fJ
T_COPY_ns = T_INITIALIZED_GATE_ns
E_COPY_fJ = E_RESET_INITIALIZED_GATE_fJ
T_NOR_OP_ns = T_INITIALIZED_GATE_ns
E_NOR_OP_fJ = E_SET_INITIALIZED_GATE_fJ
T_NOT_ns = T_INITIALIZED_GATE_ns
E_NOT_fJ = E_SET_INITIALIZED_GATE_fJ

# Inter-crossbar communication multiplier, PRISM paper §8.3.
K_INTER = 4


# ---------------------------------------------------------------------------
# 2. Measured PRISM numbers for dot_N16_B8 (16-element, 8-bit chunk dot
#    product) -- the kernel used for both K-side and V-side chunks per the
#    architecture doc (§4.1, §8.1).
# ---------------------------------------------------------------------------
@dataclass
class KernelCost:
    xz_rows: int
    xz_cols: int
    total_cycles: int
    write_count: int
    not_count: int
    and_count: int
    copy_count: int
    nor_count: int
    peak_live_cells: int | None = None

    @property
    def gate_count(self) -> int:
        return self.not_count + self.and_count + self.copy_count + self.nor_count

    @property
    def reset_initialized_count(self) -> int:
        return self.and_count + self.copy_count

    @property
    def set_initialized_count(self) -> int:
        return self.nor_count + self.not_count


DOT_N16_B8 = KernelCost(
    xz_rows=512,
    xz_cols=6,
    total_cycles=15811,
    write_count=256,
    not_count=1172,
    and_count=3284,
    copy_count=10185,
    nor_count=5814,
    peak_live_cells=1639,
)

copy_fraction = DOT_N16_B8.copy_count / DOT_N16_B8.total_cycles
gate_execution_energy_fJ = DOT_N16_B8.gate_count * E_GATE_fJ
set_initialization_energy_fJ = DOT_N16_B8.set_initialized_count * E_SET_fJ
reset_initialization_energy_fJ = DOT_N16_B8.reset_initialized_count * E_RESET_fJ
initialization_energy_fJ = set_initialization_energy_fJ + reset_initialization_energy_fJ

# NOTE on 3D crossbar geometry (clarified): the 512x6 XZ footprint is NOT
# something that needs to "fit inside" a flat 512x512 2D crossbar. X=512 is
# the same row/wordline dimension as the 2D case-study crossbar; Z=6 is how
# many layers this specific kernel's mapping needed to stack; Y is a THIRD,
# independent physical axis along which the identical XZ circuit is
# replicated. One logical crossbar instance (one of the 8 K or 16 V chunks)
# is a self-contained X x Y x Z physical device -- there is no separate
# "packing into a bigger 2D crossbar" question for this kernel.
peak_utilization = DOT_N16_B8.peak_live_cells / (DOT_N16_B8.xz_rows * DOT_N16_B8.xz_cols)

# --- Latency/energy estimates, using the Y-replication assumption flagged above ---
gate_execution_latency_ns = DOT_N16_B8.total_cycles * T_GATE_ns
initialization_latency_ns = DOT_N16_B8.total_cycles * T_SET_RESET_ns
latency_one_instance_ns = gate_execution_latency_ns + initialization_latency_ns
energy_one_instance_fJ = (
    DOT_N16_B8.and_count * E_AND_fJ
    + DOT_N16_B8.copy_count * E_COPY_fJ
    + DOT_N16_B8.nor_count * E_NOR_OP_fJ
    + DOT_N16_B8.not_count * E_NOT_fJ
)

print("=== dot_N16_B8 kernel cost (Y=1, i.e. one 16-elem chunk, one token) ===")
print(f"  XZ footprint          : {DOT_N16_B8.xz_rows} x {DOT_N16_B8.xz_cols}")
print(f"  Peak live cells       : {DOT_N16_B8.peak_live_cells}  "
      f"({peak_utilization:.1%} of the {DOT_N16_B8.xz_rows*DOT_N16_B8.xz_cols}-cell XZ plane)")
print(f"  Total cycles           : {DOT_N16_B8.total_cycles}")
print(f"  WRITE operations        : {DOT_N16_B8.write_count}")
print(f"  AND operations          : {DOT_N16_B8.and_count}")
print(f"  COPY operations         : {DOT_N16_B8.copy_count} "
      f"({copy_fraction:.1%} of total)")
print(f"  NOR operations          : {DOT_N16_B8.nor_count}")
print(f"  NOT operations          : {DOT_N16_B8.not_count}")
print(f"  SET init operations     : {DOT_N16_B8.set_initialized_count}")
print(f"  RESET init operations   : {DOT_N16_B8.reset_initialized_count}")
print(f"  Gate-execution latency  : {gate_execution_latency_ns:.1f} ns")
print(f"  Initialization latency  : {initialization_latency_ns:.1f} ns")
print(f"  Estimated latency       : {latency_one_instance_ns:.1f} ns  "
      f"[PRELIMINARY - see TODOs above]")
print(f"  Gate-execution energy   : {gate_execution_energy_fJ/1000:.2f} pJ")
print(f"  Initialization energy   : {initialization_energy_fJ/1000:.2f} pJ")
print(f"  Estimated energy        : {energy_one_instance_fJ/1000:.2f} pJ  "
      f"[PRELIMINARY - see TODOs above]")
print()
print("ASSUMPTION IN USE (CONFIRMED by user): the numbers above are for Y=1")
print("(one token, one instance). Latency is unaffected by Y-replication")
print("(all Y-lanes fire the same pulse schedule concurrently). Energy DOES")
print("scale with Y (each replica is a physically separate cell drawing its")
print("own current) -- applied below, distinctly for K-side (Y=256 tokens)")
print("and V-side (Y=128 output features = d_h), per architecture doc §4.2/§8.2.")
print()


# ---------------------------------------------------------------------------
# 3. KV Block resource model (architecture doc §9).
#    K side: 8 crossbars (128 = 8x16), each = one dot_N16_B8 instance.
#    V side: 16 crossbars (256 = 16x16), each ALSO approximated as one
#    dot_N16_B8 instance (UINT8 x INT8 instead of INT8 x INT8 -- flagged as
#    an approximation; ideally re-run PRISM for the V-side operand types).
# ---------------------------------------------------------------------------
N_K_CROSSBARS = 8
N_V_CROSSBARS = 16
LOGICAL_CROSSBARS_PER_KV_BLOCK = N_K_CROSSBARS + N_V_CROSSBARS  # 24

# Y-replication factors -- DISTINCT for K-side and V-side, per architecture
# doc §4.2 (K: Y = 256 tokens) and §8.2 (V: Y = output-feature index, only
# d_h=128 of the available 256 Y-lanes used, since d_h=128).
Y_K = 256
Y_V = 128  # = d_h

# K-side crossbars run in parallel (independent chunks, no data dependency
# until the CMOS reduction) -> QK^T score LATENCY for one 256-token block
# is dominated by ONE dot_N16_B8 instance's latency, not 8x it, and NOT
# multiplied by Y_K either (Y-replication is free in time, per above).
# ENERGY, however, must be multiplied by both the chunk count (8) AND Y_K
# (256) -- previously this file only multiplied by chunk count, silently
# understating K-side energy by 256x. Fixed below.
# TODO(user): add CMOS 8-to-1 reduction adder-tree latency/energy (currently
# treated as negligible relative to the crossbar latency above -- confirm).
qkt_block_latency_ns = latency_one_instance_ns
qkt_block_energy_fJ = energy_one_instance_fJ * N_K_CROSSBARS * Y_K

# Same logic for V side, but with Y_V=128 instead of Y_K=256 -- this file
# previously omitted this multiplier too (understating V-side energy 128x).
# TODO(user): add CMOS 16-to-1 reduction + final scale-correction cost
# (architecture doc §8.1) -- currently treated as negligible.
av_block_latency_ns = latency_one_instance_ns
av_block_energy_fJ = energy_one_instance_fJ * N_V_CROSSBARS * Y_V

attention_block_latency_ns = qkt_block_latency_ns + av_block_latency_ns
# TODO(user): add online-softmax peripheral latency/energy (§7) -- currently
# treated as negligible relative to crossbar latency; confirm this is fair.
attention_block_energy_fJ = qkt_block_energy_fJ + av_block_energy_fJ

print("=== One KV Block (one KV head, 256 tokens) ===")
print(f"  Logical crossbars       : {LOGICAL_CROSSBARS_PER_KV_BLOCK} "
      f"({N_K_CROSSBARS} K + {N_V_CROSSBARS} V)")
print(f"  QK^T block latency       : {qkt_block_latency_ns:.1f} ns")
print(f"  AV block latency         : {av_block_latency_ns:.1f} ns")
print(f"  Attention block latency  : {attention_block_latency_ns:.1f} ns")
print(f"  Attention block energy   : {attention_block_energy_fJ/1000:.2f} pJ")
print()


# ---------------------------------------------------------------------------
# 4. Context-length scaling (architecture doc §10) -- single request.
# ---------------------------------------------------------------------------
Y = 256
CONTEXTS = [4096, 16384, 32768, 131072]  # 4K, 16K, 32K, 128K
N_KV = 8  # LLaMA-3.1-8B

print("=== Context-length scaling (single request, N_KV=8) ===")
print(f"{'Context':>10} {'Blocks/head':>12} {'Blocks/req':>11} "
      f"{'Logical Xbars/req':>18} {'Attn latency/req (us)':>22}")
for L in CONTEXTS:
    blocks_per_head = ceil(L / Y)
    blocks_per_req = blocks_per_head * N_KV
    xbars_per_req = blocks_per_req * LOGICAL_CROSSBARS_PER_KV_BLOCK
    # NOTE: if all KV heads' blocks execute in parallel across independent
    # physical crossbars, total latency per decode step stays ~constant at
    # attention_block_latency_ns, NOT multiplied by blocks_per_req --
    # this is the same "spatial parallelism is free in latency, costs area"
    # property as Y-replication. TODO(user): confirm this holds, i.e. that
    # enough physical crossbars are assumed available; if blocks execute
    # serially due to a resource cap, multiply latency by blocks_per_req
    # instead.
    latency_per_req_us = attention_block_latency_ns / 1000  # assumes parallel
    ctx_label = f"{L//1024}K"
    print(f"{ctx_label:>10} {blocks_per_head:>12} {blocks_per_req:>11} "
          f"{xbars_per_req:>18,} {latency_per_req_us:>22.3f}")
print()
print("Latency column above ASSUMES unlimited physical crossbar parallelism")
print("across all KV blocks of a request (only area scales, not latency).")
print("If your physical crossbar budget is smaller than xbars_per_req, some")
print("blocks must serialize -- add a physical-crossbar-budget cap here once")
print("you decide the accelerator's total crossbar count (architecture doc")
print("§2 item 4 / §13.4).")
print()


# ---------------------------------------------------------------------------
# 5. Batch-size scaling (architecture doc §13.2, §13.3).
# ---------------------------------------------------------------------------
BATCH_SIZES = [1, 8, 32, 128]

print("=== Batch-size scaling (context = 4K, N_KV=8) ===")
print(f"{'Batch':>6} {'Total KV Blocks':>16} {'Total logical Xbars':>20}")
L = 4096
blocks_per_req = ceil(L / Y) * N_KV
for B in BATCH_SIZES:
    total_blocks = B * blocks_per_req
    total_xbars = total_blocks * LOGICAL_CROSSBARS_PER_KV_BLOCK
    print(f"{B:>6} {total_blocks:>16,} {total_xbars:>20,}")
print()
print("Cross-reference total_xbars against whatever total physical crossbar")
print("budget you pick (architecture doc §2 item 4) to get max batch size")
print("under a capacity constraint -- this is the AttAcc-Fig-4a-equivalent")
print("number the paper needs.")