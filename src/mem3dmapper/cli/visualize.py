"""
Improved mapping visualizer:

Fixes your 3 pain points:
1) Writes the operation name + cycle + details on the frame (title + text panel).
2) Shows input/output NET NAMES (not just colors) at the relevant cells.
   - By default: label ONLY the cells touched in this cycle (src + dst), so it stays readable.
   - Optional: label all occupied cells (only practical for tiny grids).
3) Produces an interactive scrubber (go prev/next cycle) AND an MP4 video export.
   - Notebook: ipywidgets slider + Prev/Next buttons
   - Script: save_frames + save_mp4

Requirements:
  pip install matplotlib numpy ipywidgets
Optional for MP4:
  pip install imageio imageio-ffmpeg

Drop-in usage:
  viz = CrossbarVisualizer.from_json_path("mapping.json")
  viz.show_widget()            # interactive scrub (best)
  viz.save_frames("frames")    # png frames
  viz.save_mp4("mapping.mp4")  # video you can scrub in any player
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


Coordinate = Tuple[int, int]


class OperationType(str, Enum):
    WRITE = "WRITE"
    COPY = "COPY"
    NOT = "NOT"
    NOR = "NOR"
    AND = "AND"


@dataclass(frozen=True)
class Operation:
    type: OperationType
    net: str
    cycle: int
    location: Coordinate
    inputs: Optional[Tuple[str, ...]] = None
    src: Optional[List[Coordinate]] = None


def _as_coord(xy: Any) -> Coordinate:
    return int(xy[0]), int(xy[1])


def load_ops(path: str) -> List[Operation]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f).get("execution")
    ops: List[Operation] = []
    for d in data:
        op = Operation(
            type=OperationType(d["type"]),
            net=str(d["net"]),
            cycle=int(d["cycle"]),
            location=_as_coord(d["location"]),
            inputs=None if d.get("inputs") is None else tuple(map(str, d["inputs"])),
            src=None if d.get("src") is None else [_as_coord(xy) for xy in d["src"]],
        )
        ops.append(op)
    ops.sort(key=lambda o: (o.cycle,))
    return ops


@dataclass
class OverwriteEvent:
    cycle: int
    dst: Coordinate
    old_net: str
    new_net: str


class CrossbarVisualizer:
    def __init__(self, ops: List[Operation]):
        if not ops:
            raise ValueError("Empty ops.")
        self.ops = ops
        self.cycles = sorted({o.cycle for o in ops})
        self.ops_by_cycle: Dict[int, List[Operation]] = {}
        for o in ops:
            self.ops_by_cycle.setdefault(o.cycle, []).append(o)

        self.min_x, self.max_x, self.min_y, self.max_y = self._bounds()
        self.w = self.max_x - self.min_x + 1
        self.h = self.max_y - self.min_y + 1

        self.states: Dict[int, Dict[Coordinate, str]] = {}
        self.overwrites: List[OverwriteEvent] = []
        self._replay()

    @classmethod
    def from_json_path(cls, path: str) -> "CrossbarVisualizer":
        return cls(load_ops(path))

    def _bounds(self):
        xs, ys = [], []
        for op in self.ops:
            xs.append(op.location[0]); ys.append(op.location[1])
            if op.src:
                for (sx, sy) in op.src:
                    xs.append(sx); ys.append(sy)
        return min(xs), max(xs), min(ys), max(ys)

    def _replay(self):
        occ: Dict[Coordinate, str] = {}

        def set_cell(c: Coordinate, net: str, cycle: int):
            old = occ.get(c)
            if old is not None and old != net:
                self.overwrites.append(OverwriteEvent(cycle, c, old, net))
            occ[c] = net

        for cyc in self.cycles:
            for op in self.ops_by_cycle.get(cyc, []):
                # all ops write their output net to op.location
                set_cell(op.location, op.net, cyc)
            self.states[cyc] = dict(occ)

    def _coord_to_rc(self, c: Coordinate) -> Tuple[int, int]:
        x, y = c
        return (y - self.min_y), (x - self.min_x)

    def _op_summary(self, op: Operation) -> str:
        ins = "" if not op.inputs else f"  inputs={list(op.inputs)}"
        return f"{op.type.value}  out={op.net}@{op.location}{ins}"

    def _cycle_overwrites(self, cycle: int) -> List[OverwriteEvent]:
        return [e for e in self.overwrites if e.cycle == cycle]

    # ---------- Rendering ----------
    def render_cycle_fixed_layout(
        self,
        cycle: int,
        *,
        annotate_all_cells: bool = False,
        annotate_touched_only: bool = True,
        max_label_len: int = 14,
        figsize=(11, 6),
        dpi: int = 140,
        save_path: str | None = None,
        show: bool = True,
    ):
        import numpy as np
        import matplotlib.pyplot as plt

        state = self.states[cycle]
        ops = self.ops_by_cycle.get(cycle, [])

        # background grid: 0 empty, 1 occupied
        grid = np.zeros((self.h, self.w), dtype=int)
        for (x, y), _net in state.items():
            r, c = self._coord_to_rc((x, y))
            grid[r, c] = 1

        # --- FIXED LAYOUT: grid axis + text axis (constant sizes) ---
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4.5, 2.0], wspace=0.05)

        ax = fig.add_subplot(gs[0, 0])       # grid
        ax_txt = fig.add_subplot(gs[0, 1])   # side panel
        ax_txt.axis("off")

        ax.imshow(grid, interpolation="nearest", aspect="equal")  # equal locks geometry

        # lock axes bounds (prevents any autoscale jitter)
        ax.set_xlim(-0.5, self.w - 0.5)
        ax.set_ylim(self.h - 0.5, -0.5)  # y down

        # fixed ticks/labels
        ax.set_xticks(range(self.w))
        ax.set_yticks(range(self.h))
        ax.set_xticklabels([str(self.min_x + i) for i in range(self.w)], fontsize=9)
        ax.set_yticklabels([str(self.min_y + i) for i in range(self.h)], fontsize=9)
        ax.set_xlabel("x (columns)")
        ax.set_ylabel("y (rows)")

        title_ops = " | ".join([o.type.value for o in ops]) if ops else "NO-OP"
        ax.set_title(f"Cycle {cycle} — {title_ops}")

        # cells to annotate (only touched cells)
        touched = set()
        for op in ops:
            if op.src:
                for s in op.src:
                    touched.add(tuple(s))
            touched.add(tuple(op.location))

        def short(s: str) -> str:
            return s if len(s) <= max_label_len else (s[: max_label_len - 1] + "…")

        # 1) Background labels for full state (faint + behind)
        if annotate_all_cells:
            for (x, y), net in state.items():
                r, c = self._coord_to_rc((x, y))
                ax.text(
                    c, r, short(net),
                    ha="center", va="center",
                    fontsize=7,
                    alpha=0.35,     # faint
                    zorder=2,       # behind
                )

        # 2) Current-op labels (bold + white box + on top)
        if annotate_touched_only:
            for cxy in touched:
                net = state.get(cxy)
                if net is None:
                    continue
                r, c = self._coord_to_rc(cxy)
                ax.text(
                    c, r, short(net),
                    ha="center", va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="black",
                    zorder=6,  # on top of everything
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.85,
                        boxstyle="round,pad=0.15",
                    ),
                )


        # draw src/dst boxes + arrows
        for op in ops:
            dr, dc = self._coord_to_rc(op.location)
            ax.add_patch(plt.Rectangle((dc - 0.5, dr - 0.5), 1, 1, fill=False, linewidth=2, zorder=4))
            if op.src:
                for s in op.src:
                    sr, sc = self._coord_to_rc(s)
                    ax.add_patch(plt.Rectangle((sc - 0.5, sr - 0.5), 1, 1,
                                            fill=False, linewidth=2, linestyle="--"))
                    ax.annotate("", xy=(dc, dr), xytext=(sc, sr),
                                arrowprops=dict(arrowstyle="->", lw=1), zorder=4)

        # overwrite marker
        ow = [e for e in self.overwrites if e.cycle == cycle]
        for e in ow:
            r, c = self._coord_to_rc(e.dst)
            ax.text(c, r - 0.35, "OW", ha="center", va="center",
                    fontsize=10, fontweight="bold")

        # --- side panel text (never changes grid size now) ---
        panel_lines = ["Ops:"]
        for op in ops:
            ins = "" if not op.inputs else f" inputs={list(op.inputs)}"
            panel_lines.append(f"• {op.type.value} out={short(op.net)}@{op.location}{ins}")
            if op.src:
                panel_lines.append(f"    src={op.src}")

        if ow:
            panel_lines.append("")
            panel_lines.append("Overwrites:")
            for e in ow:
                panel_lines.append(f"• {e.dst}: {short(e.old_net)} → {short(e.new_net)}")

        ax_txt.text(
            0.0, 1.0, "\n".join(panel_lines),
            va="top", ha="left",
            fontsize=9, family="monospace"
        )

        # IMPORTANT: no tight_layout, no bbox_inches="tight"
        if save_path:
            fig.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def render_cycle(
        self,
        cycle: int,
        *,
        annotate_touched_only: bool = True,
        annotate_all_cells: bool = False,
        max_label_len: int = 14,
        figsize=(9, 5),
        dpi: int = 140,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        import numpy as np
        import matplotlib.pyplot as plt

        state = self.states[cycle]
        ops = self.ops_by_cycle.get(cycle, [])

        # background grid: 0 empty, 1 occupied (keeps it readable)
        grid = np.zeros((self.h, self.w), dtype=int)
        for (x, y), _net in state.items():
            r, c = self._coord_to_rc((x, y))
            grid[r, c] = 1

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(grid, interpolation="nearest", aspect="equal")

        # axes labels in absolute coords
        ax.set_xticks(range(self.w))
        ax.set_yticks(range(self.h))
        ax.set_xticklabels([str(self.min_x + i) for i in range(self.w)])
        ax.set_yticklabels([str(self.min_y + i) for i in range(self.h)])
        ax.set_xlabel("x (columns)")
        ax.set_ylabel("y (rows)")

        # Title: include op types + key nets
        title_ops = " | ".join([o.type.value for o in ops]) if ops else "NO-OP"
        ax.set_title(f"Cycle {cycle} — {title_ops}")

        # Determine which cells to annotate
        touched: List[Coordinate] = []
        for op in ops:
            if op.src:
                touched.extend(op.src)
            touched.append(op.location)

        touched_set = set(touched)

        def short(s: str) -> str:
            return s if len(s) <= max_label_len else (s[: max_label_len - 1] + "…")

        # annotate cell labels
        if annotate_all_cells:
            for (x, y), net in state.items():
                r, c = self._coord_to_rc((x, y))
                ax.text(c, r, short(net), ha="center", va="center", fontsize=8)
        elif annotate_touched_only:
            for cxy in touched_set:
                net = state.get(cxy)
                if net is None:
                    continue
                r, c = self._coord_to_rc(cxy)
                ax.text(c, r, short(net), ha="center", va="center", fontsize=9, fontweight="bold")

        # draw boxes: src dashed, dst solid
        for op in ops:
            # dst
            dr, dc = self._coord_to_rc(op.location)
            ax.add_patch(plt.Rectangle((dc - 0.5, dr - 0.5), 1, 1, fill=False, linewidth=2, zorder=4))
            # src
            if op.src:
                for s in op.src:
                    sr, sc = self._coord_to_rc(s)
                    ax.add_patch(
                        plt.Rectangle((sc - 0.5, sr - 0.5), 1, 1, fill=False, linewidth=2, linestyle="--", zorder=4)
                    )
                    # arrow from src to dst (helps a lot)
                    ax.annotate(
                        "",
                        xy=(dc, dr),
                        xytext=(sc, sr),
                        arrowprops=dict(arrowstyle="->", lw=1, zorder=4),
                    )

        # overwrite marker
        ow = self._cycle_overwrites(cycle)
        if ow:
            for e in ow:
                r, c = self._coord_to_rc(e.dst)
                ax.text(c, r - 0.35, "OVERWRITE", ha="center", va="center", fontsize=8, fontweight="bold")

        # operation text panel (right side)
        panel_lines = ["Ops:"]
        for op in ops:
            panel_lines.append("• " + self._op_summary(op))
        if ow:
            panel_lines.append("")
            panel_lines.append("Overwrites:")
            for e in ow:
                panel_lines.append(f"• {e.dst}: {short(e.old_net)} → {short(e.new_net)}")

        ax.text(
            1.02,
            0.98,
            "\n".join(panel_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ---------- Notebook interactive scrubber ----------
    def show_widget(self, *, annotate_touched_only=True, max_label_len=14, figsize=(9, 5)):
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except Exception as e:
            raise RuntimeError("show_widget requires ipywidgets + IPython (Jupyter).") from e

        cycles = self.cycles
        out = widgets.Output()

        slider = widgets.IntSlider(
            value=cycles[0],
            min=cycles[0],
            max=cycles[-1],
            step=1,
            description="Cycle",
            continuous_update=False,
        )
        btn_prev = widgets.Button(description="Prev")
        btn_next = widgets.Button(description="Next")

        def draw(cyc: int):
            with out:
                clear_output(wait=True)
                self.render_cycle_fixed_layout(
                    cyc,
                    annotate_touched_only=annotate_touched_only,
                    annotate_all_cells=True,
                    max_label_len=max_label_len,
                    figsize=figsize,
                    show=True,
                )

        def on_prev(_):
            slider.value = max(slider.min, slider.value - 1)

        def on_next(_):
            slider.value = min(slider.max, slider.value + 1)

        def on_slide(change):
            if change["name"] == "value":
                draw(int(change["new"]))

        btn_prev.on_click(on_prev)
        btn_next.on_click(on_next)
        slider.observe(on_slide)

        display(widgets.HBox([btn_prev, btn_next, slider]), out)
        draw(slider.value)

    # ---------- Export ----------
    def save_frames(
        self,
        outdir: str,
        *,
        annotate_touched_only: bool = True,
        max_label_len: int = 14,
        figsize=(9, 5),
        dpi: int = 140,
    ):
        os.makedirs(outdir, exist_ok=True)
        for cyc in self.cycles:
            path = os.path.join(outdir, f"cycle_{cyc:05d}.png")
            self.render_cycle_fixed_layout(
                cyc,
                annotate_touched_only=annotate_touched_only,
                annotate_all_cells=True,
                max_label_len=max_label_len,
                figsize=figsize,
                dpi=dpi,
                save_path=path,
                show=False,
            )

    def save_mp4(self, mp4_path: str, *, fps: int = 4, tmp_frames_dir: Optional[str] = None):
        """
        Creates an MP4 you can scrub in any video player.

        Requires:
          pip install imageio imageio-ffmpeg
        """
        try:
            import imageio.v2 as imageio
        except Exception as e:
            raise RuntimeError("save_mp4 requires imageio (pip install imageio imageio-ffmpeg).") from e

        if tmp_frames_dir is None:
            tmp_frames_dir = "_tmp_mapping_frames"
        self.save_frames(tmp_frames_dir)

        frame_paths = [os.path.join(tmp_frames_dir, f"cycle_{c:05d}.png") for c in self.cycles]
        with imageio.get_writer(mp4_path, fps=fps) as w:
            for p in frame_paths:
                w.append_data(imageio.imread(p))

        # optional cleanup (comment out if you want to keep frames)
        for p in frame_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_frames_dir)
        except OSError:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crossbar mapping visualizer")
    parser.add_argument("json", help="Path to mapping JSON file")
    parser.add_argument("--mp4", help="Output MP4 file")
    parser.add_argument("--frames", help="Directory to write PNG frames")
    parser.add_argument("--fps", type=int, default=4, help="FPS for MP4 (default: 4)")

    args = parser.parse_args()

    viz = CrossbarVisualizer.from_json_path(args.json)

    if args.frames:
        viz.save_frames(args.frames)

    if args.mp4:
        viz.save_mp4(args.mp4, fps=args.fps)

    if not args.frames and not args.mp4:
        print("Nothing to do: specify --mp4 and/or --frames")

