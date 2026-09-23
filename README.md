# mem3dmapper

## Requirements

- Python >= 3.10
- Install the package (editable) and its dependencies:

  ```bash
  pip install -e .
  pip install -r requirements.txt
  ```

- Optional system dependency: [Graphviz](https://graphviz.org/) (the `dot` binary on your `PATH`) is required only for DAG visualization (`mem3dmapper.dag.visualize.render_graphviz`).
- `imageio`, `imageio-ffmpeg`, and `ipywidgets`/`ipython` are only needed for the optional interactive/video mapping visualizer (`mem3dmapper.cli.visualize`); the core synthesis/mapping pipeline does not require them.

## `public-data/`

A small set of BLIF netlists (plus one device-characterization CSV), sufficient to reproduce the paper's Table I and Table II numbers without the full local `data/` working directory.

- `public-data/table1/` — inputs for Table I (32-bit fixed-point multiplication comparison)
- `public-data/table2/2d/`, `public-data/table2/3d/` — inputs for Table II (2D vs. 3D mapping overhead), one BLIF per component per side

## Pipeline: BLIF -> mapping JSON -> energy/latency

1. Map a BLIF netlist onto a crossbar and write the execution trace to a JSON file:

   ```bash
   python -m mem3dmapper.cli.main <input.blif> <output.json> --rows <R> --columns <C>      # 3D, PRISM's concurrent scheduler
   python -m mem3dmapper.cli.map_2d <input.blif> <output.json> --rows <R> --columns <C>    # 2D, sequential NOR-NOT-only heuristic
   ```

2. Compute total energy and latency from that JSON:

   ```bash
   python -m mem3dmapper.cli.energy_latency_calculator <output.json>
   ```

## Reproducing Table I (32-bit Fixed-Point Multiplication)

```bash
# Gates + the 26.1% improvement figure (parsing only, no mapping)
python -m mem3dmapper.cli.count_gates public-data/table1/gate_comparison_netlist public-data/table1/gate_comparison_netlist_nor

# Crossbars, Latency, Energy
python -m mem3dmapper.cli.multiplier32_2d_baseline_comparison --gate-costs-csv public-data/table1/logic_lowest_energy_best_points.csv
```

## Reproducing Table II (2D vs. 3D Mapping Overhead, 512-Column Crossbars)

```bash
python -m mem3dmapper.cli.main   public-data/table2/3d/<file>.blif <out.json> --rows <R> --columns 512   # 3D: Gates, Moves(=0 by construction)
python -m mem3dmapper.cli.map_2d public-data/table2/2d/<file>.blif <out.json> --rows <R> --columns 512   # 2D: Gates, Moves
```

Rows (Layers) per component, all at 512 columns: 2-bit Full Adder / 4-bit Ripple-Carry Adder / 4x4 Dot Prod. / MatVec Partial Prod. = 3; 4x4 MatVec = 6. `Gates = NOR + NOT + AND` and `Moves = COPY` from the output JSON's `operations_count`.

`map_2d.py` does not run `mem3dmapper.mapping.validator` — its geometry and concurrent-scheduling checks are written for the 3D crossbar model and don't apply to a 2D mapping.

## Known Issues

- `mem3dmapper.mapping.parallel_mapping` (the 3D concurrent scheduler) previously had a validator false-positive (`overwrites live net ... with N use(s) remaining`) that misfired whenever a live net's copy was overwritten by an operation that also read that same net elsewhere in the same cycle, even when another copy of it survived. Fixed in `mapping/validator.py`; confirmed against the previously-failing `4bit_adder.blif` at `--rows 3` (now succeeds, 41 gates matching Table II) and the existing unit tests.
