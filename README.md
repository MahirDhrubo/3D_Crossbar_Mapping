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

## `Circuit-Validation/`

The Xyce/SPICE circuit-validation artifact is provided in `Circuit-Validation/`.
It contains the VTEAM model, naive and routed 3D crossbar netlists, sweep
scripts, generated latency/energy CSVs, and validation figures used for the
circuit-level results. See `Circuit-Validation/README.md` for the commands to
rerun the simulations or regenerate the figures.

## Scripts

The core pipeline is: map a BLIF netlist to a mapping-result JSON (`main.py` for 3D, `map_2d.py` for 2D), then optionally compute energy/latency from that JSON (`energy_latency_calculator.py`). The remaining scripts are standalone tools.

### `mem3dmapper.cli.main` — 3D mapping (PRISM's concurrent scheduler)

```bash
python -m mem3dmapper.cli.main <input.blif> <output.json> [options]
```

| Flag | Default | Description |
|---|---|---|
| `--rows` | 6 | Crossbar rows / stacked layers |
| `--columns` | 512 | Crossbar columns |
| `--cluster-window` | 21 | Cost-model cluster window |
| `--alpha` | 0.6 | Cost-model weight alpha |
| `--beta` | 0.4 | Cost-model weight beta |
| `--gamma` | 0.0 | Cost-model weight gamma |
| `--skip-validation` | off | Skip validating the mapping against the netlist |
| `--dag-dot PATH` | none | Write the netlist's DAG as a `.dot` file |
| `--dag-image PATH` | none | Render the DAG image (requires `--dag-dot`; format inferred from the file extension, e.g. `.png`/`.svg`) |

### `mem3dmapper.cli.map_2d` — 2D mapping (sequential NOR-NOT-only heuristic)

```bash
python -m mem3dmapper.cli.map_2d <input.blif> <output.json> [options]
```

| Flag | Default | Description |
|---|---|---|
| `--columns` | 512 | Crossbar row width |

the mapper starts at one crossbar and appends more as needed, so the crossbar count is part of the output, not an input.

### `mem3dmapper.cli.energy_latency_calculator` — energy/latency from a mapping JSON

```bash
python -m mem3dmapper.cli.energy_latency_calculator <mapping.json> [options]
```

| Flag | Default | Description |
|---|---|---|
| `--cycle-time-ns` | 0.1 | Per-cycle time in ns |

### `mem3dmapper.cli.count_gates` — gate-count comparison between two BLIF directories

```bash
python -m mem3dmapper.cli.count_gates <dir1> <dir2> [options]
```

Parses every matching file in each directory (paired by sort order) and prints the fractional gate-count reduction of `dir1` relative to `dir2` for each pair.

| Flag | Default | Description |
|---|---|---|
| `--ext` | `.blif` | File extension to look for |

### `mem3dmapper.cli.multiplier32_2d_baseline_comparison` — Table I cost model

```bash
python -m mem3dmapper.cli.multiplier32_2d_baseline_comparison [options]
```

Estimates energy, latency, and crossbar count for a 32-bit multiplier on SIMPLER/LOGIC/AUTO (2D) vs. PRISM (3D), from real device-level gate costs.

| Flag | Default | Description |
|---|---|---|
| `--gate-costs-csv` | `data/logic_lowest_energy_best_points.csv` | Device-characterization CSV (see `load_gate_costs`) |
| `--output-csv` | `data/runs/multiplier32_2d_baseline_energy_latency.csv` | Where to write the full per-framework breakdown |
| `--row-cap` | 512 | Cells per crossbar/row (informational only) |
| `--latency-bound` | 0.1 | Flat per-operation latency bound in ns |

### `mem3dmapper.cli.visualize` — optional mapping-trace visualizer

```bash
python -m mem3dmapper.cli.visualize <mapping.json> [--frames DIR] [--mp4 PATH] [--fps N]
```

| Flag | Default | Description |
|---|---|---|
| `--frames DIR` | none | Write one PNG frame per cycle to this directory |
| `--mp4 PATH` | none | Write an MP4 video of the mapping trace |
| `--fps` | 4 | Frames per second for the MP4 |

Requires `imageio`/`imageio-ffmpeg` for `--mp4` and `ipywidgets`/`ipython` for the notebook widget mode (see Requirements). At least one of `--frames`/`--mp4` is required.

## Reproducing Table I (32-bit Fixed-Point Multiplication)

```bash
# Gates + the 26.1% improvement figure (parsing only, no mapping)
python -m mem3dmapper.cli.count_gates public-data/table1/gate_comparison_netlist public-data/table1/gate_comparison_netlist_nor

# Crossbars, Latency, Energy
python -m mem3dmapper.cli.multiplier32_2d_baseline_comparison --gate-costs-csv public-data/table1/logic_lowest_energy_best_points.csv
```

## Reproducing Table II (2D vs. 3D Mapping Overhead, 512-Column Crossbars)

```bash
python -m mem3dmapper.cli.main   public-data/table2/3d/2bit_adder.blif      data/runs/table2/3d/2bit_adder.json      --rows 3 --columns 512
python -m mem3dmapper.cli.main   public-data/table2/3d/4bit_adder.blif      data/runs/table2/3d/4bit_adder.json      --rows 3 --columns 512
python -m mem3dmapper.cli.main   public-data/table2/3d/dot4_4bit.blif       data/runs/table2/3d/dot4_4bit.json       --rows 3 --columns 512
python -m mem3dmapper.cli.main   public-data/table2/3d/matvec_8bit_pp.blif  data/runs/table2/3d/matvec_8bit_pp.json  --rows 3 --columns 512
python -m mem3dmapper.cli.main   public-data/table2/3d/matvec4x4_8bit.blif  data/runs/table2/3d/matvec4x4_8bit.json  --rows 6 --columns 512

python -m mem3dmapper.cli.map_2d public-data/table2/2d/2bit_adder_nor.blif      data/runs/table2/2d/2bit_adder_nor.json      --columns 512
python -m mem3dmapper.cli.map_2d public-data/table2/2d/4bit_adder_nor.blif      data/runs/table2/2d/4bit_adder_nor.json      --columns 512
python -m mem3dmapper.cli.map_2d public-data/table2/2d/dot4_4bit_nor.blif       data/runs/table2/2d/dot4_4bit_nor.json       --columns 512
python -m mem3dmapper.cli.map_2d public-data/table2/2d/matvec_8bit_pp_nor.blif  data/runs/table2/2d/matvec_8bit_pp_nor.json  --columns 512
python -m mem3dmapper.cli.map_2d public-data/table2/2d/matvec4x4_8bit_nor.blif  data/runs/table2/2d/matvec4x4_8bit_nor.json  --columns 512
```