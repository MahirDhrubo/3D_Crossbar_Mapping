# Circuit Validation

This folder contains the Xyce/SPICE validation artifacts for the 3D VTEAM
crossbar circuits used by the mapping/evaluation code in the main repository.

It is intentionally separate from the Python synthesis and mapping package. The
mapping code consumes summarized operation costs; this folder contains the
device model, circuit netlists, sweep scripts, result CSVs, and figures used to
obtain and validate those costs.

## Contents

```text
Circuit-Validation/
  netlists/
    vteam_model.inc
    naive_3d_crossbar.cir
    routed_3d_crossbar.cir

  scripts/
    sweep_set_reset_floatpim_ron_roff.py
    sweep_nor_not_floatpim_ron_roff.py
    sweep_and_floatpim_ron_roff.py
    sweep_switch_nonideal_floatpim.py
    generate_spice_validation_assets.py
    plot_switch_nonideal_sensitivity.py

  results/
    logic_lowest_energy_best_points.csv
    logic_floatpim_sweep_points.csv
    series_and_row_parallel_validation.csv
    set_reset_floatpim_ron_roff_latency_energy_summary.csv
    floatpim_ron_roff_latency_energy_sweep_summary.csv
    and_floatpim_ron_roff_latency_energy_sweep_summary.csv
    switch_nonideal_floatpim_sensitivity_summary.csv

  figures/
    fig_series_and_truth_table_heatmap.pdf
    fig_series_and_transient_comparison.pdf
    fig_floatpim_valid_sweep_space.png
    fig_lowest_energy_latency.png
    switch_nonideal_floatpim_sensitivity.png
```

The detailed `*_valid.csv` files in `results/` keep the individual input-case
outputs used to decide whether each sweep point is valid.

## Requirements

- Xyce available as `Xyce` on `PATH`, or set `XYCE_BIN=/path/to/Xyce`
- Python 3.10 or newer
- Python packages listed in `requirements.txt`

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Circuit-Validation/requirements.txt
```

If Xyce is not on `PATH`, run commands as:

```bash
XYCE_BIN=/path/to/Xyce python Circuit-Validation/scripts/<script>.py
```

## Reproducing the Result CSVs

Run these commands from the repository root.

Standalone SET/RESET sweep:

```bash
python Circuit-Validation/scripts/sweep_set_reset_floatpim_ron_roff.py
```

Planar NOR/NOT sweep on the routed 3D crossbar:

```bash
python Circuit-Validation/scripts/sweep_nor_not_floatpim_ron_roff.py
```

Vertical row-parallel AND sweep on the routed 3D crossbar:

```bash
python Circuit-Validation/scripts/sweep_and_floatpim_ron_roff.py
```

Routing-switch on-resistance sensitivity:

```bash
python Circuit-Validation/scripts/sweep_switch_nonideal_floatpim.py
python Circuit-Validation/scripts/plot_switch_nonideal_sensitivity.py
```

These scripts write generated decks and transient outputs under
`Circuit-Validation/generated/`. Durable summaries are written under
`Circuit-Validation/results/`.

## Regenerating the Paper Figures

After the sweep CSVs are present, regenerate the validation figures with:

```bash
python Circuit-Validation/scripts/generate_spice_validation_assets.py
```

This script regenerates:

- `results/series_and_row_parallel_validation.csv`
- `results/logic_floatpim_sweep_points.csv`
- `results/logic_lowest_energy_best_points.csv`
- `figures/fig_series_and_truth_table_heatmap.pdf`
- `figures/fig_series_and_transient_comparison.pdf`
- `figures/fig_lowest_energy_latency.png`
- `figures/fig_floatpim_valid_sweep_space.png`
- `figures/fig_planar_nor_not_truth_outputs.png`
- `tables/tab_device_params.tex`

## Main Reported Cost File

The compact operation-cost summary is:

```text
results/logic_lowest_energy_best_points.csv
```

This file contains the lowest-energy and fastest valid points for NOR, NOT,
AND, SET, and RESET using the VTEAM parameterization and `RON=10k`,
`ROFF=10Meg`.

## Notes

- Logic is reported as `1 - state`, so LRS corresponds to logic 1 and HRS
  corresponds to logic 0.
- The routed 3D crossbar uses ideal bidirectional voltage-controlled switches
  for routing. The switch sensitivity script varies the switch on-resistance to
  show how non-ideal routing resistance affects correctness, latency, and
  energy.
- The provided validation focuses on vertical AND for the series-isolated mode
  and planar NOR/NOT for the shared-fabric mode.
