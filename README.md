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