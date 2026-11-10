# Wind Farm Co-Design RL

EnvPG layout and yaw co-design on WFCRL's seven-turbine Ablaincourt case.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ./wfcrl-env torch tensorboard tyro stable-baselines3 python-dotenv seaborn notebook
```

Run [EnvPG](envpg.ipynb) or [IPPO/MAPPO](wfcrl_baselines.ipynb). The [comparison notebook](compare_models.ipynb) evaluates the saved EnvPG pilot and WFCRL baselines.

`envpg.py` implements the generalized critic. The saved EnvPG run improves layout reward in one seed; yaw control still trails no yaw.

Run `python visualize_environment.py` to refresh the [farm](design_doc/visuals/farm.mmd), [control step](design_doc/visuals/control_step.mmd), and [co-design](design_doc/visuals/codesign.mmd) diagrams. Paste a diagram into [ToDiagram](https://todiagram.com/editor). The command also draws the [layout](design_doc/visuals/layout.png) and [wake](design_doc/visuals/wake.png); use `--run`, `--wind-direction`, or `--wind-speed` to change the inputs.
