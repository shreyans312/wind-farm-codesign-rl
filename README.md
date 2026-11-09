# Wind Farm Co-Design RL

EnvPG layout and yaw co-design on WFCRL's seven-turbine Ablaincourt case.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ./wfcrl-env torch tensorboard tyro stable-baselines3 python-dotenv seaborn notebook
```

Run [EnvPG](envpg.ipynb) or [IPPO/MAPPO](wfcrl_baselines.ipynb), then [compare checkpoints](compare_models.ipynb). Set site and spacing limits in the EnvPG notebook.
