# Wind Farm Co-Design RL

EnvPG layout and yaw co-design on WFCRL's seven-turbine Ablaincourt case.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ./wfcrl-env torch tensorboard tyro stable-baselines3 python-dotenv seaborn notebook
```

Run [EnvPG](envpg.ipynb) or [IPPO/MAPPO](wfcrl_baselines.ipynb). The [comparison notebook](compare_models.ipynb) evaluates the saved EnvPG pilot and WFCRL baselines.

`envpg.py` implements the generalized critic. The saved EnvPG run improves layout reward in one seed; yaw control still trails no yaw.
