# Wind Farm Co-Design RL

EnvPG layout and yaw co-design on WFCRL's seven-turbine Ablaincourt case.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ./wfcrl-env torch tensorboard tyro stable-baselines3 python-dotenv seaborn notebook
```

Run [EnvPG](notebooks/envpg.ipynb) or [IPPO/MAPPO](notebooks/wfcrl_baselines.ipynb). Set site and spacing limits in the EnvPG notebook.

WFCRL [environment](https://github.com/ifpen/wfcrl-env/tree/6634d468c12d00d8d6b0950684bcd67a55f02551) and [benchmark](https://github.com/ifpen/wfcrl-benchmark/tree/21ed167cdfa3bca60cad28b01bff008654e16983) sources are included under Apache 2.0.
