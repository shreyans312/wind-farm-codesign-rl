# Wind-MARL

EnvPG layout and yaw co-design for WFCRL's seven-turbine Ablaincourt FLORIS case.

```bash
.venv/bin/python envpg.py --site XMIN XMAX YMIN YMAX --max-displacement METRES --min-spacing METRES --run-dir runs/name
.venv/bin/tensorboard --logdir runs
```

Choose site and spacing limits before training. Runs save a checkpoint, configuration, and TensorBoard logs.

The [WFCRL baseline notebook](wfcrl_baselines.ipynb) runs the authors' IPPO and MAPPO code.
