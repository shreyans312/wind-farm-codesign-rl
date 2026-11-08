# Wind Farm Co-Design RL

EnvPG layout and yaw co-design for WFCRL's seven-turbine Ablaincourt FLORIS case.

```bash
.venv/bin/python envpg.py --site XMIN XMAX YMIN YMAX --max-displacement METRES --min-spacing METRES --run-dir runs/name
.venv/bin/tensorboard --logdir runs
```

Choose site and spacing limits before training. Runs save a checkpoint, configuration, and TensorBoard logs.

The [WFCRL baseline notebook](wfcrl_baselines.ipynb) runs the authors' IPPO and MAPPO code.

WFCRL [environment](https://github.com/ifpen/wfcrl-env/tree/6634d468c12d00d8d6b0950684bcd67a55f02551) and [benchmark](https://github.com/ifpen/wfcrl-benchmark/tree/21ed167cdfa3bca60cad28b01bff008654e16983) sources are included under their Apache 2.0 licenses, with local compatibility fixes.
