"""Generate ToDiagram-ready Mermaid views and a spatial layout plot."""

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "__simul__" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata

from envpg import BASIS_NAMES, Config, N, REFERENCE, feasible, layout, state_vector, windrose
from wfcrl import environments as envs


def read_run(path):
    saved = json.loads((path / "config.json").read_text())
    actual_hash = hashlib.sha256((ROOT / "envpg.py").read_bytes()).hexdigest()
    if saved.get("source_sha256") != actual_hash:
        raise ValueError("The saved run uses a different envpg.py; check out its source revision first")
    cfg = Config(**saved["parameters"])
    checkpoint_path = path / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = path / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    z = np.asarray(checkpoint["layout"], dtype=float)
    if not feasible(z, cfg):
        raise ValueError("The saved layout violates the run's design constraints")
    return cfg, z


def inspect_spaces(cfg, z):
    central = envs.make("Ablaincourt_Floris", log=False,
                        max_num_steps=cfg.horizon + 1, load_coef=cfg.load_coef)
    local = envs.make("Dec_Ablaincourt_Floris", log=False,
                      max_num_steps=cfg.horizon + 1, load_coef=cfg.load_coef)
    try:
        p = layout(z)
        central.mdp.interface.fi.reinitialize(layout_x=p[:, 0], layout_y=p[:, 1])
        central.reset(options={"wind_speed": 8.0, "wind_direction": 270.0})
        state_dim = len(state_vector(central))
        central_obs = sum(int(np.prod(s.shape)) for s in central.observation_space.spaces.values())
        first = local.possible_agents[0]
        local_obs = sum(int(np.prod(s.shape)) for s in local.observation_space(first).values())
        action = central.action_space["yaw"]
        assert action.shape == (N,)
        return state_dim, central_obs, local_obs, float(action.low[0]), float(action.high[0])
    finally:
        central.close()
        local.close()


def diagrams(cfg, z, spaces, bins):
    state_dim, central_obs, local_obs, action_low, action_high = spaces
    xmin, xmax, ymin, ymax = cfg.site
    design_nodes = "\n".join(
        f'  z{i}["{name.replace("_", " ")}: {value:+.2f} m"] --> geometry'
        for i, (name, value) in enumerate(zip(BASIS_NAMES, z), 1)
    )
    farm = f'''flowchart LR
  ref["Reference positions: {N} turbines"] --> geometry["Layout p(z)"]
{design_nodes}
  site["Site x {xmin:g}..{xmax:g} m; y {ymin:g}..{ymax:g} m"] -.-> geometry
  limits["Max shift {cfg.max_displacement:g} m; spacing at least {cfg.min_spacing:g} m"] -.-> geometry
  geometry --> wake["FLORIS steady wake and turbine power"]
  wind["Freestream wind speed and direction"] --> wake
  yaw["Current yaw angles: {N}"] --> wake
  wake --> power["Turbine power: {N} values"]
  wake --> loads["Turbine load proxies"]
  power --> reward["Shared reward = mean(1000 P / u³) - {cfg.load_coef:g} mean(abs(load))"]
  loads --> reward
'''
    control = f'''flowchart TB
  raw["WFCRL observation: {central_obs} values; freewind 2, yaw {N}, local speed {N}, local direction {N}"] --> envpg_state["EnvPG state: {state_dim} values; freewind 2, yaw {N}, yaw use {N}, time 1"]
  duty["Internal state: yaw use and step count"] --> envpg_state
  raw --> local["WFCRL agent observation: {local_obs} values; own yaw, local speed, local direction"]
  design["Fixed episode layout z"] --> actor["Central EnvPG actor"]
  envpg_state --> actor
  local --> agents["{N} separate WFCRL agents"]
  actor --> proposal["Joint yaw increments: {N} values, {action_low:g}..{action_high:g} degrees"]
  agents --> proposal
  proposal --> guard["Duty guard; executed yaw increment may be zero"]
  guard --> target["Yaw target clipped to -40..40 degrees"]
  target --> simulator["FLORIS step: 60 s"]
  wind["Freestream wind"] --> simulator
  design --> simulator
  simulator --> next["Next yaw and local wind"]
  simulator --> metrics["Power and load proxies"]
  next --> raw
  next --> duty
  metrics --> reward["Shared WFCRL reward"]
'''
    learning = f'''flowchart TB
  config["{cfg.iterations} iterations; {cfg.episodes} episodes each; {cfg.warmup_iterations} layout warmup iterations"] --> rollout
  design["Current design z; fixed for {cfg.horizon} control steps"] --> rollout["Actor plus FLORIS rollout"]
  actor["Central yaw actor"] --> rollout
  train_wind["Training wind: sample {bins} bins by frequency"] --> rollout
  rollout --> samples["State, action, cost trajectories"]
  design --> perturb["Nearby feasible design z-prime; sigma {cfg.perturb_sigma:g} m"]
  perturb --> trial["Same-state, same-action paired FLORIS step"]
  samples --> critic["Generalized critic Q(s,a,z-prime;z)"]
  trial --> critic
  samples --> returns["Auxiliary finite-horizon return fit"]
  returns --> critic
  updates["{cfg.critic_updates} critic updates per iteration; gamma {cfg.gamma:g}"] --> critic
  critic --> actor_update["Policy-gradient actor update"]
  critic --> design_update["EnvPG design gradient through Q and policy score"]
  actor_update --> actor
  design_update --> design
  eval["Common evaluation: {bins} wind bins, {cfg.horizon} steps each"] --> split["Report layout gain and yaw gain separately"]
  actor --> eval
  design --> eval
'''
    return {"farm.mmd": farm, "control_step.mmd": control, "codesign.mmd": learning}


def plot_layout(cfg, z, path):
    p = layout(z)
    delta = p - REFERENCE
    xmin, xmax, ymin, ymax = cfg.site
    pairwise = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
    pairwise[np.diag_indices(N)] = np.inf

    fig, (farm, shift) = plt.subplots(1, 2, figsize=(12, 4),
                                     gridspec_kw={"width_ratios": [1.55, 1]})
    farm.plot([xmin, xmax, xmax, xmin, xmin],
              [ymin, ymin, ymax, ymax, ymin], color="0.65", lw=1)
    farm.scatter(REFERENCE[:, 0], REFERENCE[:, 1], marker="x", color="0.5",
                 label="Reference", zorder=3)
    farm.scatter(p[:, 0], p[:, 1], facecolors="none", edgecolors="#1768ac",
                 s=65, label="Saved design", zorder=4)
    for i, (x, y) in enumerate(p, 1):
        farm.text(x, y + 26, str(i), ha="center", fontsize=8)
    farm.set(xlabel="x (m)", ylabel="y (m)", title="Farm layout")
    farm.set_xlim(xmin - 70, xmax + 70)
    farm.set_ylim(ymin - 70, ymax + 70)
    farm.set_aspect("equal", adjustable="box")
    farm.grid(alpha=.15)
    farm.legend(loc="lower left", fontsize=8)

    y = np.arange(N)
    shift.barh(y - .16, delta[:, 0], height=.3, label="Delta x")
    shift.barh(y + .16, delta[:, 1], height=.3, label="Delta y")
    shift.set_yticks(y, [str(i) for i in range(1, N + 1)])
    shift.invert_yaxis()
    shift.axvline(0, color="0.3", lw=.7)
    shift.set(xlabel="Displacement (m)", ylabel="Turbine", title="Changes from reference")
    shift.grid(axis="x", alpha=.2)
    shift.legend(fontsize=8)
    fig.suptitle(f"Maximum shift {np.linalg.norm(delta, axis=1).max():.2f} m; "
                 f"minimum spacing {pairwise.min():.1f} m", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_wake(cfg, z, direction, speed, path):
    if not (0 <= direction < 360 and speed > 0):
        raise ValueError("Wind direction must be in [0, 360), and speed must be positive")
    env = envs.make("Ablaincourt_Floris", log=False,
                    max_num_steps=cfg.horizon + 1, load_coef=cfg.load_coef)
    fi = env.mdp.interface.fi
    hub_height = float(np.asarray(fi.floris.farm.hub_heights).flat[0])
    xmin, xmax, ymin, ymax = cfg.site
    bounds = dict(x_bounds=(xmin - 100, xmax + 300),
                  y_bounds=(ymin - 150, ymax + 150))
    planes = []
    try:
        for positions in (REFERENCE, layout(z)):
            fi.reinitialize(layout_x=positions[:, 0], layout_y=positions[:, 1])
            yaw = np.zeros_like(fi.floris.farm.yaw_angles)
            plane = fi.calculate_horizontal_plane(
                height=hub_height, x_resolution=100, y_resolution=60,
                wd=[direction], ws=[speed], yaw_angles=yaw, **bounds)
            planes.append(plane.df[["x1", "x2", "u"]])
    finally:
        env.close()

    x = np.linspace(*bounds["x_bounds"], 100)
    y = np.linspace(*bounds["y_bounds"], 60)
    xx, yy = np.meshgrid(x, y)
    reference, learned = (
        griddata(plane[["x1", "x2"]], plane["u"], (xx, yy), method="linear")
        for plane in planes
    )
    difference = learned - reference
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.3), sharex=True, sharey=True)
    levels = np.linspace(0, speed, 21)
    for ax, data, positions, title in zip(
            axes[:2], (reference, learned), (REFERENCE, layout(z)),
            ("Reference layout", "Saved design")):
        image = ax.contourf(x, y, data, levels=levels,
                            cmap="viridis", extend="both")
        ax.scatter(positions[:, 0], positions[:, 1], s=13, c="white",
                   edgecolors="black", linewidths=.4)
        ax.set_title(title)
    bound = max(float(np.nanmax(np.abs(difference))), 1e-3)
    diff_image = axes[2].contourf(x, y, difference,
                                  levels=np.linspace(-bound, bound, 21),
                                  cmap="RdBu_r", extend="both")
    axes[2].set_title("Saved minus reference")
    for ax in axes:
        ax.set(xlabel="x (m)", aspect="equal")
    axes[0].set_ylabel("y (m)")
    fig.colorbar(image, ax=axes[:2], label="FLORIS u (m/s)", shrink=.7)
    fig.colorbar(diff_image, ax=axes[2], label="Delta u (m/s)", shrink=.7)
    fig.suptitle(f"{direction:g} degree wind at {speed:g} m/s; "
                 f"zero yaw; {hub_height:g} m height")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=ROOT / "runs" / "envpg-mc-codesign-pilot-seed1")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "design_doc" / "visuals")
    parser.add_argument("--wind-direction", type=float, default=270.0)
    parser.add_argument("--wind-speed", type=float, default=8.0)
    args = parser.parse_args()
    cfg, z = read_run(args.run)
    spaces = inspect_spaces(cfg, z)
    wind_data = Path(cfg.wind_data)
    if not wind_data.exists():
        wind_data = ROOT / "wfcrl-benchmark" / "data" / wind_data.name
    bins = len(windrose(wind_data))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, source in diagrams(cfg, z, spaces, bins).items():
        (args.output_dir / name).write_text(source)
    plot_layout(cfg, z, args.output_dir / "layout.png")
    plot_wake(cfg, z, args.wind_direction, args.wind_speed,
              args.output_dir / "wake.png")
    print(f"Wrote three Mermaid diagrams, layout.png, and wake.png to {args.output_dir}")


if __name__ == "__main__":
    main()
