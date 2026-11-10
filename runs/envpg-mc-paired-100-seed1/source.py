# EnvPG for the seven-turbine Ablaincourt FLORIS case. Cost is minus the unchanged WFCRL reward

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "wfcrl-env"))
from wfcrl import environments as envs  # noqa: E402
from wfcrl.environments.data_cases import floris_ablaincourt  # noqa: E402

N = 7
REFERENCE = np.stack((floris_ablaincourt.xcoords, floris_ablaincourt.ycoords), axis=1)


@dataclass
class Config:
    site: tuple[float, float, float, float]
    max_displacement: float
    min_spacing: float
    run_dir: str
    seed: int = 1
    iterations: int = 100
    episodes: int = 4
    horizon: int = 150
    gamma: float = 0.99
    load_coef: float = 0.1
    alpha: float = 0.1
    alpha_decay: float = 0.99
    perturb_sigma: float = 10.0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    design_lr: float = 100.0
    critic_updates: int = 20
    sensitivity_updates: int = 100
    batch_size: int = 128
    warmup_iterations: int = 5
    eval_every: int = 10
    wind_data: str = str(ROOT / "wfcrl-benchmark" / "data" / "smarteole.csv")
    finite_difference_m: float = 1.0
    min_gradient_cosine: float = 0.5
    max_gradient_relative_error: float = 1.0
    policy_gradient: str = "critic"
    policy_baseline: str = "leave_one_out"
    training_wind: str = "environment"
    zero_init_actor: bool = False
    design_gradient: str = "learned"
    design_probes_per_episode: int = 2
    design_step_cap_m: float = 10.0


def layout(z):
    positions = REFERENCE.copy()
    positions[1:] += np.asarray(z).reshape(6, 2)
    return positions


def feasible(z, cfg):
    p = layout(z)
    xmin, xmax, ymin, ymax = cfg.site
    if not (xmin <= p[:, 0].min() and p[:, 0].max() <= xmax
            and ymin <= p[:, 1].min() and p[:, 1].max() <= ymax):
        return False
    if np.any(np.linalg.norm(p[1:] - REFERENCE[1:], axis=1) > cfg.max_displacement):
        return False
    distances = np.linalg.norm(p[:, None] - p[None, :], axis=2)
    return bool(np.all(distances[np.triu_indices(N, 1)] >= cfg.min_spacing))


def set_layout(env, z):
    p = layout(z)
    env.mdp.interface.fi.reinitialize(layout_x=p[:, 0], layout_y=p[:, 1])


def make_env(cfg):
    # Reset advances FLORIS once, hence the extra simulator step.
    return envs.make("Ablaincourt_Floris", log=False,
                     max_num_steps=cfg.horizon + 1, load_coef=cfg.load_coef)


def state_vector(env):
    obs = env._state
    wind = np.asarray(obs["freewind_measurements"], dtype=np.float32)
    yaw = np.asarray(obs["yaw"], dtype=np.float32)
    usage = np.asarray(env.accumulated_actions["yaw"], dtype=np.float32)
    return np.r_[wind, yaw, usage, float(env.num_moves)].astype(np.float32)


def normalize_state(s, horizon):
    scale = torch.tensor([28, 360] + [40] * N + [5 * horizon] * N + [horizon],
                         device=s.device, dtype=s.dtype)
    return s / scale


def action_mask(s):
    # Match WindFarmEnv's retrospective 10% duty guard, before step increments time.
    t = s[..., -1:]
    usage = s[..., 2 + N:2 + 2 * N]
    return usage / 0.3 / ((t + 1) * 60) < 0.1


def network(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(),
                         nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, output_dim))


class Actor(nn.Module):
    def __init__(self, horizon, max_displacement, zero_mean_init=False):
        super().__init__()
        self.horizon = horizon
        self.max_displacement = max_displacement
        self.mean = network(17 + 12, N)
        self.log_std = nn.Parameter(torch.full((N,), -0.7))
        if zero_mean_init:
            nn.init.zeros_(self.mean[-1].weight)
            nn.init.zeros_(self.mean[-1].bias)

    def distribution(self, s, z):
        x = torch.cat((normalize_state(s, self.horizon), z / self.max_displacement), -1)
        return Normal(self.mean(x), self.log_std.clamp(-5, 2).exp())

    def sample(self, s, z, deterministic=False):
        d = self.distribution(s, z)
        raw = d.mean if deterministic else d.rsample()
        a = 5 * torch.tanh(raw)
        return a * action_mask(s)

    def log_prob(self, s, z, a):
        d = self.distribution(s, z)
        active = action_mask(s)
        unit = (a / 5).clamp(-1 + 1e-6, 1 - 1e-6)
        raw = torch.atanh(unit)
        logp = d.log_prob(raw) - math.log(5) - torch.log1p(-unit.square())
        return (logp * active).sum(-1)


class Critic(nn.Module):
    def __init__(self, horizon, max_displacement):
        super().__init__()
        self.horizon = horizon
        self.max_displacement = max_displacement
        self.q = network(17 + N + 12, 1)

    def forward(self, s, a, z):
        x = torch.cat((normalize_state(s, self.horizon), a / 5,
                       z / self.max_displacement), -1)
        return self.q(x).squeeze(-1)


class CostSensitivity(nn.Module):
    # Local slope of the current cost; future canonical state is layout independent

    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
        self.g = network(17 + N, 12)

    def forward(self, s, a):
        return self.g(torch.cat((normalize_state(s, self.horizon), a / 5), -1)) * 0.01


def tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def discounted_returns(costs, dones, gamma):
    returns = torch.empty_like(costs)
    future = 0.0
    for i in range(len(costs) - 1, -1, -1):
        future = costs[i] + gamma * (1 - dones[i]) * future
        returns[i] = future
    return returns


def leave_one_out_baseline(returns, times):
    baseline = torch.zeros_like(returns)
    for t in torch.unique(times):
        indices = torch.where(times == t)[0]
        if len(indices) > 1:
            baseline[indices] = (returns[indices].sum() - returns[indices]) / (len(indices) - 1)
    return baseline


def trial_step(trial, base, z_trial, action):
    # FLORIS is steady state: its next wake depends on current wind, layout and yaw
    set_layout(trial, z_trial)
    trial._state = copy.deepcopy(base._state)
    trial.num_moves = base.num_moves
    trial.mdp.interface._num_iter = base.mdp.interface._num_iter
    trial.accumulated_actions = {k: v.copy() for k, v in base.accumulated_actions.items()}
    trial.mdp._actuation_accumulator = {
        k: v.copy() for k, v in base.mdp._actuation_accumulator.items()
    }
    _, reward, _, done, info = trial.step({"yaw": action.copy()})
    return state_vector(trial), -float(reward.item()), done, info


def perturb(z, cfg, rng):
    for _ in range(1000):
        candidate = z + rng.normal(0, cfg.perturb_sigma, 12)
        if feasible(candidate, cfg):
            return candidate
    raise RuntimeError("No feasible perturbation found; reduce --perturb-sigma")


def design_step(z, gradient, cfg):
    step = cfg.design_lr
    if cfg.design_gradient == "paired":
        max_move = max(np.linalg.norm(gradient.reshape(6, 2), axis=1)) * step
        if max_move > cfg.design_step_cap_m:
            step *= cfg.design_step_cap_m / max_move
    for _ in range(20):
        candidate = z - step * gradient
        if feasible(candidate, cfg):
            return candidate, step
        step *= 0.5
    return z, 0.0


def windrose(path):
    data = pd.read_csv(path)
    direction = (data["wd"].to_numpy() % 360 + 60) % 360
    speed = data["ws"].to_numpy()
    counts, direction_edges, speed_edges = np.histogram2d(direction, speed, bins=5)
    weights = counts / counts.sum()
    return [((direction_edges[i] + direction_edges[i + 1]) / 2,
             (speed_edges[j] + speed_edges[j + 1]) / 2, weights[i, j])
            for i in range(5) for j in range(5)]


def evaluate(env, actor, z, cfg, conditions):
    set_layout(env, z)
    score, energy_mwh, load_proxy, duty, calls = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for direction, speed, weight in conditions:
            env.reset(options={"wind_speed": float(speed),
                               "wind_direction": float(direction)})
            calls += 1
            for _ in range(cfg.horizon):
                if actor is None:
                    a = np.zeros(N, dtype=np.float32)
                else:
                    s = tensor(state_vector(env)).unsqueeze(0)
                    a = actor.sample(s, tensor(z).unsqueeze(0), deterministic=True)[0].numpy()
                _, reward, _, done, info = env.step({"yaw": a.copy()})
                score += weight * float(reward.item())
                energy_mwh += weight * float(np.sum(info["power"])) / 60
                load_proxy += weight * float(np.mean(np.abs(info["load"]))) / cfg.horizon
                calls += 1
                if done:
                    break
            duty += weight * float(np.mean(env.accumulated_actions["yaw"] /
                                           (0.3 * env.num_moves * 60)))
    return score, energy_mwh, load_proxy, duty, calls


def one_step_gradient(trial, base, z, s, a, actor, alpha, cfg):
    # Paired difference of current cost plus entropy at a fixed state/action
    result = np.full(12, np.nan)
    calls = 0
    for i in range(12):
        plus, minus = z.copy(), z.copy()
        plus[i] += cfg.finite_difference_m
        minus[i] -= cfg.finite_difference_m
        if not (feasible(plus, cfg) and feasible(minus, cfg)):
            continue
        values = []
        for candidate in (plus, minus):
            _, cost, _, _ = trial_step(trial, base, candidate, a)
            with torch.no_grad():
                logp = actor.log_prob(tensor(s).unsqueeze(0),
                                       tensor(candidate).unsqueeze(0),
                                       tensor(a).unsqueeze(0)).item()
            values.append(cost + alpha * logp)
            calls += 1
        result[i] = (values[0] - values[1]) / (2 * cfg.finite_difference_m)
    return result, calls


def train(cfg):
    if not (cfg.site[0] < cfg.site[1] and cfg.site[2] < cfg.site[3]):
        raise ValueError("Site bounds must be xmin xmax ymin ymax")
    if min(cfg.max_displacement, cfg.min_spacing, cfg.perturb_sigma,
           cfg.finite_difference_m) <= 0:
        raise ValueError("Layout limits and perturbation scale must be positive")
    if not feasible(np.zeros(12), cfg):
        raise ValueError("The reference layout violates the supplied constraints")
    if cfg.horizon < 1 or cfg.episodes < 1 or cfg.iterations < 1:
        raise ValueError("Horizon, episodes and iterations must be positive")
    if min(cfg.critic_updates, cfg.sensitivity_updates, cfg.batch_size) < 1:
        raise ValueError("Update counts and batch size must be positive")
    if not (0 <= cfg.gamma <= 1 and cfg.alpha >= 0 and 0 < cfg.alpha_decay <= 1):
        raise ValueError("Invalid discount or entropy settings")
    if cfg.warmup_iterations < 0 or cfg.eval_every < 0:
        raise ValueError("Warm-up and evaluation interval must be nonnegative")
    if cfg.policy_gradient not in {"critic", "mc"}:
        raise ValueError("policy_gradient must be critic or mc")
    if cfg.policy_baseline not in {"leave_one_out", "no_yaw_residual"}:
        raise ValueError("policy_baseline must be leave_one_out or no_yaw_residual")
    if cfg.policy_baseline == "no_yaw_residual" and cfg.policy_gradient != "mc":
        raise ValueError("no_yaw_residual requires mc policy gradient")
    if cfg.training_wind not in {"environment", "windrose"}:
        raise ValueError("training_wind must be environment or windrose")
    if cfg.design_gradient not in {"learned", "paired"}:
        raise ValueError("design_gradient must be learned or paired")
    if cfg.design_probes_per_episode < 1:
        raise ValueError("design_probes_per_episode must be positive")
    if cfg.design_step_cap_m <= 0:
        raise ValueError("design_step_cap_m must be positive")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    metadata = {"parameters": asdict(cfg), "revision": revision,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "packages": {name: version(name) for name in
                             ("torch", "tensorboard", "numpy", "floris", "gymnasium")}}
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2))
    writer = SummaryWriter(run_dir / "tb")
    base, trial = make_env(cfg), make_env(cfg)
    baseline_env = make_env(cfg) if cfg.policy_baseline == "no_yaw_residual" else None
    actor = Actor(cfg.horizon, cfg.max_displacement, cfg.zero_init_actor)
    critic = Critic(cfg.horizon, cfg.max_displacement)
    sensitivity = CostSensitivity(cfg.horizon)
    target = copy.deepcopy(critic)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    sensitivity_opt = torch.optim.Adam(sensitivity.parameters(), lr=cfg.critic_lr)
    z = np.zeros(12, dtype=np.float64)
    calls = 0
    conditions = windrose(cfg.wind_data) if cfg.eval_every > 0 else []
    training_conditions = windrose(cfg.wind_data) if cfg.training_wind == "windrose" else []
    if training_conditions:
        wind_weights = np.array([condition[2] for condition in training_conditions])
    reference_no_yaw = None
    if conditions:
        reference_no_yaw, _, _, _, eval_calls = evaluate(
            base, None, np.zeros(12), cfg, conditions)
        calls += eval_calls
    best_score = -np.inf
    try:
        for iteration in range(cfg.iterations):
            # The generalized critic's implicit future-layout copy changes each iteration.
            transitions, base_samples, no_yaw_returns = [], [], []
            baseline_cache = {}
            rewards, powers, loads, duties = [], [], [], []
            probe = None
            paired_direct = np.zeros(12)
            paired_valid = True
            alpha = cfg.alpha * cfg.alpha_decay ** iteration
            set_layout(base, z)
            if baseline_env is not None:
                set_layout(baseline_env, z)
            for episode in range(cfg.episodes):
                if cfg.design_gradient == "paired" and iteration >= cfg.warmup_iterations:
                    discount = cfg.gamma ** np.arange(cfg.horizon)
                    probe_times = rng.choice(cfg.horizon, cfg.design_probes_per_episode,
                                             p=discount / discount.sum())
                    probe_counts = np.bincount(probe_times, minlength=cfg.horizon)
                else:
                    probe_counts = np.zeros(cfg.horizon, dtype=int)
                seed = int(rng.integers(0, 2**31))
                if training_conditions:
                    index = rng.choice(len(training_conditions), p=wind_weights)
                    direction, speed, _ = training_conditions[index]
                    base.reset(options={"wind_speed": float(speed),
                                        "wind_direction": float(direction)})
                else:
                    base.reset(seed=seed)
                wind = base._state["freewind_measurements"]
                if baseline_env is not None:
                    key = (float(wind[0]), float(wind[1]))
                    if key not in baseline_cache:
                        baseline_env.reset(options={"wind_speed": key[0],
                                                    "wind_direction": key[1]})
                        calls += 1
                        baseline_costs = []
                        for _ in range(cfg.horizon):
                            _, baseline_reward, _, _, _ = baseline_env.step(
                                {"yaw": np.zeros(N, dtype=np.float32)})
                            baseline_costs.append(-float(baseline_reward.item()))
                            calls += 1
                        baseline_cache[key] = discounted_returns(
                            tensor(baseline_costs),
                            tensor([0.] * (cfg.horizon - 1) + [1.]), cfg.gamma)
                    no_yaw_returns.append(baseline_cache[key])
                trial.reset(options={"wind_speed": float(wind[0]),
                                     "wind_direction": float(wind[1])})
                calls += 2  # Both resets calculate one wake.
                for t in range(cfg.horizon):
                    s = state_vector(base)
                    z_base_t = tensor(z).unsqueeze(0)
                    with torch.no_grad():
                        a = actor.sample(tensor(s).unsqueeze(0), z_base_t)[0].numpy()
                    if probe_counts[t]:
                        physical, probe_calls = one_step_gradient(
                            trial, base, z, s, a, actor, 0.0, cfg)
                        calls += probe_calls
                        if np.isfinite(physical).all():
                            paired_direct += (probe_counts[t] /
                                              (cfg.design_probes_per_episode * cfg.episodes)) * physical
                        else:
                            paired_valid = False
                    if (cfg.design_gradient == "learned" and probe is None
                            and iteration >= cfg.warmup_iterations):
                        physical_gradient, probe_calls = one_step_gradient(
                            trial, base, z, s, a, actor, alpha, cfg)
                        probe = (s, a, physical_gradient)
                        calls += probe_calls
                    z_trial = perturb(z, cfg, rng)
                    s_trial, cost, trial_done, _ = trial_step(trial, base, z_trial, a)
                    _, reward, _, base_done, info = base.step({"yaw": a.copy()})
                    calls += 2
                    if trial_done != base_done:
                        raise RuntimeError("Trial and base horizons diverged")
                    transitions.append((s, a, s_trial, cost, -float(reward.item()),
                                        float(trial_done), z_trial))
                    base_samples.append((s, a, t, episode))
                    rewards.append(float(reward.item()))
                    powers.append(float(np.sum(info["power"])))
                    loads.append(float(np.mean(np.abs(info["load"]))))
                    if base_done:
                        break
                duties.append(float(np.mean(base.accumulated_actions["yaw"] /
                                            (0.3 * base.num_moves * 60))))
            states = tensor(np.stack([x[0] for x in transitions]))
            actions = tensor(np.stack([x[1] for x in transitions]))
            next_states = tensor(np.stack([x[2] for x in transitions]))
            costs = tensor([x[3] for x in transitions])
            base_costs = tensor([x[4] for x in transitions])
            dones = tensor([x[5] for x in transitions])
            designs = tensor(np.stack([x[6] for x in transitions]))
            z_base = tensor(z).expand(len(transitions), -1)
            for _ in range(cfg.critic_updates):
                ids = torch.randint(len(transitions), (min(cfg.batch_size, len(transitions)),))
                with torch.no_grad():
                    a_next = actor.sample(next_states[ids], z_base[ids])
                    y = (base_costs[ids] + alpha * actor.log_prob(states[ids], z_base[ids], actions[ids])
                         + cfg.gamma * (1 - dones[ids])
                         * target(next_states[ids], a_next, z_base[ids]))
                loss = (critic(states[ids], actions[ids], z_base[ids]) - y).square().mean()
                critic_opt.zero_grad()
                loss.backward()
                critic_opt.step()
                with torch.no_grad():
                    for p, q in zip(critic.parameters(), target.parameters()):
                        q.lerp_(p, 0.01)
            for _ in range(cfg.sensitivity_updates):
                ids = torch.randint(len(transitions), (min(cfg.batch_size, len(transitions)),))
                predicted = (sensitivity(states[ids], actions[ids]) *
                             (designs[ids] - z_base[ids])).sum(-1)
                slope_loss = ((predicted - (costs[ids] - base_costs[ids])) / 0.1).square().mean()
                sensitivity_opt.zero_grad()
                slope_loss.backward()
                sensitivity_opt.step()
            weights = tensor([cfg.gamma ** x[2] for x in base_samples])
            weights /= weights.sum()
            q_base = critic(states, actions, z_base)
            logp_base = actor.log_prob(states, z_base, actions)
            if cfg.policy_gradient == "mc":
                with torch.no_grad():
                    returns = discounted_returns(base_costs + alpha * logp_base.detach(),
                                                 dones, cfg.gamma)
                    times = torch.tensor([sample[2] for sample in base_samples])
                    if cfg.policy_baseline == "no_yaw_residual":
                        no_yaw = torch.cat(no_yaw_returns)
                        baseline = no_yaw + leave_one_out_baseline(returns - no_yaw, times)
                    else:
                        baseline = leave_one_out_baseline(returns, times)
                    policy_signal = returns - baseline + alpha
                    writer.add_scalar("diagnostic/critic_mc_rmse",
                                      (q_base.detach() - returns).square().mean().sqrt().item(),
                                      iteration)
                    writer.add_scalar("train/advantage_std", policy_signal.std().item(),
                                      iteration)
            else:
                policy_signal = q_base.detach() + alpha
            actor_loss = (weights * policy_signal * logp_base).sum()
            actor_opt.zero_grad()
            actor_loss.backward()
            # Calculate both gradients at the same actor/critic parameters.
            z_grad_input = tensor(z).repeat(len(transitions), 1).requires_grad_()
            lp = actor.log_prob(states, z_grad_input, actions)
            local_cost_delta = (sensitivity(states, actions).detach() *
                                (z_grad_input - z_base)).sum(-1)
            if cfg.design_gradient == "paired":
                design_objective = (weights * policy_signal * lp).sum()
            else:
                design_objective = (weights * (local_cost_delta + policy_signal * lp)).sum()
            gradient = torch.autograd.grad(design_objective, z_grad_input)[0].sum(0)
            design_gradient = gradient.detach().numpy().astype(np.float64)
            if cfg.design_gradient == "paired":
                design_gradient += paired_direct
                writer.add_scalar("diagnostic/paired_direct_norm",
                                  np.linalg.norm(paired_direct), iteration)
            gradient_check_passed = False
            if cfg.design_gradient == "paired":
                gradient_check_passed = (iteration >= cfg.warmup_iterations and
                                         paired_valid and np.isfinite(design_gradient).all())
            elif probe is not None:
                probe_s, probe_a, physical = probe
                z_probe = tensor(z).unsqueeze(0).requires_grad_()
                probe_state = tensor(probe_s).unsqueeze(0)
                probe_action = tensor(probe_a).unsqueeze(0)
                local = (sensitivity(probe_state, probe_action).detach() *
                         (z_probe - tensor(z).unsqueeze(0))).sum()
                local = local + alpha * actor.log_prob(probe_state, z_probe, probe_action).sum()
                learned = torch.autograd.grad(local, z_probe)[0][0].detach().numpy()
                valid = np.isfinite(physical)
                if valid.any():
                    cosine = float(np.dot(learned[valid], physical[valid]) /
                                   (np.linalg.norm(learned[valid]) *
                                    np.linalg.norm(physical[valid]) + 1e-12))
                    relative_error = float(np.linalg.norm(learned[valid] - physical[valid]) /
                                           (np.linalg.norm(physical[valid]) + 1e-12))
                    writer.add_scalar("diagnostic/one_step_gradient_cosine", cosine, iteration)
                    writer.add_scalar("diagnostic/one_step_gradient_relative_error",
                                      relative_error, iteration)
                    writer.add_scalar("diagnostic/checked_coordinates", int(valid.sum()), iteration)
                    gradient_check_passed = (cosine >= cfg.min_gradient_cosine and
                                             relative_error <= cfg.max_gradient_relative_error)
            actor_opt.step()
            step = 0.0
            if gradient_check_passed:
                z, step = design_step(z, design_gradient, cfg)
            writer.add_scalar("train/reward_mean", np.mean(rewards), iteration)
            writer.add_scalar("train/power_mw_mean", np.mean(powers), iteration)
            writer.add_scalar("train/load_proxy_mean", np.mean(loads), iteration)
            writer.add_scalar("train/yaw_duty_fraction_mean", np.mean(duties), iteration)
            writer.add_scalar("loss/critic", loss.item(), iteration)
            writer.add_scalar("loss/cost_sensitivity", slope_loss.item(), iteration)
            writer.add_scalar("loss/actor", actor_loss.item(), iteration)
            writer.add_scalar("design/gradient_norm", np.linalg.norm(design_gradient), iteration)
            writer.add_scalar("design/accepted_step", step, iteration)
            writer.add_scalar("design/gradient_check_passed", int(gradient_check_passed), iteration)
            writer.add_scalar("train/entropy_temperature", alpha, iteration)
            for i, value in enumerate(z):
                writer.add_scalar(f"layout/offset_{i + 1:02d}_m", value, iteration)
            improved = False
            if cfg.eval_every > 0 and (iteration % cfg.eval_every == 0
                                       or iteration == cfg.iterations - 1):
                score, energy, load, duty, eval_calls = evaluate(base, actor, z, cfg, conditions)
                calls += eval_calls
                improved = score > best_score
                best_score = max(best_score, score)
                writer.add_scalar("eval/windrose_score", score, iteration)
                writer.add_scalar("eval/energy_mwh", energy, iteration)
                writer.add_scalar("eval/load_proxy_mean", load, iteration)
                writer.add_scalar("eval/yaw_duty_fraction_mean", duty, iteration)
                no_yaw_score, _, _, _, eval_calls = evaluate(
                    base, None, z, cfg, conditions)
                calls += eval_calls
                writer.add_scalar("eval/no_yaw_same_layout", no_yaw_score, iteration)
                writer.add_scalar("eval/controller_gain", score - no_yaw_score, iteration)
                writer.add_scalar("eval/layout_gain_no_yaw",
                                  no_yaw_score - reference_no_yaw, iteration)
            writer.add_scalar("simulator/wake_calls", calls, iteration)
            print(f"iteration={iteration} reward={np.mean(rewards):.4f} "
                  f"calls={calls} gradient={np.linalg.norm(design_gradient):.4g} "
                  f"check={gradient_check_passed}", flush=True)
            checkpoint = {"actor": actor.state_dict(), "critic": critic.state_dict(),
                          "cost_sensitivity": sensitivity.state_dict(),
                          "target": target.state_dict(), "layout": z.copy(),
                          "iteration": iteration, "calls": calls,
                          "best_score": best_score}
            torch.save(checkpoint, run_dir / "latest.pt")
            if improved:
                torch.save(checkpoint, run_dir / "best.pt")
            writer.flush()
    finally:
        writer.close()
        base.close()
        trial.close()
        if baseline_env is not None:
            baseline_env.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--site", nargs=4, type=float, required=True,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    p.add_argument("--max-displacement", type=float, required=True)
    p.add_argument("--min-spacing", type=float, required=True)
    p.add_argument("--run-dir", required=True)
    for name, kind in [("seed", int), ("iterations", int), ("episodes", int),
                       ("horizon", int), ("gamma", float), ("load_coef", float),
                       ("alpha", float), ("alpha_decay", float),
                       ("perturb_sigma", float), ("actor_lr", float),
                       ("critic_lr", float), ("design_lr", float),
                       ("critic_updates", int), ("sensitivity_updates", int),
                       ("batch_size", int),
                       ("warmup_iterations", int), ("eval_every", int),
                       ("wind_data", str), ("finite_difference_m", float),
                       ("min_gradient_cosine", float),
                       ("max_gradient_relative_error", float),
                       ("policy_gradient", str), ("policy_baseline", str),
                       ("training_wind", str),
                       ("design_gradient", str), ("design_probes_per_episode", int),
                       ("design_step_cap_m", float)]:
        p.add_argument("--" + name.replace("_", "-"), type=kind,
                       default=getattr(Config, name))
    p.add_argument("--zero-init-actor", action="store_true")
    a = p.parse_args()
    values = vars(a)
    values["site"] = tuple(values["site"])
    return Config(**values)


if __name__ == "__main__":
    train(parse_args())
