# EnvPG for the seven-turbine Ablaincourt FLORIS case. Cost is minus the unchanged WFCRL reward

import argparse
import copy
import hashlib
import json
import math
import shutil
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
DESIGN_DIM = 4


def layout_basis():
    tangent = REFERENCE[-1] - REFERENCE[0]
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    coordinate = (REFERENCE - REFERENCE.mean(axis=0)) @ tangent
    coordinate /= np.max(np.abs(coordinate))
    stagger = (-1.0) ** np.arange(N)
    stagger -= stagger.mean()
    modes = np.stack((coordinate[:, None] * tangent,
                      coordinate[:, None] * normal,
                      (coordinate**2 - np.mean(coordinate**2))[:, None] * normal,
                      stagger[:, None] * normal))
    return modes / np.max(np.linalg.norm(modes, axis=2), axis=1)[:, None, None]


BASIS = layout_basis()


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
    gamma: float = 1.0
    load_coef: float = 0.1
    alpha: float = 0.01
    alpha_decay: float = 0.9
    perturb_sigma: float = 10.0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    design_lr: float = 100.0 / 150
    critic_updates: int = 200
    batch_size: int = 128
    warmup_iterations: int = 20
    eval_every: int = 10
    wind_data: str = str(ROOT / "wfcrl-benchmark" / "data" / "smarteole.csv")
    finite_difference_m: float = 1.0
    min_gradient_cosine: float = 0.5
    max_gradient_relative_error: float = 1.0
    zero_init_actor: bool = True
    target_tracking: bool = True


def layout(z):
    return REFERENCE + np.einsum("k,kij->ij", np.asarray(z).reshape(DESIGN_DIM), BASIS)


def feasible(z, cfg):
    p = layout(z)
    xmin, xmax, ymin, ymax = cfg.site
    if not (xmin <= p[:, 0].min() and p[:, 0].max() <= xmax
            and ymin <= p[:, 1].min() and p[:, 1].max() <= ymax):
        return False
    if np.any(np.linalg.norm(p - REFERENCE, axis=1) > cfg.max_displacement):
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
    def __init__(self, horizon, max_displacement, zero_mean_init=False,
                 target_tracking=True, design_dim=DESIGN_DIM):
        super().__init__()
        self.horizon = horizon
        self.max_displacement = max_displacement
        self.target_tracking = target_tracking
        self.mean = network(17 + design_dim, N)
        self.log_std = nn.Parameter(torch.full((N,), -0.7))
        if zero_mean_init:
            nn.init.zeros_(self.mean[-1].weight)
            nn.init.zeros_(self.mean[-1].bias)

    def distribution(self, s, z):
        x = torch.cat((normalize_state(s, self.horizon), z / self.max_displacement), -1)
        mean = self.mean(x)
        if self.target_tracking:
            target_yaw = 40 * torch.tanh(mean)
            mean = (target_yaw - s[..., 2:2 + N]) / 5
        return Normal(mean, self.log_std.clamp(-5, 2).exp())

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
    def __init__(self, horizon, max_displacement, gamma=0.99):
        super().__init__()
        self.horizon = horizon
        self.max_displacement = max_displacement
        self.gamma = gamma
        self.value = network(17 + N + DESIGN_DIM, 1)
        self.local_design = network(17 + N + DESIGN_DIM, 1)

    def remaining_scale(self, s):
        remaining = (self.horizon - s[..., -1]).clamp(min=0)
        if self.gamma == 1:
            return remaining
        return (1 - self.gamma ** remaining) / (1 - self.gamma)

    def forward(self, s, a, current_design, future_design=None):
        if future_design is None:
            future_design = current_design
        state_action = torch.cat((normalize_state(s, self.horizon), a / 5), -1)
        future_input = torch.cat((state_action, future_design / self.max_displacement), -1)
        current_input = torch.cat((state_action, current_design / self.max_displacement), -1)
        base_value = self.remaining_scale(s) * self.value(future_input).squeeze(-1)
        local_difference = (self.local_design(current_input) -
                            self.local_design(future_input)).squeeze(-1)
        return base_value + local_difference


def tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def discounted_returns(costs, dones, gamma):
    returns = torch.empty_like(costs)
    future = 0.0
    for i in range(len(costs) - 1, -1, -1):
        future = costs[i] + gamma * (1 - dones[i]) * future
        returns[i] = future
    return returns


def generalized_td_target(actor, target, states, actions, next_states,
                          costs, dones, current_designs, future_designs, alpha, gamma):
    with torch.no_grad():
        next_actions = actor.sample(next_states, future_designs)
        return (costs + alpha * actor.log_prob(states, current_designs, actions)
                + gamma * (1 - dones)
                * target(next_states, next_actions, future_designs))


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
        candidate = z + rng.normal(0, cfg.perturb_sigma, DESIGN_DIM)
        if feasible(candidate, cfg):
            return candidate
    raise RuntimeError("No feasible perturbation found; reduce --perturb-sigma")


def design_step(z, gradient, cfg):
    step = cfg.design_lr
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
    result = np.full(DESIGN_DIM, np.nan)
    calls = 0
    for i in range(DESIGN_DIM):
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
    if not feasible(np.zeros(DESIGN_DIM), cfg):
        raise ValueError("The reference layout violates the supplied constraints")
    if cfg.horizon < 1 or cfg.episodes < 1 or cfg.iterations < 1:
        raise ValueError("Horizon, episodes and iterations must be positive")
    if min(cfg.critic_updates, cfg.batch_size) < 1:
        raise ValueError("Update counts and batch size must be positive")
    if not (0 <= cfg.gamma <= 1 and cfg.alpha >= 0 and 0 < cfg.alpha_decay <= 1):
        raise ValueError("Invalid discount or entropy settings")
    if cfg.warmup_iterations < 0 or cfg.eval_every < 0:
        raise ValueError("Warm-up and evaluation interval must be nonnegative")
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
    shutil.copy2(__file__, run_dir / "source.py")
    writer = SummaryWriter(run_dir / "tb")
    base, trial = make_env(cfg), make_env(cfg)
    actor = Actor(cfg.horizon, cfg.max_displacement,
                  cfg.zero_init_actor, cfg.target_tracking)
    critic = Critic(cfg.horizon, cfg.max_displacement, cfg.gamma)
    target = copy.deepcopy(critic)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    z = np.zeros(DESIGN_DIM, dtype=np.float64)
    calls = 0
    conditions = windrose(cfg.wind_data) if cfg.eval_every > 0 else []
    training_conditions = windrose(cfg.wind_data)
    wind_weights = np.array([condition[2] for condition in training_conditions])
    reference_no_yaw = None
    if conditions:
        reference_no_yaw, _, _, _, eval_calls = evaluate(
            base, None, np.zeros(DESIGN_DIM), cfg, conditions)
        calls += eval_calls
    best_score = -np.inf
    try:
        for iteration in range(cfg.iterations):
            # The generalized critic's implicit future-layout copy changes each iteration.
            transitions, base_samples = [], []
            rewards, powers, loads, duties, yaw_angles = [], [], [], [], []
            probe = None
            alpha = cfg.alpha * cfg.alpha_decay ** iteration
            set_layout(base, z)
            for episode in range(cfg.episodes):
                index = rng.choice(len(training_conditions), p=wind_weights)
                direction, speed, _ = training_conditions[index]
                base.reset(options={"wind_speed": float(speed),
                                    "wind_direction": float(direction)})
                wind = base._state["freewind_measurements"]
                trial.reset(options={"wind_speed": float(wind[0]),
                                     "wind_direction": float(wind[1])})
                calls += 2  # Both resets calculate one wake.
                for t in range(cfg.horizon):
                    s = state_vector(base)
                    z_base_t = tensor(z).unsqueeze(0)
                    with torch.no_grad():
                        a = actor.sample(tensor(s).unsqueeze(0), z_base_t)[0].numpy()
                    if probe is None and iteration >= cfg.warmup_iterations:
                        physical_gradient, probe_calls = one_step_gradient(
                            trial, base, z, s, a, actor, alpha, cfg)
                        probe = (s, a, physical_gradient)
                        calls += probe_calls
                    z_trial = perturb(z, cfg, rng)
                    with torch.no_grad():
                        a_trial = actor.sample(tensor(s).unsqueeze(0),
                                               tensor(z_trial).unsqueeze(0))[0].numpy()
                    s_trial, cost, trial_done, _ = trial_step(trial, base, z_trial, a_trial)
                    paired_next, paired_cost, paired_done, _ = trial_step(
                        trial, base, z_trial, a)
                    _, reward, _, base_done, info = base.step({"yaw": a.copy()})
                    base_next = state_vector(base)
                    calls += 3
                    if not (trial_done == paired_done == base_done):
                        raise RuntimeError("Paired and base horizons diverged")
                    if not np.allclose(paired_next, base_next, atol=1e-5):
                        raise RuntimeError("Layout changed the canonical next state")
                    transitions.append((s, a, s_trial, cost, -float(reward.item()),
                                        float(trial_done), z_trial, a_trial, paired_cost))
                    base_samples.append(t)
                    rewards.append(float(reward.item()))
                    powers.append(float(np.sum(info["power"])))
                    loads.append(float(np.mean(np.abs(info["load"]))))
                    yaw_angles.append(float(np.mean(np.abs(base._state["yaw"]))))
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
            trial_actions = tensor(np.stack([x[7] for x in transitions]))
            paired_costs = tensor([x[8] for x in transitions])
            z_base = tensor(z).expand(len(transitions), -1)
            with torch.no_grad():
                logp_base_all = actor.log_prob(states, z_base, actions)
                logp_paired_all = actor.log_prob(states, designs, actions)
            for _ in range(cfg.critic_updates):
                ids = torch.randint(len(transitions), (min(cfg.batch_size, len(transitions)),))
                y = generalized_td_target(
                    actor, target, states[ids], trial_actions[ids], next_states[ids],
                    costs[ids], dones[ids], designs[ids], z_base[ids], alpha, cfg.gamma)
                prediction = critic(states[ids], trial_actions[ids],
                                    designs[ids], z_base[ids])
                q_base_batch = critic(states[ids], actions[ids], z_base[ids])
                with torch.no_grad():
                    design_target = (paired_costs[ids] - base_costs[ids] + alpha *
                                     (logp_paired_all[ids] - logp_base_all[ids]))
                q_design = critic(states[ids], actions[ids], designs[ids], z_base[ids])
                scale = critic.remaining_scale(states[ids])
                td_scale = (y / scale).square().mean().sqrt().detach().clamp_min(1e-3)
                design_scale = design_target.square().mean().sqrt().clamp_min(1e-3)
                td_loss = (((prediction - y) / scale) / td_scale).square().mean()
                design_loss = (((q_design - q_base_batch) - design_target) /
                               design_scale).square().mean()
                loss = td_loss + design_loss
                critic_opt.zero_grad()
                loss.backward()
                critic_opt.step()
                with torch.no_grad():
                    for p, q in zip(critic.parameters(), target.parameters()):
                        q.lerp_(p, 0.01)
            weights = tensor([cfg.gamma ** t for t in base_samples])
            weights /= cfg.episodes
            q_base = critic(states, actions, z_base)
            logp_base = actor.log_prob(states, z_base, actions)
            with torch.no_grad():
                mc_returns = discounted_returns(
                    base_costs + alpha * logp_base.detach(), dones, cfg.gamma)
                q_estimate = q_base.detach()
                writer.add_scalar("diagnostic/critic_mc_rmse",
                                  (q_estimate - mc_returns).square().mean().sqrt().item(), iteration)
                writer.add_scalar("diagnostic/critic_q_mean", q_estimate.mean().item(), iteration)
                writer.add_scalar("diagnostic/critic_mc_mean", mc_returns.mean().item(), iteration)
                centered_q = q_estimate - q_estimate.mean()
                centered_mc = mc_returns - mc_returns.mean()
                correlation = (centered_q * centered_mc).mean() / (
                    centered_q.square().mean().sqrt() *
                    centered_mc.square().mean().sqrt() + 1e-12)
                writer.add_scalar("diagnostic/critic_mc_correlation", correlation.item(), iteration)
            # An action-independent baseline leaves Eq. 15 unchanged in expectation.
            with torch.no_grad():
                q_zero = critic(states, torch.zeros_like(actions), z_base)
                policy_signal = q_base.detach() - q_zero + alpha
                writer.add_scalar("diagnostic/actor_advantage_std",
                                  (q_base.detach() - q_zero).std().item(), iteration)
            actor_loss = (weights * policy_signal * logp_base).sum()
            actor_opt.zero_grad()
            actor_loss.backward()
            # Calculate both gradients at the same actor/critic parameters.
            z_grad_input = tensor(z).repeat(len(transitions), 1).requires_grad_()
            lp = actor.log_prob(states, z_grad_input, actions)
            design_objective = (weights * (critic(states, actions, z_grad_input, z_base)
                                           + (q_base.detach() - q_zero) * lp)).sum()
            gradient = torch.autograd.grad(design_objective, z_grad_input)[0].sum(0)
            design_gradient = gradient.detach().numpy().astype(np.float64)
            update_allowed = (iteration >= cfg.warmup_iterations and
                              np.isfinite(design_gradient).all())
            if probe is not None:
                probe_s, probe_a, physical = probe
                z_probe = tensor(z).unsqueeze(0).requires_grad_()
                probe_state = tensor(probe_s).unsqueeze(0)
                probe_action = tensor(probe_a).unsqueeze(0)
                local = critic(probe_state, probe_action, z_probe,
                               tensor(z).unsqueeze(0)).sum()
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
                    physical_check_passed = (cosine >= cfg.min_gradient_cosine and
                                             relative_error <= cfg.max_gradient_relative_error)
                    writer.add_scalar("diagnostic/physical_gradient_check_passed",
                                      int(physical_check_passed), iteration)
            actor_opt.step()
            step = 0.0
            if update_allowed:
                z, step = design_step(z, design_gradient, cfg)
            writer.add_scalar("train/reward_mean", np.mean(rewards), iteration)
            writer.add_scalar("train/power_mw_mean", np.mean(powers), iteration)
            writer.add_scalar("train/load_proxy_mean", np.mean(loads), iteration)
            writer.add_scalar("train/yaw_duty_fraction_mean", np.mean(duties), iteration)
            writer.add_scalar("train/yaw_angle_abs_mean_deg", np.mean(yaw_angles), iteration)
            writer.add_scalar("loss/critic", loss.item(), iteration)
            writer.add_scalar("loss/critic_td", td_loss.item(), iteration)
            writer.add_scalar("loss/critic_design_contrast", design_loss.item(), iteration)
            writer.add_scalar("diagnostic/actor_surrogate", actor_loss.item(), iteration)
            writer.add_scalar("design/gradient_norm", np.linalg.norm(design_gradient), iteration)
            writer.add_scalar("design/accepted_step", step, iteration)
            writer.add_scalar("design/update_allowed", int(update_allowed), iteration)
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
                  f"update={update_allowed}", flush=True)
            checkpoint = {"actor": actor.state_dict(), "critic": critic.state_dict(),
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
                       ("critic_updates", int),
                       ("batch_size", int),
                       ("warmup_iterations", int), ("eval_every", int),
                       ("wind_data", str), ("finite_difference_m", float),
                       ("min_gradient_cosine", float),
                       ("max_gradient_relative_error", float)]:
        p.add_argument("--" + name.replace("_", "-"), type=kind,
                       default=getattr(Config, name))
    p.add_argument("--zero-init-actor", action=argparse.BooleanOptionalAction,
                   default=Config.zero_init_actor)
    p.add_argument("--target-tracking", action=argparse.BooleanOptionalAction,
                   default=Config.target_tracking)
    a = p.parse_args()
    values = vars(a)
    values["site"] = tuple(values["site"])
    return Config(**values)


if __name__ == "__main__":
    train(parse_args())
