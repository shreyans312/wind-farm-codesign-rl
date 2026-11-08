import numpy as np
import torch

from envpg import (
    Actor, Config, action_mask, feasible, make_env, one_step_gradient,
    set_layout, state_vector, trial_step,
)


def config(tmp_path):
    # These loose bounds are only for a simulator smoke test.
    return Config(site=(0, 3000, -1000, 1000), max_displacement=100,
                  min_spacing=1, run_dir=str(tmp_path / "run"), horizon=3)


def test_layout_and_bounded_policy(tmp_path):
    cfg = config(tmp_path)
    assert feasible(np.zeros(12), cfg)
    assert not feasible(np.full(12, 200), cfg)
    actor = Actor(cfg.horizon, cfg.max_displacement)
    s = torch.tensor([[8., 270.] + [0.] * 14 + [0.]])
    z = torch.zeros((1, 12))
    action = actor.sample(s, z)
    assert action.shape == (1, 7)
    assert torch.all(action.abs() <= 5)
    assert torch.isfinite(actor.log_prob(s, z, action)).all()
    assert action_mask(s).all()


def test_trial_matches_base_at_same_layout(tmp_path):
    cfg = config(tmp_path)
    base, trial = make_env(cfg), make_env(cfg)
    try:
        z = np.zeros(12)
        set_layout(base, z)
        base.reset(options={"wind_speed": 8., "wind_direction": 270.})
        trial.reset(options={"wind_speed": 8., "wind_direction": 270.})
        for action in [np.array([2., 0., 0., 0., 0., 0., 0.]), np.zeros(7)]:
            next_trial, cost, done_trial, info_trial = trial_step(trial, base, z, action)
            _, reward, _, done_base, info_base = base.step({"yaw": action.copy()})
            np.testing.assert_allclose(next_trial, state_vector(base), atol=1e-5)
            np.testing.assert_allclose(cost, -reward.item(), atol=1e-5)
            np.testing.assert_allclose(info_trial["power"], info_base["power"], atol=1e-5)
            np.testing.assert_allclose(info_trial["load"], info_base["load"], atol=1e-5)
            assert done_trial == done_base
    finally:
        base.close()
        trial.close()


def test_physical_layout_gradient_is_measurable(tmp_path):
    cfg = config(tmp_path)
    base, trial = make_env(cfg), make_env(cfg)
    try:
        base.reset(options={"wind_speed": 8., "wind_direction": 270.})
        trial.reset(options={"wind_speed": 8., "wind_direction": 270.})
        s = state_vector(base)
        a = np.zeros(7)
        gradient, calls = one_step_gradient(
            trial, base, np.zeros(12), s, a,
            Actor(cfg.horizon, cfg.max_displacement), 0.0, cfg)
        assert calls == 24
        assert np.isfinite(gradient).all()
        assert np.linalg.norm(gradient) > 1e-4
    finally:
        base.close()
        trial.close()
