# Ablaincourt FLORIS environment

## Environment

The selected case is `Ablaincourt_Floris`, with seven NREL 5 MW turbines and one centralized controller. FLORIS recalculates a steady wake field after each joint yaw decision. The control interval is 60 seconds. The turbine positions in metres are

$$
\begin{aligned}
\mathbf{x}^0 &= (484.8,\ 797.1,\ 1038.8,\ 1377.6,\ 1716.9,\ 2057.3,\ 2400.0),\\
\mathbf{y}^0 &= (274.0,\ 251.0,\ 66.9,\ -22.7,\ -112.5,\ -195.3,\ -259.0).
\end{aligned}
$$

The FLORIS template uses air density 1.225 kg/m³, turbulence intensity 0.06, wind shear 0.12, and a 3 × 3 rotor grid. These and the turbine model remain fixed in the co-design study.

## Observation

`reset()` and `step()` return `yaw`, `wind_speed`, and `wind_direction` arrays of length seven, plus `freewind_measurements = (speed, direction)`. The centralized observation therefore has 23 values. Yaw is the commanded target in degrees; local wind speed is in m/s and wind direction is in degrees. Power and the load proxy appear in `step()`'s `info` dictionary.

This observation omits elapsed control steps and accumulated yaw movement, both of which affect the duty guard. Local wind measurements also depend on turbine positions. The EnvPG note defines a state with a common meaning across candidate layouts.

## Control

The agent supplies one yaw increment for each turbine:

$$
\mathbf a_t=(\Delta\psi_{1,t},\ldots,\Delta\psi_{7,t}),\qquad
\Delta\psi_{i,t}\in[-5^\circ,5^\circ],\qquad
\psi_{i,t+1}=\operatorname{clip}(\psi_{i,t}+\Delta\psi_{i,t}^{\mathrm{exec}},-40^\circ,40^\circ).
$$

The environment may replace a requested increment with zero when its estimated actuation time has reached 10% of elapsed time. It estimates movement using 0.3°/s and previous commands, so this is a retrospective accounting rule rather than a direct measurement of actuator travel.

## Layout

The co-design variables are offsets from the coordinates above. Turbine 1 remains fixed as a coordinate anchor; each of turbines 2–7 can move in x and y, giving 12 variables. The layout is constant during an episode. A site boundary, movement limits around the reference positions, and minimum turbine separation are required before optimization; the repository does not specify their values.

## Reward and wind

With $P_{i,t+1}^{\mathrm{kW}}$ denoting turbine power and $\widetilde L_{i,k,t+1}$ the four scaled FLORIS load-proxy channels, the unshaped reward is

$$
r_t=\frac{1}{7}\sum_{i=1}^{7}\frac{P_{i,t+1}^{\mathrm{kW}}}{u_{\infty,t}^{3}}
-\frac{0.1}{28}\sum_{i=1}^{7}\sum_{k=1}^{4}|\widetilde L_{i,k,t+1}|.
$$

The load channels describe turbulence and wind variation over rotor planes; they are not lifetime fatigue measurements. Wind is an external condition. Without an explicit reset value, speed is sampled as $8W$ for $W\sim\mathrm{Weibull}(8)$ and clipped to 3–28 m/s; direction is sampled from $\mathcal N(270^\circ,(20^\circ)^2)$ and wrapped to 0–360°. `wind_time_series` supplies changing wind within an episode.

## Configuration

The relevant constructor settings are `controls` (yaw), `continuous_control` (true), `max_num_steps` (500), `load_coef` (0.1), `reward_shaper` (identity), and `log` (true). The case has `dt=60 s` and `t_init=0`; the simulator horizon is derived from warm-up and requested episode length. `reset(seed=..., options=...)` controls wind sampling or passes explicit speed and direction. Reset itself advances the simulator once, so the effective controlled step count should be checked when setting an experimental horizon.
