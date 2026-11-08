# Ablaincourt FLORIS

## Farm

Ablaincourt_Floris has seven NREL 5 MW turbines. FLORIS computes a steady wake after each 60 s control step. The reference coordinates in metres are

$$
\begin{aligned}
\mathbf x^0&=(484.8,797.1,1038.8,1377.6,1716.9,2057.3,2400.0),\\
\mathbf y^0&=(274.0,251.0,66.9,-22.7,-112.5,-195.3,-259.0).
\end{aligned}
$$

Air density is 1.225 kg/m³, turbulence intensity 0.06, wind shear 0.12, and the rotor grid is 3 × 3.

## Observation and control

The centralized observation has 23 values: seven yaw targets, seven local wind speeds, seven local wind directions, and freestream speed and direction. Power and four load-proxy values per turbine are returned in step info.

$$
\mathbf a_t=(\Delta\psi_{1,t},\ldots,\Delta\psi_{7,t}),
\qquad \Delta\psi_{i,t}\in[-5^\circ,5^\circ].
$$

Yaw targets are clipped to ±40°. The duty guard can replace an increment with zero once accumulated yaw movement, estimated at 0.3°/s, reaches 10% of elapsed time.

## Layout, wind and reward

The design is the x/y offset of turbines 2–7 (12 variables); turbine 1 stays fixed. Layout is constant within an episode. Site bounds, movement limits and minimum spacing must be supplied separately.

For power P in kW, freestream speed u in m/s, and the four scaled FLORIS load proxies L per turbine, the checkout reward is

$$
r_t=\frac{1}{7}\sum_{i=1}^{7}\frac{P_{i,t+1}}{u_{\infty,t}^{3}}
-\frac{0.1}{28}\sum_{i=1}^{7}\sum_{k=1}^{4}|L_{i,k,t+1}|.
$$

Without explicit wind, reset samples speed from a clipped 8 × Weibull(8) law and direction from a wrapped normal centred at 270° with 20° standard deviation. Reset consumes one FLORIS update, so a horizon of T controlled steps needs max_num_steps = T+1.
