# EnvPG on Ablaincourt

## Model

Turbine 1 stays at its reference position. The x/y offsets of turbines 2–7 form the design \(\mathbf z\in\mathbb R^{12}\). Site bounds, maximum displacement and minimum spacing are inputs; WFCRL does not specify them. A layout stays fixed for an episode.

$$
s_t=(u_\infty,\phi_\infty,\boldsymbol\psi_t,\mathbf h_t,t),
\qquad
\mathbf a_t=(\Delta\psi_{1,t},\ldots,\Delta\psi_{7,t})\in[-5^\circ,5^\circ]^7.
$$

The 17-value state contains freestream wind, seven yaw targets, seven accumulated yaw movements and time. The actor also receives \(\mathbf z\). Stock local wind measurements are omitted because they change meaning with layout. With fixed wind and steady FLORIS, layout changes reward but not this state's next-step transition. The actor uses a bounded stochastic action distribution and sets duty-blocked increments to zero.

Set \(c_{\mathbf z}=-r_{\mathbf z}\). With entropy weight \(\alpha\), EnvPG minimizes

$$
J_H(\theta,\mathbf z)=
\mathbb E\left[\sum_{t=0}^{T-1}\gamma^t
\left(c_{\mathbf z}(s_t,a_t,s_{t+1})
+\alpha\log\pi_\theta(a_t\mid s_t,\mathbf z)\right)\right].
$$

## Update

At base layout \(\mathbf z_0\), collect policy rollouts. For each sampled state and action, run one paired transition at a nearby feasible layout \(\mathbf z'\). A base-layout critic \(Q_\varphi\) learns by TD; a separate local slope fits the paired cost difference:

$$
g_\omega(s,a)^\top(\mathbf z'-\mathbf z_0)
\approx c_{\mathbf z'}(s,a,s')-c_{\mathbf z_0}(s,a,s').
$$

The updates use

$$
\begin{aligned}
\widehat{\nabla_\theta J_H}
&=\sum_t\gamma^t(Q_\varphi+\alpha)\nabla_\theta\log\pi_\theta(a_t\mid s_t,\mathbf z_0),\\
\widehat{\nabla_{\mathbf z}J_H}
&=\sum_t\gamma^t\left[g_\omega+
(Q_\varphi+\alpha)\nabla_{\mathbf z}\log\pi_\theta(a_t\mid s_t,\mathbf z_0)\right].
\end{aligned}
$$

Here \(Q_\varphi\) and \(g_\omega\) are evaluated at \((s_t,a_t)\). The local slope replaces direct differentiation of the paper's generalized critic because the chosen state has no layout-dependent transition. A layout step is accepted only when its one-step slope agrees with paired FLORIS finite differences and the new layout is feasible. This continuous-action, constrained version is an empirical adaptation of EnvPG.

## Comparison

Score each fixed layout and policy on the 25 weighted Ablaincourt wind bins, with 150 controlled steps per bin. Use the same reward coefficient (0.1 in this checkout), wind cases, horizon and simulator-call budget for EnvPG, IPPO and MAPPO. Retrain each controller on the original and learned layouts. The WFCRL paper uses a default load coefficient of 1, so its published scores are not directly comparable. A full-trajectory design-gradient check remains to be done before drawing performance conclusions.
