# EnvPG formulation for Ablaincourt

## Design and objective

The starting point is `Ablaincourt_Floris`. One policy controls the seven yaw settings. Let $\mathbf p_i^0=(x_i^0,y_i^0)$ be the repository's turbine positions. Keep turbine 1 fixed and move the other six:

$$
\mathbf z=(\delta x_2,\delta y_2,\ldots,\delta x_7,\delta y_7)\in\mathbb R^{12},\qquad
\mathbf p_1=\mathbf p_1^0,\quad
\mathbf p_i=\mathbf p_i^0+(\delta x_i,\delta y_i)\quad(i=2,\ldots,7).
$$

Feasible layouts must lie inside a specified site, remain within agreed movement limits around $\mathbf p_i^0$, and satisfy a minimum pairwise turbine separation. Those limits have not been defined in WFCRL and require a research decision. Turbine hardware, FLORIS settings, control bounds, and reward stay fixed. A chosen layout stays in place for an entire evaluation episode.

The control objective is to maximize expected discounted WFCRL reward across a defined set of wind conditions. EnvPG is written as cost minimization, so the code uses $c_{\mathbf z}=-r_{\mathbf z}$. With entropy weight $\alpha$, the training objective is

$$
J_H(\theta,\mathbf z)=
\mathbb E_{\pi_\theta,P_{\mathbf z}}
\left[\sum_{t=0}^{T-1}\gamma^t
\left(c_{\mathbf z}(s_t,a_t,s_{t+1})
+\alpha\log\pi_\theta(a_t\mid s_t,\mathbf z)\right)\right].
$$

Episode length $T$, discount $\gamma$, wind set, perturbation scale, entropy schedule, and optimization settings must be fixed and recorded before comparing methods.

## State and action

For this formulation, use

$$
s_t=(u_{\infty},\phi_{\infty},\boldsymbol\psi_t,\mathbf h_t,t),
\qquad
 a_t=(\Delta\psi_{1,t},\ldots,\Delta\psi_{7,t})\in[-5^\circ,5^\circ]^7.
$$

Here $\boldsymbol\psi_t$ contains yaw targets and $\mathbf h_t$ contains accumulated yaw increments used by the duty rule. The policy also receives $\mathbf z$. Supply wind explicitly at reset; yaw, usage, and $t$ start at zero. Thus the initial state has the same meaning under every layout. The state has 17 values before adding the 12 design values to the actor input.

Local wind measurements from the stock observation are omitted. Their values change when layout changes, which complicates a derivative taken at a fixed state. With fixed incoming wind, FLORIS's next state as defined above depends on yaw, action, and duty history; layout affects power and load, hence cost. This first formulation therefore exercises EnvPG's cost-sensitivity term and the design dependence of the policy, but not a design-dependent state transition. A later formulation that gives the actor local wind must account for its layout-dependent observation map.

WFCRL may force a yaw increment to zero under its duty rule. The policy distribution should make blocked components deterministic zeros and compute its log-density over the remaining components. Record requested and executed actions; the latter enter the critic and simulator transition.

## Simulation interface

Keep the implementation outside the stable `wfcrl` package. EnvPG needs a transition function taking $(s,a,\mathbf z)$ and returning $(s',c,\mathbf P,\widetilde{\mathbf L},\mathrm{done})$. It should reproduce `WindFarmEnv`'s clipping, duty accounting, power units, reward, and truncation. FLORIS can be reinitialized with new `layout_x` and `layout_y` values. Use one simulator for base-layout rollouts and another for temporary perturbed-layout transitions, so trials cannot change the baseline trajectory.

At the same wind, layout, state, and yaw action, this interface should agree with the existing centralized environment on yaw targets, power, loads, reward, and episode end. The adapter must also establish the true controlled horizon because WFCRL advances FLORIS during reset.

## EnvPG updates

At an outer iteration let $\mathbf z_0$ be the current layout. Ordinary rollouts use $\mathbf z_0$. Nearby feasible trial layouts $\mathbf z'$ are used for single-transition samples at the same canonical states and actions. The actor $\pi_\theta(a\mid s,\mathbf z)$ is stochastic and bounded. The paper's generalized critic is

$$
Q^H_{\pi,\mathbf z_0}(s,a,\mathbf z')=
\mathbb E_{s'\sim P_{\mathbf z'}(\cdot\mid s,a)}
\left[c_{\mathbf z'}(s,a,s')
+\alpha\log\pi_\theta(a\mid s,\mathbf z')
+\gamma V^H_{\pi,\mathbf z_0}(s')\right].
$$

For the canonical state used here, the next state is independent of layout. The implementation therefore learns a base-layout critic $Q_\varphi$ by TD and a local cost slope $g_\omega(s,a)$ from paired base and trial transitions:

$$
y_0=c_{\mathbf z_0}(s,a,s')
+\alpha\log\pi_\theta(a\mid s,\mathbf z_0)
+\gamma(1-\mathrm{done})
\mathbb E_{a'\sim\pi_\theta(\cdot\mid s',\mathbf z_0)}
\left[Q_{\bar\varphi}(s',a',\mathbf z_0)\right].
$$

$$
g_\omega(s,a)^\top(\mathbf z'-\mathbf z_0)
\approx c_{\mathbf z'}(s,a,s')-c_{\mathbf z_0}(s,a,s').
$$

This locally linear critic decomposition uses the exact paired cost difference as a training target. It is an adaptation of the paper's direct trial-design TD critic, and its slope is checked against paired FLORIS finite differences before a layout step is accepted. The target critic is updated slowly; the trial data are refreshed as the base layout changes.

On base-layout rollouts, descend in policy parameters using the paper's entropy-regularized policy-gradient term

$$
\widehat{\nabla_\theta J_H}=
\sum_t\gamma^t
\bigl(Q_\varphi(s_t,a_t,\mathbf z_0)+\alpha\bigr)
\nabla_\theta\log\pi_\theta(a_t\mid s_t,\mathbf z_0).
$$

After critic warm-up, estimate the design gradient on those same base-layout rollouts:

$$
\widehat{\nabla_{\mathbf z}J_H}=
\sum_t\gamma^t\left[
g_\omega(s_t,a_t)
+\bigl(\alpha+Q_\varphi(s_t,a_t,\mathbf z_0)\bigr)
\left.\nabla_{\mathbf z}\log\pi_\theta(a_t\mid s_t,\mathbf z)\right|_{\mathbf z_0}
\right].
$$

Average these estimates over sampled rollouts and take a feasible step that reduces estimated cost. Feasibility backtracking is a practical extension to the paper's unconstrained update. Base-design actions are used for both paired transitions, as in the paper's pseudocode. The sensitivity fit, finite-difference gate, continuous action distribution, and design constraints are explicit adaptations.

## Evaluation

The code compares the learned current-step layout slope with paired finite differences of entropy-regularized cost at one fixed state and action each iteration after warm-up. It logs cosine similarity and relative error, and accepts a layout step only when both pass configurable limits. Before reporting research results, the full 12-coordinate gradient should also be compared with finite differences of complete fixed-layout $J_H$ under matched wind and action randomness; that longer check is not yet implemented.

The main evaluation uses the Ablaincourt wind rose from WFCRL's Scenario II: 25 speed–direction bins with their empirical weights. Run each policy for 150 controlled steps at each bin and count the common farm reward once per step:

$$
S(\mathbf z,\pi)=\sum_{j=1}^{25}\rho_j\,
\mathbb E\!\left[\sum_{t=0}^{149}r_t\mid w_j,\mathbf z,\pi\right].
$$

Use the same wind bins, the checkout's `load_coef=0.1`, control limits, evaluation horizon, and random seeds for every method. The paper states a default load coefficient of 1, so its published scores are not direct numerical baselines for this setting. Train IPPO and MAPPO from the authors' separate `wfcrl-benchmark` code on the original layout, alongside the centralized EnvPG actor with its design frozen. Then train joint EnvPG and retrain IPPO and MAPPO on its final layout. This gives controller comparisons at each layout and a layout comparison within each controller family. The stock multi-agent policies receive local observations while the proposed EnvPG actor receives the canonical global state, so these are system-level results; any claim about the RL algorithms themselves needs a separate matched-observation comparison.

For the co-design algorithm comparison, also optimize the same 12 layout coordinates by a derivative-free search with a centralized yaw controller retrained at each candidate layout. Charge every simulator call, including EnvPG's trial transitions and the search method's candidate evaluations, to a common budget. Report score against calls, then final score over training seeds, farm energy, load proxy, yaw duty, and layout feasibility. The paper derives EnvPG for finite state and action spaces, so agreement of this continuous-action implementation with finite differences is an empirical validation of the extension rather than a theorem for this simulator.
