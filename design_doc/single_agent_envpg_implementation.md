# EnvPG on Ablaincourt

## Design and control

The design $z\in\mathbb R^4$ moves all seven turbines through four geometry modes: longitudinal spread, transverse shear, transverse bow and alternating stagger. Each mode is normalized so a 1 m coefficient moves at most one turbine by 1 m. This searches a four-dimensional layout subspace. The site $(400,2500,-400,400)$ m, 150 m displacement limit and 250 m minimum spacing are our experiment choices, not WFCRL settings. The layout stays fixed during each 150-step episode.

Let $t$ and $n$ be unit vectors along and across the reference row, and $u_i$ the centered longitudinal coordinate scaled to $[-1,1]$. Before normalizing each mode, the turbine displacements are

$$
B_{1i}=u_i t,\quad B_{2i}=u_i n,\quad
B_{3i}=(u_i^2-\overline{u^2})n,\quad
B_{4i}=((-1)^{i-1}-\overline{(-1)^{i-1}})n,\qquad
p_i(z)=p_i^0+\sum_{k=1}^4 z_k\widetilde B_{ki}.
$$

The controller sees freestream wind, seven yaw angles, seven accumulated yaw movements, time and $z$. It predicts target yaw angles and samples increments in $[-5^\circ,5^\circ]$ around the target error. The WFCRL duty guard can set an increment to zero. We omit local wind measurements because they change with layout, while the chosen state still determines the next yaw and duty state under steady FLORIS.

The paper assumes finite state/action spaces and smooth design dependence. Continuous yaw and WFCRL's hard duty guard make this an adaptation of EnvPG. The stagger-mode FLORIS slope also changes noticeably between 0.5 m and 2 m finite differences, so local smoothness is only approximate; we perturb designs at 1 m scale.

## Objective

Using the unchanged WFCRL reward $r_t$, cost is $c_t=-r_t$. The actor and layout minimize

$$
J_H(\theta,z)=\mathbb E\!\left[\sum_{t=0}^{149}\gamma^t
\big(c_t+\alpha\log\pi_\theta(a_t\mid s_t,z)\big)\right].
$$

The evaluation score is the undiscounted WFCRL reward summed over the same 25 weighted wind bins for every model. We show yaw gain and layout gain separately.

We use $\gamma=1$ because evaluation also sums all 150 rewards. We use $\alpha_0=0.01$ and $\alpha_{k+1}=0.9\alpha_k$. The initial joint log-probability is around $-15$, so this contributes about $-0.15$ per step, versus roughly $-2.6$ physical cost; the temperature then approaches zero.

The actor starts with zero target yaw; its stochastic policy still explores. The critic gets 200 minibatch updates per iteration, and the layout stays fixed for the first 20 iterations. These choices follow the fixed-policy value-calibration check. The layout step size is $100/150$ because its gradient now sums over 150 steps rather than averaging them.

## Generalized critic

At the current layout $z_k$, a nearby feasible $z'$ supplies a one-step transition. The action for that transition is sampled from $\pi_\theta(\cdot\mid s,z')$. The critic sees $z'$ for the current cost and $z_k$ for future value:

$$
y=c_{z'}(s,a,s')+\alpha\log\pi_\theta(a\mid s,z')
+\gamma\,\mathbb E_{a'\sim\pi_\theta(\cdot\mid s',z_k)}
Q_{\bar\phi}(s',a',z_k).
$$

A TD update fits $Q_\phi(s,a,z')$ to $y$. Sampled full-episode returns also fit its value at $z_k$; this auxiliary fit reduces value bias in our finite-horizon case. The policy uses $Q_\phi(s,a,z_k)+\alpha$. The layout uses the two terms of the EnvPG gradient,

$$
\nabla_z Q_\phi(s,a,z)\big|_{z_k}
+Q_\phi(s,a,z_k)\,\nabla_z\log\pi_\theta(a\mid s,z_k).
$$

We also fit a paired Bellman difference. With the same state and action, changing the layout leaves our next state vector unchanged, so the generalized $Q$ difference equals the immediate cost and entropy difference. This extra simulator sample trains the local design slope; updates still use gradients of the learned $Q$.

FLORIS finite differences check the learned layout slope; they do not supply the EnvPG update.

Our critic uses the paper's two design copies explicitly: $Q_\phi(s,a,z;z_k)=h(t)v_\phi(s,a,z_k)+g_\phi(s,a,z_k)^\top(z-z_k)/150$, with $h(t)=\sum_{j=0}^{149-t}\gamma^j$. The base value vanishes at the horizon; the local design term is linear over the 1 m perturbation and is not multiplied by the remaining horizon. We subtract $Q_\phi(s,0,z_k;z_k)$ from the actor and design score terms. This action-independent baseline leaves their expected gradients unchanged.
