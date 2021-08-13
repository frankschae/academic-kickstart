---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Sensitivity Analysis of Hybrid Differential Equations"
subtitle: "GSoC 2021 -- third blog post"
summary: ""
authors: [Frank Schäfer and Moritz Schauer]
tags: [GSoC 2021, Hybrid differential equations, Adjoint sensitivity methods, Event handling]
categories: []
date: 2021-07-16T13:24:04+02:00
lastmod: 2021-07-16T13:24:04+02:00
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

In this post, we discuss sensitivity analysis of hybrid differential equations[^1] and highlight differences between explicit[^2] and implicit discontinuities[^3] [^4]. As a paradigmatic example, we consider a bouncing ball described by the ODE

$$
\begin{aligned}
   \text{d}z(t) &= v(t) \text{d}t, \\\\
   \text{d}v(t) &= -g  \text{d}t
\end{aligned}   
$$

with initial condition

$$
\begin{aligned}
   z(t=0) &= z_0 = 5, \\\\
   v(t=0) &= v_0 = -0.1.
\end{aligned}  
$$

The initial condition contains the initial height $z_0$ and initial velocity $v_0$ of the ball. We have two important parameters in this system. First, there is the gravitational constant $g=10$ modeling the acceleration of the ball due to an approximately constant gravitational field. Second, we include a dissipation factor $\gamma=0.8$ ([coefficient of restitution](https://en.wikipedia.org/wiki/Coefficient_of_restitution)) that accounts for a non-perfect elastic bounce on the ground.

We can straightforwardly integrate the ODE analytically

$$
\begin{aligned}
z(t) &= z_0 + v_0 t - \frac{g}{2} t^2, \\\\
v(t) &= v_0 - g  t
\end{aligned}  
$$

or numerically using the OrdinaryDiffEq package from the [SciML](https://sciml.ai/) ecosystem.

```julia
using ForwardDiff, Zygote, OrdinaryDiffEq, DiffEqSensitivity
using Plots, LaTeXStrings

##
### simulate forward
function f(du,u,p,t)
  du[1] = u[2]
  du[2] = -p[1]
end

z0 = 5.0
v0 = -0.1
t0 = 0.0
tend = 1.9
g = 10
γ = 0.8

u0 = [z0,v0]
tspan = (t0,tend)
p = [g, γ]
prob = ODEProblem(f,u0,tspan,p)

# plot forward trajectory
sol = solve(prob,Tsit5(),saveat=0.1)
pl = plot(sol, label = ["z(t)" "v(t)"], labelfontsize=20, legendfontsize=20, lw = 2, xlabel = "t", legend=:bottomleft)
hline!(pl, [0.0], label=false, color="black")
savefig(pl,"BB_forward_no_bounce.png")
```

{{< figure library="true" src="BB_forward_no_bounce.png" title="" lightbox="true" >}}

## Forward simulation with events

At around $t^\star \approx 1$, the ball hits the ground $z^\star(t^\star) = 0$, and is inelastically reflected while dissipating a fraction of its energy. This discontinuity can be modeled by re-initializing the ODE with new initial conditions

$$
\begin{aligned}  
z_+&= \lim_{t \rightarrow {t^\star}^+} z(t) =  \lim_{t \rightarrow {t^\star}^-}  z(t) = z_- ,\\\\
v_+&= \lim_{t \rightarrow {t^\star}^+} v(t) =  -\gamma \lim_{t \rightarrow {t^\star}^-}  v(t) =  -\gamma v_-
\end{aligned}  
$$

on the right-hand side of the event. Given our analytical solution for the state as a function of time, we can easily compute the event time $t^\star$ as

$$
t^\star = \frac{v_0 + \sqrt{v_0^2 + 2 g z_0}}{g}.
$$

### Explicit events

We can define the bounce of the ball as an explicit event by inserting the values of the initial condition and the parameters into $t^\star$. We obtain

$$
t^\star = 0.99005.
$$

The full trajectory $z_{\rm exp}(t)$ is determined by

$$
z_{\rm exp}(t) = \begin{cases}
  z^{1}_{\rm exp}(t) &=z_0 + v_0 t - \frac{g}{2} t^2 &, \forall t \leq t^\star, \\\\
  z^{2}_{\rm exp}(t) &=-0.4901 g - 0.5 g (-0.99005 + t)^2 +\\\\
  &+0.99005 v_0 + z_0 - (-0.99005 + t) (-0.99005 g + v_0)\gamma  &, \forall t > t^\star,
\end{cases}
$$

where we used:

$$
\begin{aligned}  
z_{-, \rm exp}&= z_0 + 0.99005 v_0 -0.4901 g,\\\\
v_{-, \rm exp}&= (v_0 - 0.99005 g) .
\end{aligned}  
$$

and

$$
\begin{aligned}  
z_{+, \rm exp}&= z_0 + 0.99005 v_0 -0.4901 g, \\\\
v_{+, \rm exp}&= -(v_0 - 0.99005 g) \gamma .
\end{aligned}  
$$

Numerically, we use a `DiscreteCallback` in this case to simulate the system

```julia
# DiscreteCallback (explicit event)
tstar = (v0 + sqrt(v0^2+2*z0*g))/g
condition1(u,t,integrator) = (t == tstar)
affect!(integrator) = integrator.u[2] = -integrator.p[2]*integrator.u[2]
cb1 = DiscreteCallback(condition1,affect!,save_positions=(true,true))

sol1 = solve(prob,Tsit5(),callback=cb1, saveat=0.1, tstops=[tstar])
```

Evidently, by choosing an explicit definition of the event, the impact time is fixed. Thus, if we perturb the initial conditions or the parameters, the event location remains at $t^\star = 0.99005$ while it should actually change (for a fixed ground).


### Implicit events

The physically more meaningful description of a bouncing ball is therefore given by an implicit description of the event in form of a condition (event function)

$$
 g(z,v,p,t) = z(t),
$$

where an event occurs if $g(z^\star,v^\star,p,t^\star) = 0$. Note that we have already used this condition to define our impact time $t^\star$ of the explicit event.

As in the previous case, we can analytically compute the full trajectory of the ball. At the event time, we have

\begin{aligned}  
z_{-, \rm imp}&= 0, \\\\
v_{-, \rm imp}&= - \sqrt{v_0^2 + 2 g z_0}
\end{aligned}

for the left and

\begin{aligned}  
z_{+, \rm imp}&= 0 \\\\
v_{+, \rm imp}&= \gamma \sqrt{v_0^2 + 2 g z_0}.
\end{aligned}

for the right limit. Thus, the full trajectory $z_{\rm imp}(t)$ is given by


$$
z_{\rm imp}(t) = \begin{cases}
  z^{1}_{\rm imp}(t) &=z_0 + v_0 t - \frac{g}{2} t^2 &, \forall t \leq t^\star ,\\\\
  z^{2}_{\rm imp}(t) &= -\frac{(-g t + v_0 + \sqrt{v_0^2 + 2 g z_0})}{2 g} \times, \\\\
   &\times (-g t + v_0 + \sqrt{v_0^2 + 2 g z_0} (1 + 2 \gamma)) &, \forall t > t^\star.
\end{cases}
$$

Numerically, we use a `ContinuousCallback` in this case.

```julia
# ContinuousCallback (implicit event)
condition2(u,t,integrator) = u[1] # Event when condition(u,t,integrator) == 0
cb2 = ContinuousCallback(condition2,affect!,save_positions=(true,true))
sol2 = solve(prob,Tsit5(),callback=cb2,saveat=0.1)
```

We can verify that both callbacks lead to the same forward time evolution (for fixed initial conditions and parameters).
```julia
# plot forward trajectory
pl1 = plot(sol1, label = ["z(t)" "v(t)"], title="explicit event", labelfontsize=20, legendfontsize=20, lw = 2, xlabel = "t", legend=:bottomright)
pl2 = plot(sol2, label = ["z(t)" "v(t)"], title="implicit event", labelfontsize=20, legendfontsize=20, lw = 2, xlabel = "t", legend=:bottomright)
hline!(pl1, [0.0], label=false, color="black")
hline!(pl2, [0.0], label=false, color="black")
pl = plot(pl1,pl2)
savefig(pl,"BB_forward.png")
```
{{< figure library="true" src="BB_forward.png" title="" lightbox="true" >}}

In addition, the implicitly defined impact time via the `ContinuousCallback` also shifts the impact time when changing the initial conditions or the parameters
```julia
# animate forward trajectory
sol3 = solve(remake(prob,u0=[u0[1]+0.5,u0[2]]),Tsit5(),callback=cb2,saveat=0.01)

plt2 = plot(sol2, label = false, labelfontsize=20, legendfontsize=20, lw = 1, xlabel = "t", legend=:bottomright, color="black", xlims=(t0,tend))
hline!(plt2, [0.0], label=false, color="black")
plot!(plt2, sol3, tspan=(t0,tend), color=[1 2], label = ["z(t)" "v(t)"], labelfontsize=20, legendfontsize=20, lw = 2, xlabel = "t", legend=:bottomright, denseplot=true, xlims=(t0,tend), ylims=(-11,9))
# scatter!(plt2, [t2,t2], sol3(t2), color=[1, 2], label=false)

list_plots = []
for t in sol3.t
  tstart = 0.0

  plt1 = plot(sol2, label = false, labelfontsize=20, legendfontsize=20, lw = 1, xlabel = "t", legend=:bottomright, color="black")
  hline!(plt1, [0.0], label=false, color="black")
  plot!(plt1, sol3, tspan=(t0,t), color=[1 2], label = ["z(t)" "v(t)"], labelfontsize=20, legendfontsize=20, lw = 2, xlabel = "t", legend=:bottomright, denseplot=true, xlims=(t0,tend), ylims=(-11,9))
  scatter!(plt1,[t,t], sol3(t), color=[1, 2], label=false)
  plt = plot(plt1,plt2)
  push!(list_plots, plt)
end

plot(list_plots[100])

anim = animate(list_plots,every=1)
```
{{< figure library="true" src="BB.gif" title="" lightbox="true" >}}

The original curve is shown in black in the figure above. In other words, the event time $t^\star=t^\star(p,z_0,v_0,t_0)$ is a function of the parameters and initial conditions, and is implicitly defined by the event condition. Therefore, the sensitivity of the event time with respect to parameters $\frac{\text{d}t^\star}{\text{d}p}$ or initial conditions $\frac{\text{d}t^\star}{\text{d}z_0}$, $\frac{\text{d}t^\star}{\text{d}v_0}$ must be taken into account.

## Sensitivity analysis with events

We are often interested in computing the change of a loss function with respect to changes of the parameters or initial condition. For this purpose, let us first consider the mean square error loss function

$$
L(z,y) = \sum_i(z(t_i) - y_i)^2
$$

with respect to target values $y_i$ at time points $t_i$ incident before, after, or at the event time. Let $\alpha$ denote any of the inputs $(z_0,v_0,g,\gamma)$. The sensitivity with respect to $\alpha$ is then given by the chain rule:

$$
\frac{\text{d}L}{\text{d} \alpha} =  2\sum_i (z(t_i) - y_i) \frac{\text{d}z(t_i)}{\text{d} \alpha}.
$$

For the bouncing ball, we can easily compute those sensitivities by inserting our results for $z_{\rm imp}(t_i)$ and $z_{\rm exp}(t_i)$. One can verify that the sensitivities are different in the two cases, as expected.


However, in most systems, we won't be able to solve analytically a differential equation

$$
\text{d}x(t) = f(x,p,t) \text{d}t
$$

with initial condition $x_0=x(t_0)$. Instead, we have to numerically solve for the state $x(t)$. Regarding the computation of the sensitivities, we may then choose one of the [available algorithms](https://diffeq.sciml.ai/stable/analysis/sensitivity/) for the given differential equation. Currently, `BacksolveAdjoint()`, `InterpolatingAdjoint()`, `QuadratureAdjoint()`, `ReverseDiffAdjoint()`, `TrackerAdjoint()`, and `ForwardDiffAdjoint()` are compatible events in ordinary differential equations. We write the loss function in the following as a function of time, state, and parameters

$$
\begin{aligned}
L = L(t,x,p).
\end{aligned}
$$

In the following, let us focus on the `BacksolveAdjoint()` algorithm, which computes the sensitivities

$$
\begin{aligned}
\frac{\text{d}L}{\text{d}x(t_{0})} &= \lambda(t_{0}),\\\\
\frac{\text{d}L}{\text{d}p} &= \lambda_{p}(t_{0}),
\end{aligned}
$$

with respect to the initial state and the parameters, by solving an ODE for $\lambda(t)$ in reverse time from $t_N$ to $t_0$

$$
\begin{aligned}
\frac{\text{d}\lambda(t)}{\text{d}t} &= -\lambda(t)^\dagger \frac{\text{d} f(\rightarrow x(t), p, t)}{\text{d} x(t)} - \frac{\text{d} L(t, \rightarrow x(t), p)}{\text{d} x(t)}^\dagger \delta(t-t_i), \\\\
\frac{\text{d}\lambda_{p}(t)}{\text{d}t} &= -\lambda(t)^\dagger \frac{\text{d} f(x(t), \rightarrow p, t)}{\text{d} p},
\end{aligned}
$$

with initial conditions:

$$
\begin{aligned}
\lambda(t_{N})&= 0, \\\\
\lambda_{p}(t_{N}) &= 0.
\end{aligned}
$$

The arrows indicate the variable with respect to which we differentiate. Note that computing the vector-Jacobian products (vjp) in the adjoint ODE requires the value of $x(t)$ along its trajectory. In `BacksolveAdjoint()`, we recompute $x(t)$--together with the adjoint variables--backwards in time starting with its final value $x(t_N)$. A derivation of the ODE adjoint is given in [Chris' MIT 18.337 lecture notes](https://mitmath.github.io/18337/lecture11/adjoints).

### Explicit events

To make `BacksolveAdjoint()` compatible with explicit events[^2], we have to store the event times $t^\star_j$ as well as the state $x({t_j^\star}^-)$ and parameters $p=p({t_j^\star}^-)$ (if they are changed) at the left limit of $t^\star_j$. We then solve the adjoint ODE backwards in time between the events. As soon as we reach an event from the right limit ${t_j^\star}^+$, we update the augmented state according to

$$
\begin{aligned}
\lambda({t_j^\star}^-) &= \lambda({t_j^\star}^+)^\dagger \frac{\text{d} a(\rightarrow x({t_j^\star}^-), p({t_j^\star}^-), {t_j^\star}^-)}{\text{d} x({t_j^\star}^-)} \\\\
\lambda_p({t_j^\star}^-) &= \lambda_p({t_j^\star}^+) -  \lambda({t_j^\star}^+)^\dagger \frac{\text{d} a(x({t_j^\star}^-), \rightarrow p({t_j^\star}^-), {t_j^\star}^-)}{\text{d} p({t_j^\star}^-)}
\end{aligned}
$$

where $a$ is the affect function applied at the discontinuity. That is, to lift the adjoint from the right to the left limit, we compute a vjp with the adjoint $\lambda({t_j^\star}^+)$ at the right limit and the Jacobian of the affect function evaluated on the left limit.

In particular, we apply a loss function callback before and after this update if the state was saved in the forward evolution and entered directly into the loss function.

### Implicit events

#### special case: event as termination condition

Define $u(t) = (t, x(t))$. Let us first re-derive the case, where the implicit event terminates the ODE and where we have a loss function acting on $t^\star_1$, $x(t^\star_1)$, and $p$, as considered by Ricky T. Q. Chen, Brandon Amos, and Maximilian Nickel in their ICLR 2021 paper[^4]. We are interested in

$$
\frac{\text{d}L(u(t^\star_1(\rightarrow p), \rightarrow p), \rightarrow p)}{\text{d}p} = \frac{\text{d}L(t^\star_1({\color{black}\rightarrow} p), \text{solve}(t_0, x_0, t^\star_1({\color{black}\rightarrow}p), \rightarrow p),\rightarrow p)}{\text{d}p},
$$

which indicates that changing $p$ changes both $t^\star_1$ as well as $x^\star_1$ in $t^\star_1$.

In a first step, we need to compute the sensitivity of $t^\star_1(p)$ with respect to $p$ and $x_0$ based on the event condition $F(t, p) = g(u(t, p)) = 0$.  We can apply the [implicit function theorem](https://www.uni-siegen.de/fb6/analysis/overhagen/vorlesungsbeschreibungen/skripte/analysis3_1.pdf) which yields:

$$
\begin{aligned}
\frac{\text{d}t^\star_1(p)}{\text{d}p} &= - \left(\frac{\text{d}g(\rightarrow t^\star_1, \text{solve}(t_0, x_0, \rightarrow t^\star_1, p))}{\text{d}t^\star_1}\right)^{-1} \frac{\text{d}g(t^\star_1, \text{solve}(t_0, x_0, t^\star_1, \rightarrow p))}{\text{d}p} .\\\\
\end{aligned}
$$

The total derivative[^5] inside the bracket is defined as:
$$
\begin{aligned}
\frac{\text{d}g}{\text{d}t^\star_1} \stackrel{\text{def}}{=} \frac{\text{d}g(\rightarrow t^\star_1, \text{solve}(t_0, x_0, \rightarrow t^\star_1, p))}{\text{d}t^\star_1} &= \frac{\text{d}g(\rightarrow t^\star_1, \text{solve}(t_0, x_0, t^\star_1, p))}{\text{d}t^\star_1} + \frac{\text{d}g(t^\star_1, \text{solve}(t_0, x_0, \rightarrow t^\star_1, p))}{\text{d}t^\star_1}\\\\
\end{aligned}
$$

Since

$$
\frac{\text{d}(\text{solve}(t_0, x_0, \rightarrow t^\star_1, p))}{\text{d}t^\star_1} = f(x^\star, p^\star, t^\star_1)
$$

by definition of the ODE, we can write

$$
\begin{aligned}
\frac{\text{d}g(t^\star_1, \text{solve}(t_0, x_0, \rightarrow t^\star_1, p))}{\text{d}t^\star_1} = \frac{\text{d}g(t^\star_1, \text{solve}(t_0, x_0, \rightarrow t^\star_1, p))}{\text{d} u^\star(t^\star_1)}  f(x^\star, p^\star, t^\star_1).
\end{aligned}
$$

Furthermore, we have
$$
\begin{aligned}
\frac{\text{d}g(t^\star_1, \text{solve}(t_0, x_0, t^\star_1, \rightarrow p))}{\text{d}p} = \frac{\text{d}g(t^\star_1, \text{solve}(t_0, x_0, t^\star_1,\rightarrow p))}{\text{d} u^\star(t^\star_1)}^\dagger  \frac{\text{d}\text{ solve}(t_0, x_0, t^\star_1,\rightarrow p))}{\text{d}p}
\end{aligned}
$$
for the second term of $\frac{\text{d}t^\star_1(p)}{\text{d}p}$. We define

$$
\frac{\text{d}g}{\text{d}u^\star_1} \stackrel{\text{def}}{=} \frac{\text{d}g(t^\star_1, \text{solve}(t_0, x_0, t^\star_1,\rightarrow p))}{\text{d} u^\star(t^\star_1)}
$$

We can now write the gradient as:

$$
\begin{aligned}
\frac{\text{d}L(t^\star_1({\color{black}\rightarrow} p), \text{solve}(t_0, x_0, t^\star_1({\color{black}\rightarrow}p), \rightarrow p),\rightarrow p)}{\text{d}p} &= \frac{\text{d}L(t^\star_1(p), \text{solve}(t_0, x_0, t^\star_1(p),  p), \rightarrow p)}{\text{d}p} \\\\
+& \frac{\text{d}L(t^\star_1(p), \text{solve}(t_0, x_0, t^\star_1(p),  \rightarrow p), p)}{\text{d}p} \\\\
+& \frac{\text{d}L(\rightarrow t^\star_1(p), \text{solve}(t_0, x_0, \rightarrow t^\star_1(p),  p), p)}{\text{d}t^\star_1} \frac{\text{d} t^\star_1(p)}{\text{d}p},
\end{aligned}
$$

which, after insertion of our results above, can be casted into the form:

$$
\begin{aligned}
\frac{\text{d}L(t^\star_1({\color{black}\rightarrow} p), \text{solve}(t_0, x_0, t^\star_1({\color{black}\rightarrow}p), \rightarrow p),\rightarrow p)}{\text{d}p} &= v^\dagger \frac{\text{d}\text{ solve}(t_0, x_0, t^\star_1(p), \rightarrow p)}{\text{d}p} \\\\
&+ \frac{\text{d}L(t^\star_1(p), \text{solve}(t_0, x_0, t^\star_1(p), p), \rightarrow p)}{\text{d}p},
\end{aligned}
$$

with

$$
\begin{aligned}
v &= \xi \left(-\frac{\text{d}g}{\text{d}t^\star_1}\right)^{-1} \frac{\text{d}g}{\text{d}u^\star_1} + \frac{\text{d}L(t^\star_1(p), \text{solve}(t_0, x_0, t^\star_1(p),  p), p)}{\text{d} u^\star(t^\star_1)},
\end{aligned}
$$

where we introduced the scalar pre-factor

$$
\begin{aligned}
\xi = \left( \frac{\text{d}L(\rightarrow t^\star_1(p), \text{solve}(t_0, x_0, t^\star_1(p),  p), p)}{\text{d}t^\star_1} +  \frac{\text{d}L(t^\star_1(p), \text{solve}(t_0, x_0, t^\star_1(p),  p), p)}{\text{d} u^\star(t^\star_1)}^\dagger f(x^\star, p^\star, t^\star_1)\right).
\end{aligned}
$$

This means that if we terminate the ODE integration by an implicit event, we compute the sensitivities as follows (for simplicity we drop terms due to an explicit dependence of the loss function on the parameters or time):

1. use an ODE solver to solve forward until the event is triggered
$$
u_i = \text{solve}(t_0, x_0, t^\star_1(p),  p).
$$
$u_i(t_i)=(t_i,x_i)$ are the stored values which enter the loss function.
2. compute the loss function gradient with respect to the state at $t^\star_1$.
$$
\lambda_-^\text{0} = \frac{\text{d}L(t^\star_1(p), \rightarrow \text{solve}(t_0, x_0, t^\star_1(p),  p), p)}{\text{d} u^\star(t^\star_1)}.
$$
3. (instead of using the `BacksolveAdjoint()` algorithm with $\lambda_-^\text{0}$ directly,) use the corrected version containing the dependence on the event time. For this, compute  $\frac{\text{d}g}{\text{d}t^\star_1}, \frac{\text{d}g}{\text{d}u^\star_1}$, and $f(x^\star, p, t^\star_1)$.
Then, the corrected version of the adjoint is given by

$$
\lambda_- = - \left( {\lambda_-^\text{0}}^\dagger f(x^\star, p^\star, t^\star_1) \right)\left(\frac{\text{d}g}{\text{d}t^\star_1}\right)^{-1} \frac{\text{d}g}{\text{d}u^\star_1} + \lambda_-^\text{0}.
$$

$\lambda_-$ can then be used as initial condition to $\text{backsolve_adjoint}(\lambda_-, t^\star_1, x(t^\star_1), t_0)$ which backpropagates the adjoint $\lambda_-$ from $t^\star_1$ to $t_0$.


If there is an additional affect function $a$ associated with the event, i.e. a right limit, we must additionally compute

$$
\begin{aligned}
\lambda_+^\text{0} =  \frac{\text{d}L(t^\star_1(p), \rightarrow a\left(\text{solve}(t_0, x_0, t^\star_1(p),  p)\right), p)}{\text{d} u^\star(t^\star_1))}.
\end{aligned}
$$

Compute the vjp as in the case of a 'DiscreteCallback'

$$
\lambda_-^\text{0} = {\lambda_+^\text{0}}^\dagger \frac{\text{d} a(\rightarrow x({t_j^\star}^-), p({t_j^\star}^-), {t_j^\star}^-)}{\text{d} x({t_j^\star}^-)}
$$

and correct it as above

$$
\begin{aligned}
\lambda_- = - \left( {\lambda_-^\text{0}}^\dagger f(x({t_1^\star}^-), p({t_1^\star}^-), t^\star_1) \right)\left(\frac{\text{d}g}{\text{d}{t_1^\star}^-}\right)^{-1} \frac{\text{d}g}{\text{d}{u^\star_1}^-} + \lambda_-^\text{0}.
\end{aligned}
$$

If both limits contribute to the loss function, the contributions are added.

#### generalization: several events

As pointed out by Chen et al. as well as by Timo C. Wunderlich and Christian Pehle[^3], one can chain together the events and differentiate through the entire time evolution on a time interval $(t_0, t_{\text{end}})$. That is, we are generally allowed to segment the time evolution over an interval $[t_0, t]$ into one from $[t_0, s]$ and a subsequent one from $[s, t]$:

$$
\text{solve}(t_0, x_0, t, p)  = \text{solve}(s, \text{solve}(t_0, x_0, s, p), t-s, p),
$$

such that also loss function contributions are chained. Therefore, we have the following modification:


1. Segment the trajectory at the event times. Use $\text{backsolve_adjoint}(\lambda_{0}, t_\text{end}, x(t_\text{end}), t^\star_N)$ to backprogagate the loss function gradient from the final state until the right limit of the last event location.

2. In addition to the steps above, subtract a correction:

$$
\lambda_\text{c} = - \left( {\lambda_+}^\dagger f(x({t_N^\star}^+), p({t_N^\star}^+), t^\star_N) \right)\left(\frac{\text{d}g}{\text{d}{t_N^\star}^-}\right)^{-1} \frac{\text{d}g}{\text{d}{u^\star_N}^-},
$$

where $\lambda_+$ is the right-hand limit of the adjoint state before the loss gradient ($\lambda_+^\text{0}$ above) was added. Iterate over the remaining events.


## Outlook

We are still refining the adjoints in case of implicit discontinuities (`ContinuousCallbacks`). For further information, the interested reader is encouraged to track the associated issues [#383](https://github.com/SciML/DiffEqSensitivity.jl/issues/383) and [#374](https://github.com/SciML/DiffEqSensitivity.jl/issues/374), and [PR #445](https://github.com/SciML/DiffEqSensitivity.jl/pull/445) in the DiffEqSensitivity.jl package.

If you have any questions or comments, please don’t hesitate to contact me!

[^1]: Michael Poli, Stefano Massaroli, et al., arXiv preprint arXiv:2106.04165 (2021).
[^2]: Junteng Jia, Austin R. Benson, arXiv preprint arXiv:1905.10403 (2019).
[^3]: Timo C. Wunderlich and Christian Pehle, Sci. Rep. *11*, 12829 (2021).
[^4]: Ricky T. Q. Chen, Brandon Amos, Maximilian Nickel, arXiv preprint arXiv:2011.03902 (2020).
[^5]: For a function $f$ of more than one variable $y = f(t, x_1(t),x_2(t),\dots,x_N(t))$, the [total derivative](https://en.wikipedia.org/wiki/Differential_of_a_function#Differentials_in_several_variables) with respect to the independent variable $t$ is given by the sum of all partial derivatives
$$
\begin{aligned}
\frac{\text{d}y}{\text{d}t} &= \frac{\text{d}f(\rightarrow t, x_1(\rightarrow t),x_2(\rightarrow t),\dots,x_N(\rightarrow t))}{\text{d}t} \\\\
&= \frac{\text{d}f(\rightarrow t, x_1(t),x_2(t),\dots,x_N(t))}{\text{d}t} + \frac{\text{d}f(t, x_1(\rightarrow t),x_2(t),\dots,x_N(t))}{\text{d}t}\\\\
&+ \frac{\text{d}f(t, x_1(t),x_2(\rightarrow t),\dots,x_N(t))}{\text{d}t} + \dots +  \frac{\text{d}f(t, x_1(t),x_2(t),\dots,x_N(\rightarrow t))}{\text{d}t}.
\end{aligned}
$$
