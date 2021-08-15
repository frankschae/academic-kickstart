---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "High weak order solvers and adjoint sensitivity analysis for stochastic differential equations"
subtitle: "GSoC 2020: The Julia Language -- Final report."
summary: ""
authors: []
tags: [GSoC 2020]
categories: []
date: 2020-08-26T22:59:49+02:00
lastmod: 2020-08-26T22:59:49+02:00
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

## Project summary

In this project, we have implemented new promising tools within the [SciML](https://sciml.ai) organization which are relevant for tasks such as optimal control or parameter estimation for stochastic differential equations.
The high weak order solvers will allow for massive performance advantages for fitting expectations of equations.
Instead of automatic differentiation (AD) through the operations of an SDE solver, which scales poorly in memory, one can now use efficient stochastic adjoint sensitivity methods.

## Blog posts

The following posts describe the work during the entire period in more detail:

1) [GSoC 2020: High weak order SDE solvers and their utility in neural SDEs](https://frankschae.github.io/post/gsoc2020-high-weak-order-solvers-sde-adjoints/)
2) [High weak order SDE solvers](https://frankschae.github.io/post/high-weak/)

## Docs

The documentation of the solvers is available [here](https://diffeq.sciml.ai/latest/solvers/sde_solve/#High-Weak-Order-Methods).
Docs with respect to the adjoint sensitivity tools will be available [here](https://diffeq.sciml.ai/latest/analysis/sensitivity/).


## Achievements

Please find below a list of the PRs carried out during GSoC in the different repositories in chronological order within the SciML ecosystem.

#### StochasticDiffEq.jl

Merged:

* [Inplace version of DRI1 scheme](https://github.com/SciML/StochasticDiffEq.jl/pull/285)
* [RI1 method](https://github.com/SciML/StochasticDiffEq.jl/pull/289)
* [tstop fixes for reverse time propagation](https://github.com/SciML/StochasticDiffEq.jl/pull/305)
* [Fixes for EulerHeun scheme](https://github.com/SciML/StochasticDiffEq.jl/pull/317)
* [Speed up the tests for the DRI1 and RI1 schemes](https://github.com/SciML/StochasticDiffEq.jl/pull/327)
* [RI3, RI5, RI6, RDI2WM, RDI3WM, and RDI4WM schemes](https://github.com/SciML/StochasticDiffEq.jl/pull/328)
* [RDI1WM scheme](https://github.com/SciML/StochasticDiffEq.jl/pull/329)
* [Adaptive version of the stochastic Runge-Kutta schemes by embedding](https://github.com/SciML/StochasticDiffEq.jl/pull/332)
* [RS1 and RS2 schemes](https://github.com/SciML/StochasticDiffEq.jl/pull/333)
* [PL1WM scheme](https://github.com/SciML/StochasticDiffEq.jl/pull/334)
* [NON scheme](https://github.com/SciML/StochasticDiffEq.jl/pull/337)
* [Citations for weak methods](https://github.com/SciML/StochasticDiffEq.jl/pull/338)
* [Stochastic improved and modified Euler methods](https://github.com/SciML/StochasticDiffEq.jl/pull/342)
* [COM scheme](https://github.com/SciML/StochasticDiffEq.jl/pull/343)
* [Computationally more efficient NON variant (NON2)](https://github.com/SciML/StochasticDiffEq.jl/pull/348)

Open:

* [Static array tests](https://github.com/SciML/StochasticDiffEq.jl/pull/288)
* [Levy area for non-commutative noise processes](https://github.com/SciML/StochasticDiffEq.jl/pull/347)

#### DiffEqSensitivity.jl

Merged:

* [Adjoint sensitivities for steady states](https://github.com/SciML/DiffEqSensitivity.jl/pull/235)
* [Concrete_solve dispatch for steady state problem](https://github.com/SciML/DiffEqSensitivity.jl/pull/237)
* [BacksolveAdjoint for SDEs](https://github.com/SciML/DiffEqSensitivity.jl/pull/242)
* [GPU savety for SDE BacksolveAdjoint](https://github.com/SciML/DiffEqSensitivity.jl/pull/256)
* [Tests for concrete solve with respect to SDEs](https://github.com/SciML/DiffEqSensitivity.jl/pull/258)
* [Alternative differentiation choices (vjps) for noise Jacobian](https://github.com/SciML/DiffEqSensitivity.jl/pull/260)
* [Fixes and tests for inplace formulation of BacksolveAdjoint](https://github.com/SciML/DiffEqSensitivity.jl/pull/265)
* [Efficient SDE BacksolveAdjoint for scalar noise](https://github.com/SciML/DiffEqSensitivity.jl/pull/268)
* [Generalization of the SDE Adjoint for non-diagonal noise processes and diagonal noise processes with mixing terms](https://github.com/SciML/DiffEqSensitivity.jl/pull/275)
* [InterpolatingAdjoint for SDEs](https://github.com/SciML/DiffEqSensitivity.jl/pull/295)
* [Citations for backsolve, steadystate and interpolation adjoint](https://github.com/SciML/DiffEqSensitivity.jl/pull/298)
* [Allow for more general noise processes: replace NoiseGrid by NoiseWrapper](https://github.com/SciML/DiffEqSensitivity.jl/pull/299)
* [Checkpointing fix for BacksolveAdjoint in case of ODEs and SDEs](https://github.com/SciML/DiffEqSensitivity.jl/pull/303)
* [Cheaper non-diagonal noise tests](https://github.com/SciML/DiffEqSensitivity.jl/pull/305)

Open:

* [Support adjoints for SDEs written in the Ito sense](https://github.com/SciML/DiffEqSensitivity.jl/pull/317)

#### DiffEqNoiseProcess.jl

Merged:

* [Multi-dimensional Brownian motion tests](https://github.com/SciML/DiffEqNoiseProcess.jl/pull/48)
* [Bug fix for inplace form of NoiseGrid](https://github.com/SciML/DiffEqNoiseProcess.jl/pull/49)
* [Reversible NoiseWrapper](https://github.com/SciML/DiffEqNoiseProcess.jl/pull/51)
* [Relax the size constraints of the available noise processes](https://github.com/SciML/DiffEqNoiseProcess.jl/pull/53)
* [Generalization of the real-valued white noise process function](https://github.com/SciML/DiffEqNoiseProcess.jl/pull/54)
* [Fix of an extraction issue with NoiseGrid](https://github.com/SciML/DiffEqNoiseProcess.jl/pull/55)
* [Allow NoiseWrapper to start at user-specified time points for interpolating parts of a trajectory](https://github.com/SciML/DiffEqNoiseProcess.jl/pull/56)
* [Extraction and endpoint fixes on NoiseWrapper](https://github.com/SciML/DiffEqNoiseProcess.jl/pull/56)
* [Reversal of SDEs written in the Ito sense](https://github.com/SciML/DiffEqNoiseProcess.jl/pull/62)


#### DiffEqGPU.jl

Merged:

* [Memory efficient reduction function for ensemble problems](https://github.com/SciML/DiffEqGPU.jl/pull/59)


#### ModelingToolkit.jl

Merged:

* [modelingtoolkitize for SDESystem with conversion function between Ito and Stratonovich sense](https://github.com/SciML/DiffEqSensitivity.jl/pull/317)


#### DiffEqDevTools.jl

Merged:

* [NoiseWrapper alternative for analyticless convergence tests of SDE solvers](https://github.com/SciML/DiffEqDevTools.jl/pull/62)
* [test_convergence() dispatch for ensemble simulations](https://github.com/SciML/DiffEqDevTools.jl/pull/74)
* [Work precision set for ensemble problems](https://github.com/SciML/DiffEqDevTools.jl/pull/75)


#### DiffEqBase.jl

Merged:

* [Fix concrete_solve tests](https://github.com/SciML/DiffEqBase.jl/pull/503)


## Future work

There is still a lot that we'd like to do, e.g.,

* Writing up more docs and examples
* Implementing drift-implicit weak stochastic Runge-Kutta solvers
* Finishing the SDE adjoints for the Ito sense
* Implementing a virtual Brownian tree to store the noise processes in O(1) memory
* Setting up an OptimalControl library that allows for easy usage of the new tools within a symbolic interface
* Benchmarking of the new solvers and adjoints

Contributions, suggestions & comments are always welcome! You might like to join our slac channels #diffeq-bridged and #neuralsde to get in touch.

## Acknowledgement

I would like to thank my mentors [Chris Rackauckas](https://github.com/ChrisRackauckas), [Moritz Schauer](https://github.com/mschauer), and  [Yingbo Ma](https://github.com/YingboMa) for their amazing support during this project. It was a great opportunity to work in such an inspiring collaboration and I highly appreciate their detailed feedback.
I would also like to thank [Christoph Bruder](https://www.quantumtheory-bruder.physik.unibas.ch/people.html), Niels Lörch, Martin Koppenhöfer, and Michal Kloc for helpful comments on my blog posts.
Many thanks to the very supportive julia community and to Google's open source program for funding this experience!
