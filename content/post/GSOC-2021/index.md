---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Neural Hybrid Differential Equations and Adjoint Sensitivity Analysis"
subtitle: "GSoC 2021: The NumFOCUS organization -- Final report."
summary: ""
authors: []
tags: [GSoC 2021]
categories: []
date: 2021-08-13T21:41:45+02:00
lastmod: 2021-08-13T21:41:45+02:00
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

In this project, we have implemented state-of-the-art sensitivity tools for chaotic dynamical systems, continuous adjoint sensitivity methods for hybrid differential equations, as well as an high level API for automatic differentiation.

Possible fields of application for these tools range from model discovery with explicit dosing times in pharmacology, over accurate gradient estimates for chaotic fluid dynamics, to the control of open quantum systems. A more detailed summary is available [on the GSoC page](https://summerofcode.withgoogle.com/projects/#5357798591823872).


## Blog posts

The following blog posts describe the work in more detail throughout the GSoC period:

1) [Neural Hybrid Differential Equations](https://frankschae.github.io/post/hybridde/)
2) [Shadowing Methods for Forward and Adjoint Sensitivity Analysis of Chaotic Systems](https://frankschae.github.io/post/shadowing/)
3) [Sensitivity Analysis of Hybrid Differential Equations](https://frankschae.github.io/post/bouncing_ball/)
4) [AbstractDifferentiation.jl for AD-backend agnostic code](https://frankschae.github.io/post/abstract_differentiation/)

## Docs

Documentation with respect to the adjoint sensitivity tools will be available [on the local sensitivity analysis](https://diffeq.sciml.ai/latest/analysis/sensitivity/) and [on the control of automatic differentiation choices](http://scimlbase.sciml.ai/dev/fundamentals/Differentiation/) pages.

## Achievements

Below is a list of PRs in the various repositories in chronological order.

#### DiffEqSensitivity.jl

Merged:

* [Add additive noise downstream test for DiffEqFlux](https://github.com/SciML/DiffEqSensitivity.jl/pull/415)
* [DiscreteCallback fixes](https://github.com/SciML/DiffEqSensitivity.jl/pull/416)
* [Allow for changes of p in callbacks](https://github.com/SciML/DiffEqSensitivity.jl/pull/417)
* [Fix for using the correct uleft/pleft in continuous callback](https://github.com/SciML/DiffEqSensitivity.jl/pull/418)
* [Fix broadcasting error on steady state adjoint](https://github.com/SciML/DiffEqSensitivity.jl/pull/419)
* [Forward Least Squares Shadowing (LSS)](https://github.com/SciML/DiffEqSensitivity.jl/pull/420)
* [Adjoint-mode for the LSS method](https://github.com/SciML/DiffEqSensitivity.jl/pull/422)
* [concrete_solve dispatch for LSS methods](https://github.com/SciML/DiffEqSensitivity.jl/pull/423)
* [Non-Intrusive Least Square Shadowing (NILSS)](https://github.com/SciML/DiffEqSensitivity.jl/pull/437)
* [concrete_solve for NILSS](https://github.com/SciML/DiffEqSensitivity.jl/pull/442)
* [Remove allocation in NILSS](https://github.com/SciML/DiffEqSensitivity.jl/pull/443)
* [Handle additional callback case](https://github.com/SciML/DiffEqSensitivity.jl/pull/444)
* [State-dependent Continuous Callbacks for BacksolveAdjoint](https://github.com/SciML/DiffEqSensitivity.jl/pull/445)

Open:

#### AbstractDifferentiation.jl

Merged:

*[Fixes gradient, Jacobian, Hessian, and vjp tests](https://github.com/JuliaDiff/AbstractDifferentiation.jl/pull/2)

Open:

*[Add ForwardDiff and Zygote](https://github.com/JuliaDiff/AbstractDifferentiation.jl/pull/3)

#### OrdinaryDiffEq.jl

Merged:

*[Fix discrete reverse mode for some standard controllers](https://github.com/SciML/OrdinaryDiffEq.jl/pull/1424)

#### SteadyStateDiffEq.jl

Merged:

*[convert alg.tspan to type of prob.u0](https://github.com/SciML/SteadyStateDiffEq.jl/pull/31)

#### DiffEqNoiseProcess.jl

Merged:

*[Allow solvers to use Noise Grid with SVectors](https://github.com/SciML/DiffEqNoiseProcess.jl/pull/94)

#### StochasticDiffEq.jl

Merged:

* [Remove Ihat2 matrix from weak solvers](https://github.com/SciML/StochasticDiffEq.jl/pull/428)


#### DiffEqDocs.jl

Merged:

* [Small typo on plot page](https://github.com/SciML/DiffEqDocs.jl/pull/490)


## Future work

Besides the implementation of more shadowing methods, such as

* [NILSAS](https://arxiv.org/abs/1801.08674),
* [FD-NILSS](https://arxiv.org/abs/1711.06633), or
* [Fast linear response](https://arxiv.org/abs/2009.00595),

we are planning to

* benchmark the new adjoints,
* refine the AbstractDifferentiation.jl package and use it within DiffEqSensitivity.jl,
* add more docs, and examples.

If you have any further suggestions or comments, check out our slac/zulip channels #sciml-bridged and #diffeq-bridged or the [Julia language discourse](https://discourse.julialang.org/).

## Acknowledgement

Many thanks to my mentors [Chris Rackauckas](https://github.com/ChrisRackauckas), [Moritz Schauer](https://github.com/mschauer), [Yingbo Ma](https://github.com/YingboMa), and [Mohamed Tarek](https://github.com/mohamed82008) for their great, ongoing support before, during, and after this project. It was a great opportunity to be part of such an inspiring collaboration. I highly appreciate our quick and flexible meeting times.
I would also like to thank [Christoph Bruder](https://www.quantumtheory-bruder.physik.unibas.ch/people.html), [Julian Arnold](https://github.com/arnoldjulian), and [Martin Koppenhöfer](https://github.com/mako-git) for helpful comments on my blog posts. Special thanks to [Michael Poli](https://github.com/Zymrael) and [Stefano Massaroli](https://github.com/massastrello) for their suggestions with respect to adjoints for hybrid differential equations. Finally, thanks to the very supportive julia community and to Google's open source program for funding this experience!
