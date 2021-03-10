---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Control of (Stochastic) Quantum Dynamics with Differentiable Programming"
summary: "Quantum control based on parametrized controllers trained with gradient information computed by (adjoint) sensitivity methods."
authors: [FS in collaboration with Pavel Sekatski, Martin Koppenhöfer, Niels Lörch, Christoph Bruder, and Michal Kloc]
tags: [SciML, differentiable programming, NODE, NSDE, adjoint sensitivity methods, automatic differentiation, quantum control]
categories: []
date: 2021-03-10T00:20:29+01:00

# Optional external URL for project (replaces project detail page).
external_link: ""

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: Smart
  preview_only: false

# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
# links:
# - name: Follow
#   url: https://twitter.com
#   icon_pack: fab
#   icon: twitter

url_code: "https://github.com/frankschae/Control-of-Stochastic-Quantum-Dynamics-with-Differentiable-Programming"
url_pdf: ""
url_slides: ""
url_video: ""

links:
- name: SDE control paper
  url: "https://arxiv.org/abs/2101.01190"
- name: ODE control paper
  url: "https://iopscience.iop.org/article/10.1088/2632-2153/ab9802"
- name: CMD2020GEFES talk
  url: "https://www.youtube.com/watch?v=v8mJVGVdkNQ&list=PLWIVj90xdDE-2eeyFuiooxWcF8kw323Iv&index=3&t=0s"


# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---

Conceptually, it is straightforward to determine the time evolution of a quantum system for a fixed initial state given its (time-dependent) Hamiltonian or Lindbladian. Depending on the physical context, the dynamics is described by an ordinary or stochastic differential equation. In quantum state control, which is of paramount importance for quantum computation, we aim at solving the inverse problem. That is, starting from a distribution of initial states, we seek protocols that allow us to reach a desired target state by optimization of free parameters of the differential equation (control drives) in a certain time interval. To solve this control problem, we implement the system dynamics as part of a fully differentiable program and use a loss function that quantifies the distance from the target state. Specifically, we employ a neural network that maps an observation of the state of the qubit to a control drive defined via the differential equation for each time interval. To implement efficient training, we backpropagate the gradient information from the loss function through the SDE solver using adjoint sensitivity methods. Such a procedure should ultimately combine powerful tools from machine learning and scientific computation.
