---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Performance Bounds for Quantum Control"
summary: "We compute bounds for the best attainable control performance that may serve as certificates of fundamental limitations or performance targets."
authors: [FS in collaboration with Flemming Holtorf, Julian Arnold, Chris Rackauckas, and Alan Edelman]
tags: [quantum control, convex optimization, stochastic optimal control, sum-of-squares]
categories: []
date: 2023-04-29T17:26:49-04:00

# Optional external URL for project (replaces project detail page).
external_link: ""

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
links:
- name: arXiv preprint
  url: "https://arxiv.org/abs/2304.03366"
- name: QCE
  url: "https://ieeexplore.ieee.org/abstract/document/10313626"

url_code: "" # https://github.com/frankschae/Performance-Bounds-for-Quantum-Control
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---

Control of devices at the quantum level holds enormous potential for current and future applications in the field of quantum information science. However, due to the nonlinear and stochastic nature of quantum systems under continuous observation, analytical solutions to all but the simplest quantum control problems remain unknown. In this project, we present a convex optimization framework to compute informative bounds on the best attainable control performance. Since our approach provides an under-approximator for the value function, we can use it directly to construct near-optimal heuristic controllers as demonstrated for a qubit subjected to homodyne detection and photon counting.
