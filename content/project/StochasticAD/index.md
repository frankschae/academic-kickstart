---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Automatic Differentiation of Programs with Discrete Randomness"
summary: "We develop and implement AD algorithms for handling programs that can contain discrete randomness."
authors: [FS in collaboration with Gaurav Arya, Moritz Schauer, Ruben Seyer, Alex Lew, Mathieu Huot, Kartik Chandra, Vikash Mansinghka, Jonathan Ragan-Kelley, Chris Rackauckas]
tags: []
categories: []
date: 2024-01-15T10:49:24-05:00

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
- name: NeurIPS 2022
  url: "https://arxiv.org/abs/2210.08572"
- name: ICML 2023 Workshop
  url: "https://arxiv.org/abs/2306.07961"

url_code: "https://github.com/gaurav-arya/StochasticAD.jl"
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
Automatic differentiation (AD) has become ubiquitous throughout scientific computing and deep learning. However, AD systems have been restricted to the subset of programs that have a continuous dependence on parameters. Programs that have discrete stochastic behaviors governed by distribution parameters, such as flipping a coin with probability p of being heads, pose a challenge to these systems. In this work we develop a new AD methodology for programs with discrete randomness. We demonstrate how this method gives an unbiased and low-variance estimator.
