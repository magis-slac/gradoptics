.. diffoptics documentation master file, created by
   sphinx-quickstart on Tue May 24 15:47:39 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Differentiable Optics via Ray Tracing
=====================================
`gradoptics <https://github.com/magis-slac/gradoptics>`__ is a ray tracing based optical simulator built using PyTorch [1] to enable automatic differentiation.

The API is designed similar to rendering softwares, and has been heavily inspired by Physically Based Rendering (Pharr, Jakob, Humphreys) [2].

Installation
------------

.. code-block:: console

	pip install gradoptics

Work in progress
----------------

- Currently, some optical element normals are aligned with the optical axis -> more general orientations in progress
- Currently, monochromatic -> no chromatic aberrations

Project History
---------------

This project was started in 2020 by Michael Kagan and Maxime Vandegar at SLAC National Accelerator Laboratory.


Feedback and Contributions
--------------------------

Please use issues on GitHub for reporting bugs and suggesting features (including better documentation).

We appreciate all contributions. In general, we recommend using pull requests to make changes to gradoptics.

Support
-------

`gradoptics <https://github.com/magis-slac/gradoptics>`__ was developed in the context of the MAGIS-100 experiment

References
----------

[1] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al. PyTorch: An imperative style, high-performance deep learning library. In NeurIPS, 2019.

[2] Matt Pharr, Wenzel Jakob, and Greg Humphreys. 2016. Physically Based Rendering: From Theory to Implementation (3rd ed.). Morgan Kaufmann Publishers Inc.


.. toctree::
   :hidden:
   :maxdepth: 2


   tutorial
   citations
   optical_elements
   distributions
   light_sources
   ray_tracing
   transforms
   inference
   integrator
   