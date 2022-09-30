[![Tests](https://github.com/magis-slac/gradoptics/actions/workflows/main.yml/badge.svg)](https://github.com/magis-slac/gradoptics/actions)
[![Build Status](https://travis-ci.com/magis-slac/gradoptics.svg?token=LBAvFbnCy9PEgexzsTUS&branch=main)](https://travis-ci.com/magis-slac/gradoptics)
[![Documentation Status](https://readthedocs.org/projects/gradoptics/badge/?version=latest)](https://gradoptics.readthedocs.io/en/latest/?badge=latest)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/magis-slac/gradoptics/blob/master/README.md)
![version](https://img.shields.io/badge/version-0.0.2-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Differentiable Optics via Ray Tracing
[*gradoptics*](https://github.com/magis-slac/gradoptics) is a ray tracing based optical simulator built using PyTorch [[1]](#1) to enable automatic differentiation. 

The API is designed similar to rendering softwares, and has been heavily inspired by *Physically Based Rendering* (Pharr, Jakob, Humphreys) [[2]](#2). 


## Getting Started
[Getting Started](https://github.com/magis-slac/gradoptics/blob/main/docs/tutorials/Quick-Start.ipynb)


## Installation


```commandline
pip install gradoptics
```

Then, you should be ready to go!
```python
import gradoptics as optics
```

## Work in progress
- Currently, some optical element normals are aligned with the optical axis -> more general orientations in progress
- Currently, monochromatic -> no chromatic aberrations

## Project History

This project was started in 2020 by Michael Kagan and Maxime Vandegar at SLAC National Accelerator Laboratory.

## Feedback and Contributions

Please use issues on GitHub for reporting bugs and suggesting features (including better documentation).

We appreciate all contributions. In general, we recommend using pull requests to make changes to [*gradoptics*](https://github.com/magis-slac/gradoptics).  

#### Testing

If you modify [*gradoptics*](https://github.com/magis-slac/gradoptics), please use pytest for checking your code.

```commandline
pytest tests/tests.py 
```


## Support

[*gradoptics*](https://github.com/magis-slac/gradoptics) was developed in the context of the MAGIS-100 experiment 

## References
<a id="1">[1]</a> 
A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al. PyTorch: An imperative style, high-performance deep learning library. In NeurIPS, 2019.

<a id="1">[2]</a> 
Matt Pharr, Wenzel Jakob, and Greg Humphreys. 2016. Physically Based Rendering: From Theory to Implementation (3rd ed.). Morgan Kaufmann Publishers Inc. 
