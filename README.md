[![Tests](https://github.com/magis-slac/differentiable-optics/actions/workflows/main.yml/badge.svg)](https://github.com/magis-slac/differentiable-optics/actions)
[![Build Status](https://travis-ci.com/magis-slac/differentiable-optics.svg?token=LBAvFbnCy9PEgexzsTUS&branch=main)](https://travis-ci.com/magis-slac/differentiable-optics)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/magis-slac/differentiable-optics/blob/master/README.md)
![version](https://img.shields.io/badge/version-0.0.1-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Differentiable Optics via Ray Tracing
[*diffoptics*](https://github.com/magis-slac/differentiable-optics) is a ray tracing based optical simulator built using PyTorch [[1]](#1) to enable automatic differentiation. 

The API is designed similar rendering software, and has been heavily inspired by *Physically Based Rendering* (Pharr, Jakob, Humphreys) [[2]](#2). 


## Getting Started
[Getting Started](https://github.com/magis-slac/differentiable-optics/blob/main/docs/tutorials/quickStart.ipynb)


## Installation


```commandline
cd differentiable-optics
pip install -r requirements.txt
pip install -e .
```

Then, you should be ready to go!
```python
import diffoptics as optics
```

## Work in progress
- Currently using ideal lenses (thin lens approximation) -> thick and compound lenses in progress
- Currently the sensor, lens and window normals are aligned with the optical axis -> more general orientations in progress
- Currently monochromatic -> no chromatic abberations
- Adding PSF to imaging -> in progress 

## Project History

This project was started in 2020 by Michael Kagan and Maxime Vandegar at SLAC National Accelerator Laboratory.

## Feedback and Contributions

Please use issues on GitHub for reporting bugs and suggesting features (including better documentation).

We appreciate all contributions. In general, we recommend using pull requests to make changes to [*diffoptics*](https://github.com/magis-slac/differentiable-optics).  

#### Testing

If you modify [*diffoptics*](https://github.com/magis-slac/differentiable-optics), please use pytest for checking your code.

```commandline
pytest tests/tests.py 
```


## Support

[*diffoptics*](https://github.com/magis-slac/differentiable-optics) was developed in the context of the MAGIS-100 experiment 

## References
<a id="1">[1]</a> 
A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al. PyTorch: An imperative style, high-performance deep learning library. In NeurIPS, 2019.

<a id="1">[2]</a> 
Matt Pharr, Wenzel Jakob, and Greg Humphreys. 2016. Physically Based Rendering: From Theory to Implementation (3rd ed.). Morgan Kaufmann Publishers Inc. 
