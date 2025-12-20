This title is short and catchy, but does not convey the full range of models covered
by this library. We could also say:

- Bayesian Generalized Additive Models for **Location, Scale, and Shape** (and beyond)
- Bayesian **Structured Additive Distributional Regression**

This library provides functionality to make the setup of generalized additive models
in [Liesel](https://github.com/liesel-devs/liesel) convenient.
It uses [ryp](https://github.com/Wainberg/ryp) to obtain basis and penalty matrices
from the R package [mgcv](https://cran.r-project.org/web/packages/mgcv/index.html),
nd relies on [formulaic](https://github.com/matthewwardrop/formulaic) to parse Wilkinson
formulas, known to many from the formula syntax in R.

Learn more in the Liesel paper:

- Riebl, H., Wiemann, P. F. V., & Kneib, T. (2023).
  Liesel: A probabilistic programming framework for developing semi-parametric
  regression models and custom Bayesian inference algorithms (No. arXiv:2209.10975).
  arXiv. <http://arxiv.org/abs/2209.10975>
