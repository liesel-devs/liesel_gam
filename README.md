# Generalized Additive Models Functionality in Liesel

[![pre-commit](https://github.com/liesel-devs/liesel_gam/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/liesel-devs/liesel_gam/actions/workflows/pre-commit.yml)
[![notebooks](https://github.com/liesel-devs/liesel_gam/actions/workflows/pytest-notebooks.yml/badge.svg)](https://github.com/liesel-devs/liesel_gam/tree/main/notebooks)
[![pytest](https://github.com/liesel-devs/liesel_gam/actions/workflows/pytest.yml/badge.svg)](https://github.com/liesel-devs/liesel_gam/actions/workflows/pytest.yml)
[![pytest-cov](https://raw.githubusercontent.com/liesel-devs/liesel_gam/refs/heads/main/tests/coverage.svg)](https://github.com/liesel-devs/liesel_gam/actions/workflows/pytest.yml)

This package provides functionality to make the setup of
semiparametric generalized additive distributional regression models in [Liesel](https://github.com/liesel-devs/liesel)
convenient. It works nicely with [liesel-devs/smoothcon](https://github.com/liesel-devs/smoothcon),
which can be used to obtain basis and penalty matrices from the R package [mgcv](https://cran.r-project.org/web/packages/mgcv/index.html).

## Disclaimer

This package is experimental and under active development. That means:

- The API cannot be considered stable. If you depend on this package, pin the version.
- Testing has not been extensive as of now. Please check and verify!
- There is currently no documentation beyond this readme.

In any case, this package comes with no warranty or guarantees.

## Installation

You can install `liesel_gam` from pypi:

```bash
pip install liesel_gam
```

You can also install the development version from GitHub via pip:

```bash
pip install git+https://github.com/liesel-devs/liesel_gam.git
```

## Illustration

This is a short pseudo-code illustration without real data. For full examples, please
consider the [notebooks](https://github.com/liesel-devs/liesel_gam/blob/main/notebooks).

```python
import liesel.model as lsl
import liesel.goose as gs

import liesel_gam as gam

import jax.numpy as jnp
```

Set up the response model.

```python
loc = gam.AdditivePredictor("loc")
scale = gam.AdditivePredictor("scale", inv_link=jnp.exp) # terms will be added on the linked level

y = lsl.Var.new_obs(
    value=...,
    distribution=lsl.Dist(..., loc=loc, scale=scale),
    name="y"
)
```

Add intercept terms

```python
loc += gam.Intercept(
    value=0.0, # this is the default
    distribution=None, # this is the default
    inference=gs.MCMCSpec(gs.IWLSKernel), # supply inference information here
    name="b0"
)

scale += gam.Intercept( # this term will be applied on the log link level
    value=0.0,
    distribution=None,
    inference=gs.MCMCSpec(gs.IWLSKernel),
    name="g0"
)

```

Add a smooth term, which can be any structured additive term defined by a basis matrix
and a penalty matrix. A potentially rank-deficient multivariate normal prior will
be set up for the coefficient of this term.

```python
loc += gam.SmoothTerm(
    basis=...,
    penalty=...,
    scale=lsl.Var.new_param(..., name="tau"),
    inference=gs.MCMCSpec(gs.IWLSKernel),
    name="s(x)"
)
```

Add a linear term.

```python
loc += gam.LinearTerm(
    x=..., # 1d-array or 2d-array are both allowed
    distribution=lsl.Dist(...),
    inference=gs.MCMCSpec(gs.IWLSKernel),
    name="x"
)
```

Get a Liesel EngineBuilder instance to set up MCMC sampling.

```python
model = lsl.Model([y])
eb = gs.LieselMCMC(model).get_engine_builder() # get your engine builder instance
```

## Contents

```python
import liesel.model as lsl
import liesel.goose as gs

import liesel_gam as gam
```

This package provides the following classes and functions:

- `gam.AdditivePredictor`: A `lsl.Var` object that provides a convenient way to define an additive predictor.
- `gam.SmoothTerm`: A `lsl.Var` object that provides a convenient way to set up a structured additive term with a singular multivariate normal prior, given a basis matrix, a penalty matrix, and a `lsl.Var` representing the prior scale parameter.
  - The alternative constructor `gam.SmoothTerm.new_ig` can be used to quickly set up a term with an inverse gamma prior on the prior variance parameter. This variance parameter will be initialized with a suitable Gibbs kernel.
- `gam.LinearTerm`: A `lsl.Var` object that provides a convenient way to set up a linear term.
- `gam.Intercept`: A `lsl.Var` parameter object that represents an intercept.
- `gam.Basis`: An observed `lsl.Var` object that represents a basis matrix.

A bit more behind the scenes:

- `gam.MultivariateNormalSingular`: An implementation of the singular multivariate normal distribution in the `tensorflow_probability` interface.
- `gam.star_ig_gibbs` and `gam.init_star_ig_gibbs`: Shortcuts for setting up a `gs.GibbsKernel` for a variance parameter with an inverse gamma prior.

## Usage

Usage is illustrated in the following notebooks.

- [notebooks/test_gam_gibbs.ipynb](https://github.com/liesel-devs/liesel_gam/blob/main/notebooks/test_gam_gibbs.ipynb): Uses the `gam.SmoothTerm.new_ig` constructor for the quickest and most convenient setup.
- [notebooks/test_gam_manual.ipynb](https://github.com/liesel-devs/liesel_gam/blob/main/notebooks/test_gam_manual.ipynb): Uses `gam.SmoothTerm` with a manually initialized scale parameter. This is less convenient, but demonstrates how to use any  `lsl.Var` for the scale parameter.

## Usage with bases and penalties from `mgcv` via `smoothcon`

We can get access to a large class of possible basis and penalty matrices by
interfacing with the wonderful R package [mgcv](https://cran.r-project.org/web/packages/mgcv/index.html)
via [liesel-devs/smoothcon](https://github.com/liesel-devs/smoothcon).

Example notebooks that illustrate smoothcon usage are provided in the [smoothcon
repository](https://github.com/liesel-devs/smoothcon/tree/main/notebooks).
