---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Liesel-GAM

Build Bayesian additive models for the mean, scale, or other parameters of a
response distribution. Liesel-GAM supplies linear effects, smooths, spatial
terms, group effects, and their priors. You connect the predictors to a response
distribution in Liesel, then use Goose for MCMC inference.
The development tutorial also covers Laplace fitting and MCMC initialization.

## Install

Install Liesel-GAM with Python 3.13 or 3.14:

```bash
pip install liesel_gam
```

These guides assume a Liesel version with the `optim-base` optimizer API
(tested at `a68a8f4`). Until that API is in your installed release, install
the pinned development version in the same environment:

```bash
pip install "liesel @ git+https://github.com/liesel-devs/liesel.git@a68a8f4fe71001ebbf90e1e8872c8be0a3263914"
```

Smooth bases and penalties use the native `smoothcon` package. No R installation
is required. Install Graphviz separately to render model graphs locally.

## Prerequisites

These guides assume Liesel model construction and a basic Goose sampling workflow.
For those topics, see the [model-building guide draft](https://github.com/liesel-devs/liesel/blob/docs/model-goose-guides/docs/source/model-building.md)
and [Goose guide draft](https://github.com/liesel-devs/liesel/blob/docs/model-goose-guides/docs/source/sampling.md).
The tutorials here keep the complete runnable code while focusing on additive
terms, smoothing priors, and effect interpretation.

## Build a predictor

This complete model uses simulated data with a nonlinear mean and fixes the
response standard deviation at 1. The tutorials below show how to estimate it.

```{code-cell} ipython3
import liesel.model as lsl
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_gam as gam

df = gam.demo_data(n=200, seed=1)
tb = gam.TermBuilder.from_df(df)

mu = gam.AdditivePredictor("mu")
mu += tb.lin("x_lin + C(x_cat)")
mu += tb.ps("x_nonlin", k=15)

y = lsl.Var.new_obs(
    df["y"].to_numpy(),
    dist=lsl.Dist(tfd.Normal, loc=mu, scale=1.0),
    name="y",
)
model = lsl.Model(y)
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: A normal response depends on an additive predictor with linear, categorical,
      and nonlinear terms.
---
model.plot()
```

The predictor includes an intercept. The smooth is centered, and its variance
prior controls roughness; see {doc}`guides/priors` for defaults and alternatives.

## Start here

```{toctree}
:maxdepth: 1
:caption: Start here

Overview <self>
tutorials/smooth-curve
Location and scale <tutorials/location-scale>
Laplace (development) <tutorials/laplace>
```

## Common tasks

```{toctree}
:maxdepth: 1
:caption: Common tasks

guides/terms
Priors and samplers <guides/priors>
guides/interactions
guides/initialization
guides/prediction
guides/custom-bases
guides/large-data
```

## Example library

The maintained Markdown examples execute during the build. Older JSON notebooks
remain as archived examples with saved outputs; they are not execution-validated
by the build. They illustrate additional effect types and specialized workflows.
Start with the tutorials above for the main fitting and prediction workflow.

```{toctree}
:maxdepth: 1
:caption: Example library

notebooks_lin
notebooks_univariate
notebooks_composite
notebooks_multivariate
notebooks_large_data
```

## Relevant Literature

Fahrmeier et al. (2013) is a textbook that introduces structured additive
regression concepts from the ground up. Wood (2017) is another seminal textbook on
generalized additive models. The R package mgcv provides many basis and penalty
constructions implemented by the standalone `smoothcon` dependency.


- Fahrmeir, L., Kneib, T., Lang, S., & Marx, B. (2013). Regression—Models, methods and
  applications. Springer. https://doi.org/10.1007/978-3-642-34333-9
- Wood, S. N. (2017). Generalized additive models (2nd ed.). Chapman & Hall/CRC.
- R package mgcv: https://cran.r-project.org/web/packages/mgcv/index.html

The other references are seminal papers on structured additive distributional
regression.

- Kneib, T., Klein, N., Lang, S., & Umlauf, N. (2019). Modular regression—A Lego system
  for building structured additive distributional regression models with tensor product
  interactions. TEST, 28(1), 1–39. https://doi.org/10.1007/s11749-019-00631-z
- Umlauf, N., Klein, N., & Zeileis, A. (2018). Bamlss: Bayesian additive models for
  location, scale, and shape (and beyond). Journal of Computational and Graphical
  Statistics, 27(3), 612–627. https://doi.org/10.1080/10618600.2017.1407325
- Klein, N., Kneib, T., Lang, S., & Sohn, A. (2015). Bayesian structured additive
  distributional regression with an application to regional income inequality in
  Germany. The Annals of Applied Statistics, 9(2), 1024–1052.
  https://doi.org/10.1214/15-AOAS823




## API Reference

### High-level API

```{autosummary}
:toctree: generated
 :caption: High-level API
 :nosignatures:

 ~liesel_gam.AdditivePredictor
 ~liesel_gam.MVAdditivePredictor
 ~liesel_gam.TermBuilder
 ~liesel_gam.MVTermBuilder
 ~liesel_gam.BasisBuilder
```

### Plots

```{autosummary}
:toctree: generated
 :caption: Plots
 :nosignatures:

 ~liesel_gam.plot_1d_smooth
 ~liesel_gam.plot_2d_smooth
 ~liesel_gam.plot_forest
 ~liesel_gam.plot_polys
 ~liesel_gam.plot_regions
 ~liesel_gam.plot_1d_smooth_clustered
```

### Summary

```{autosummary}
:toctree: generated
 :caption: Summary
 :nosignatures:

 ~liesel_gam.summarise_1d_smooth
 ~liesel_gam.summarise_nd_smooth
 ~liesel_gam.summarise_lin
 ~liesel_gam.summarise_cluster
 ~liesel_gam.summarise_regions
 ~liesel_gam.summarise_1d_smooth_clustered
 ~liesel_gam.summarise_by_samples
 ~liesel_gam.polys_to_df
```

### Bases

```{autosummary}
:toctree: generated
 :caption: Bases
 :nosignatures:

 ~liesel_gam.Basis
 ~liesel_gam.ApproximationSpec
 ~liesel_gam.MRFBasis
 ~liesel_gam.LinBasis
```

### Terms and Variables

```{autosummary}
:toctree: generated
 :caption: Terms
 :nosignatures:

 ~liesel_gam.StrctTerm
 ~liesel_gam.StrctInteractionTerm
 ~liesel_gam.StrctTensorProdTerm
 ~liesel_gam.MultivariateStrctTerm
 ~liesel_gam.MultivariateStrctInteractionTerm
 ~liesel_gam.MultivariateStrctLinTerm
 ~liesel_gam.MultivariateTPTerm
 ~liesel_gam.MultivariateIntercept
 ~liesel_gam.MultivariateContribution
 ~liesel_gam.LinTerm
 ~liesel_gam.StrctLinTerm
 ~liesel_gam.LinMixin
 ~liesel_gam.IndexingTerm
 ~liesel_gam.RITerm
 ~liesel_gam.MRFTerm
 ~liesel_gam.BasisDot
 ~liesel_gam.ScaleIG
 ~liesel_gam.CatVar
 ~liesel_gam.UserVar
```

### Distribution

```{autosummary}
:toctree: generated
 :caption: Distribution
 :nosignatures:

 ~liesel_gam.MultivariateNormalSingular
 ~liesel_gam.MultivariateNormalStructured
 ~liesel_gam.StructuredPenaltyOperator
```

### Other

```{autosummary}
:toctree: generated
 :caption: Other
 :nosignatures:

 ~liesel_gam.DictRegistry
 ~liesel_gam.PandasRegistry
 ~liesel_gam.CategoryMapping
 ~liesel_gam.MRFSpec
 ~liesel_gam.NameManager
 ~liesel_gam.VarIGPrior
 ~liesel_gam.scale_ig
 ~liesel_gam.scale_wb
 ~liesel_gam.demo_data
 ~liesel_gam.demo_data_ta
 ~liesel_gam.LinearConstraintEVD
 ~liesel_gam.basis_setup_sample
 ~liesel_gam.category_coverage_indices
```

```{rubric} In/Out

```

```{autosummary}
:toctree: generated
 :caption: In/Out
 :nosignatures:

 ~liesel_gam.io.read_bnd
 ~liesel_gam.io.polygon_is_closed
```

### Experimental

The API of modules, classes and functions in the experimental module is less stable
than in other modules of `liesel_gam`. If you depend on this, expect changes in the
future.

```{autosummary}
:toctree: generated
 :caption: Experimental
 :nosignatures:

 ~liesel_gam.experimental.BSplineApprox
```

## Acknowledgements and Funding

We are
grateful to the [German Research Foundation (DFG)](https://www.dfg.de/en) for funding the development
through grant 443179956.

```{image} https://raw.githubusercontent.com/liesel-devs/liesel/main/docs/source/_static/uni-goe.svg
:alt: University of Göttingen
```

```{image} https://raw.githubusercontent.com/liesel-devs/liesel/main/docs/source/_static/funded-by-dfg.svg
:alt: Funded by DFG
```

# Indices and tables

* {ref}`genindex`
* {ref}`search`
