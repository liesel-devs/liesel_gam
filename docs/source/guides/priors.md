---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Set priors and samplers

Choose the coefficient and smoothing-scale priors for your terms, then attach
any inference specifications their constructors do not supply. This guide covers
Liesel-GAM's defaults and overrides; general kernel choice and grouping belong to
the [Goose kernel guide draft](https://github.com/liesel-devs/liesel/blob/docs/model-goose-guides/docs/source/goose-kernels.md).

## Understand the defaults

The default predictor intercept and `lin` coefficients have constant,
improper priors. They are not automatically regularized. Structured terms use
normal coefficient priors governed by their penalty and scale $\tau$.
The default hyperprior is
$\tau^2 \sim \operatorname{InverseGamma}(1, 0.005)$; it is on the
**variance**, not the scale. Smaller scales penalize departures from the
penalty's null space more strongly. The response distribution has its own
parameters, such as the normal response standard deviation.

The default smoothing prior is not independent of the units, basis, or penalty
scaling. Current constructors use design-aware penalty scaling where supported;
see {meth}`liesel_gam.Basis.scale_penalty` for its definition. Choose hyperpriors
in the context of plausible effect sizes and check sensitivity. An improper
coefficient prior also requires the likelihood and identification constraints
to yield a proper posterior.

## Change a smoothing prior

Use {class}`~liesel_gam.VarIGPrior` to keep a conjugate inverse-gamma variance
prior. This is a complete term setup:

The examples use `gam` and a builder `tb` for `gam.demo_data(n=200, seed=1)`.
See the {doc}`smooth-curve tutorial <../tutorials/smooth-curve>` for the setup.

```{code-cell} ipython3
import liesel.goose as gs
import liesel.model as lsl
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_gam as gam
```

```{code-cell} ipython3
:tags: [remove-cell]

df = gam.demo_data(n=200, seed=1)
tb = gam.TermBuilder.from_df(df)
```

```{code-cell} ipython3
smooth = tb.ps(
    "x_nonlin",
    k=15,
    scale=gam.VarIGPrior(2.0, 0.1),
)
```

For a different family, supply a positive scale variable. Here the prior is on
$\tau$ itself, with a half-normal scale of 1. These values illustrate the
API; calibrate them for your application.

```{code-cell} ipython3
tau = lsl.Var.new_param(
    1.0,
    dist=lsl.Dist(tfd.HalfNormal, scale=1.0),
    bijector=tfb.Exp(),
    inference=gs.MCMCSpec(gs.NUTSKernel),
    name="tau_custom",
)
custom_smooth = tb.ps(
    "x_nonlin",
    k=15,
    scale=tau,
    name="custom_smooth",
)
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: A smooth coefficient prior depends on a positive half-normal scale with an
      unconstrained source parameter.
---
custom_smooth.plot()
```

The scale is positive; Goose samples its unconstrained log-scale parameter.
See the [Liesel transformation guide draft](https://github.com/liesel-devs/liesel/blob/docs/model-goose-guides/docs/source/model-transformations.md) for the general bijector workflow. Passing a
numeric `scale=1.0` instead fixes the scale; it does not estimate it.
For a builder-wide custom prior, `default_scale_fn` must construct a fresh
variable for each term unless sharing a scale is intentional.

## Choose coefficient updates

`TermBuilder` and the default predictor intercept use
`gs.MCMCSpec(gs.IWLSKernel.untuned)`. A usual structured term with the default
variance prior gets a conjugate Gibbs update for its variance. A term's
`inference` argument changes only the coefficient update, not its scale update:

```{code-cell} ipython3
nuts_smooth = tb.ps(
    "x_nonlin",
    k=15,
    name="nuts_smooth",
    inference=gs.MCMCSpec(gs.NUTSKernel),
)
```

The default scale Gibbs update remains in place for this term. Conversely, the
custom `tau` above needs its own sampler; the coefficient sampler does not
supply one. Include adaptation for tuned kernels such as NUTS and HMC. The
{doc}`smooth-curve tutorial <../tutorials/smooth-curve>` shows the complete fit
and the diagnostics to inspect.

Anisotropic {doc}`tensor interactions <interactions>` generally lose the simple
conjugate variance update. Their constructors replace default scale Gibbs
specifications with log-scale HMC updates. Finish constructing interactions
before configuring the sampling run.

For optimization-based starting values, see {doc}`initialization`. Select the
coefficient parameters explicitly to hold smoothing variances fixed during
initialization while preserving their MCMC samplers.
