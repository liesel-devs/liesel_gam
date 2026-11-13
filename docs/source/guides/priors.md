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
see {meth}`liesel_gam.Basis.scale_penalty <liesel_gam.Basis.scale_penalty>` for its definition. Choose hyperpriors
in the context of plausible effect sizes and check sensitivity. An improper
coefficient prior also requires the likelihood and identification constraints
to yield a proper posterior.

## Change a smoothing prior

Use {class}`VarIGPrior <liesel_gam.VarIGPrior>` to keep a conjugate inverse-gamma variance
prior. This is a complete term setup:

The examples use `gam` and a builder `tb` for `gam.demo_data(n=200, seed=1)`.
See the {doc}`smooth-curve tutorial <../tutorials/smooth-curve>` for the setup.

```{code-cell} ipython3
import jax.numpy as jnp
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

## Calibrate a smooth to an effect bound

Use {class}`SDPrior <liesel_gam.SDPrior>` to choose a prior through a maximum
absolute effect. This example targets 95% prior probability that the stochastic
part stays within ±3 over the evaluation grid. The bound uses additive predictor
units: with a log link, these are log units. Unpenalized directions remain in the
model and are outside this bound.

```{code-cell} ipython3
sd_prior = gam.SDPrior.from_max_abs_effect(
    3.0,
    probability=0.95,
)
sd_smooth = tb.ps("x_nonlin", k=15, scale=sd_prior, name="sd_smooth")
```

The calibrated prior is Weibull with concentration 0.5 on $\tau^2$; the term
exposes $\tau$. Its automatic starting value is the prior median of $\tau$.
Supply `value=...` to choose a positive starting $\tau$ explicitly. The recipe
assigns NUTS to the log variance by default. Supply a different `inference`
specification to override it, or `inference=None` to disable automatic sampling.

Inspect a pure calculation through
{meth}`sd_prior.calibrate <liesel_gam.SDPrior.calibrate>`:

```{code-cell} ipython3
calibration = sd_prior.calibrate(sd_smooth.basis)
{
    "variance_prior_scale": round(calibration.prior_scale, 4),
    "suggested_tau_start": round(calibration.initial_scales[0], 4),
    "validation_probability": round(
        float(calibration.diagnostics["validation_probability"]), 3
    ),
}
```

The {class}`CalibrationResult <liesel_gam.CalibrationResult>` is immutable and
does not change a model's variables. Its suggestions and diagnostics describe
this calculation. The automatic scalar grid has 129 points, including the input
range endpoints; `grid_size=` overrides that count. Both the finite grid and
Monte Carlo validation are approximations, not certified continuous-domain bounds.

Reuse the recipe as an opt-in builder default. Supported scalar constructors are
{meth}`ps <liesel_gam.TermBuilder.ps>`, {meth}`np <liesel_gam.TermBuilder.np>`,
{meth}`cr <liesel_gam.TermBuilder.cr>`, {meth}`cs <liesel_gam.TermBuilder.cs>`,
{meth}`cc <liesel_gam.TermBuilder.cc>`, {meth}`bs <liesel_gam.TermBuilder.bs>`, and
{meth}`cp <liesel_gam.TermBuilder.cp>`. They receive independently calibrated
scale variables. Random intercepts, Markov random fields (MRFs), and eligible
custom bases also support this default. Other term constructors require an explicit
`scale=` with this builder default. Multivariate SD calibration is unsupported.
A builder without this opt-in uses the inverse-gamma default described above.

```{code-cell} ipython3
sd_tb = gam.TermBuilder.from_df(df, default_scale_fn=sd_prior)
another_smooth = sd_tb.np("x_nonlin", k=10)
```

## Bound group coefficients and region fields

Use the same recipe for a random intercept. Declare all categories, including
ones absent from the setup data:

```{code-cell} ipython3
group = gam.CatVar(
    ["treated", "control", "treated"],
    categories=["treated", "unused", "control"],
    name="group",
)
group_effect = sd_tb.ri(group)
group_calibration = sd_prior.calibrate(group_effect.basis)
group_calibration.diagnostics["evaluation_points"]
```

Calibration includes each of the three coefficients once. Repeating observations
or changing category frequencies alone leaves the random-intercept calibration
unchanged. A fixed positive semidefinite (PSD) `penalty=` uses category-code order,
including unused categories. An explicit zero penalty has no stochastic component
to calibrate and raises an error.

For an MRF, the bound applies to region field values after the fitted constraints,
penalty scaling, diagonalization, and low-rank transformations:

```{code-cell} ipython3
region = gam.CatVar(
    ["north", "east", "south", "north"],
    categories=["north", "east", "south", "west"],
    name="region",
)
neighbors = {
    "north": ["east", "west"],
    "east": ["north", "south"],
    "south": ["east", "west"],
    "west": ["north", "south"],
}
region_effect = sd_tb.mrf(region, nb=neighbors)
region_calibration = sd_prior.calibrate(region_effect.basis)
region_calibration.diagnostics["evaluation_points"]
```

Both finite domains use every declared level or represented region;
`grid_size=` never subsamples them. MRF setup geometry can depend on setup data,
so changing those data can change its calibration. For singular penalties,
unpenalized directions remain outside the bound. If calibration reports that a raw
penalty is too ill-conditioned, rebuild the term with `diagonal_penalty=True`.

## Reuse a calibrated coefficient in a product

{meth}`rs <liesel_gam.TermBuilder.rs>` calibrates its group coefficients through
{meth}`ri <liesel_gam.TermBuilder.ri>`.
{meth}`vc <liesel_gam.TermBuilder.vc>` preserves the supplied coefficient term's
prior and starting value. Neither calibrates the multiplied contribution, even
when the multiplier is random:

```{code-cell} ipython3
random_slope = sd_tb.rs("x_lin", cluster="x_cat")
varying_coefficient = sd_tb.vc(another_smooth, by="x_lin")
```

## Calibrate a compatible custom basis

{meth}`f <liesel_gam.TermBuilder.f>` uses the same numeric scalar grid when you
supply a known row-wise basis. Its input is a one-column numeric matrix, and its
output must be a matrix with one row per input row:

```{code-cell} ipython3
def quadratic_basis(x):
    return jnp.column_stack((x[:, 0], x[:, 0] ** 2))


custom_effect = sd_tb.f(
    "x_lin",
    basis_fn=quadratic_basis,
    row_wise=True,
)
```

Omitting the penalty gives independent, identically distributed (iid)
coefficients. You can also supply a fixed PSD penalty. Automatic calibration
rejects categorical inputs, unknown or
non-row-wise behavior, multiple input columns, and vector or higher-dimensional
outputs. Use an ordinary explicit scale for custom bases outside this contract.
Later arbitrary basis or
constraint edits do not automatically update calibration.

## Choose coefficient updates

{class}`TermBuilder <liesel_gam.TermBuilder>` and the default predictor intercept use
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
