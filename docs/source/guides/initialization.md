---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Initialize a GAM

Use a conditional coefficient fit to obtain starting values for MCMC. Keep
smoothing and group-effect variances fixed during this step, then let their
usual samplers update them during MCMC. This uses the `optim-base` API assumed
by these guides; it does not require `LaplaceLoss`.

## Choose the parameters

Start with `model`, `mu`, and `smooth` from a normal mean model: an intercept
and a centered P-spline (`k=10`) for `x_nonlin`, with response standard deviation
fixed at 1. The {doc}`smooth-curve tutorial <../tutorials/smooth-curve>` shows
how to construct this model. This example uses `gam.demo_data(n=120, seed=7)`
and pandas as `pd`. The graph shows the prerequisite model:

```{code-cell} ipython3
import liesel.model as lsl
import liesel.optim as opt
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_gam as gam
```

```{code-cell} ipython3
:tags: [remove-cell]

df = gam.demo_data(n=120, seed=7)
tb = gam.TermBuilder.from_df(df)
smooth = tb.ps("x_nonlin", k=10)
mu = gam.AdditivePredictor("mu")
mu += smooth

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
    alt: A normal response with one centered smooth and an intercept; the smoothing
      variance remains a parameter.
---
model.plot()
```

Select the intercept and smooth coefficients explicitly:

```{code-cell} ipython3
coef_keys = [mu.intercept.name, smooth.coef.name]
scale_before = float(smooth.scale.value)

fit = opt.LieselOptim(
    model,
    optimizers=[opt.LBFGS(coef_keys)],
    loss_monitor="train_full_data",
    show_progress=False,
).fit()
```

```{code-cell} ipython3
pd.Series(
    {
        "status": fit.status,
        "epochs": fit.n_epochs,
        "best_epoch": fit.min_monitor_epoch,
    },
).to_frame("value")
```

`NegLogProbLoss` is the default and includes both likelihood and priors.
Parameters omitted from `coef_keys` stay at their current values. For several
terms, include each coefficient vector you want to initialize. This also applies
to the main effects and tensor terms in {doc}`interactions`: select their
coefficients and leave the shared smoothing scales fixed.

Avoid bare `optimizers="lbfgs"` here: it selects all model parameters.
Do not change parameter flags, remove priors, or replace the scale variables.
Holding variances fixed is an optimizer choice, not a change to the model.
Their starting values affect the coefficient fit; choose plausible values in
light of the data units and penalty scaling.

## Treat response scale separately

A smoothing variance controls a coefficient prior. A response scale controls
variation in the observations. In a location-scale GAM, you can include the
log-scale predictor's intercept and coefficients while holding its smoothing
variance fixed. A scalar response standard deviation may instead stay fixed or
be optimized deliberately through its unconstrained parameter.

Joint coefficient/variance modes can favor strong smoothing and depend on the
parameterization. Transforming a positive variance prevents invalid optimizer
steps but does not remove that dependence. Conditional initialization avoids
using such a joint mode as the default smoothing estimate.

## Pass the fit to Goose

Inspect `fit.status` and the saved fit before using it; an earlier finite
position can still be available after a later optimizer failure. Then update
the model's starting state:

```{code-cell} ipython3
model.state = model.update_state(fit.position_min_monitor)
```

```{code-cell} ipython3
pd.DataFrame(
    {"before": [scale_before], "after": [float(smooth.scale.value)]},
    index=["smoothing scale"],
)
```

Fitting leaves the model unchanged until this assignment. Its IWLS, Gibbs, or
HMC specifications remain attached, so the next `gs.LieselMCMC(model)` uses the
updated values and still samples the smoothing parameters. Keep basis evaluation
inside the model and pass raw covariates for prediction.

A single fitted state does not disperse chains. Follow the
[Goose initialization guide draft](https://github.com/liesel-devs/liesel/blob/docs/model-goose-guides/docs/source/goose-initialization.md)
for distinct starts, and retain the warmup and diagnostics required by your
kernels. For optimizer controls and loss monitoring, see the
[Liesel optimization guide](https://github.com/liesel-devs/liesel/blob/optim-base/docs/source/optimization.md).
