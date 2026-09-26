---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Fit a smooth curve

Fit a nonlinear mean with a P-spline and learn how much smoothing the data support.
You will build a normal response model, sample its posterior, and plot the fitted
curve with uncertainty. This tutorial assumes the [Liesel model](https://github.com/liesel-devs/liesel/blob/docs/model-goose-guides/docs/source/model-building.md)
and [Goose sampling](https://github.com/liesel-devs/liesel/blob/docs/model-goose-guides/docs/source/sampling.md)
workflows (currently guide drafts). It runs from top to bottom without external data.

## Generate observations

These simulated measurements follow a curved trend with constant measurement
noise. Simulation lets us compare the fitted curve with the known mean.

```{code-cell} ipython3
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd
import plotnine as p9
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_gam as gam
```

```{code-cell} ipython3
rng = np.random.default_rng(2026)
x = np.sort(rng.uniform(-2.0, 2.0, 160))
true_mean = np.sin(2.0 * x) + 0.3 * x
df = pd.DataFrame(
    {
        "x": x,
        "y": true_mean + rng.normal(0.0, 0.35, len(x)),
    },
)
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Simulated measurements follow a curved trend with roughly constant spread.
---
(
    p9.ggplot(df, p9.aes("x", "y"))
    + p9.geom_point(alpha=0.5)
    + p9.labs(x="Covariate", y="Measurement")
    + p9.theme_minimal()
)
```

The trend is nonlinear, but its shape is unknown to the model. A P-spline
represents that trend with a flexible basis and penalizes rough coefficients.

## Define the model

The predictor supplies an intercept. Its default prior is constant (improper),
and the smooth is centered over the training observations to separate it from
that intercept. `k=12` sets the number of basis functions before the constraint;
it does not fix the amount of smoothing.

The smooth uses the default prior $\tau^2 \sim \operatorname{InverseGamma}(1, 0.005)$
on its smoothing variance. Smaller $\tau$ penalizes roughness more strongly.
We give the response standard deviation `sigma` a separate half-normal prior
with scale 1, appropriate for these simulated response units. Smoothing variance
and response variance play different roles.

```{code-cell} ipython3
tb = gam.TermBuilder.from_df(df)

mu = gam.AdditivePredictor("mu")
smooth = tb.ps("x", k=12)
mu += smooth

sigma = lsl.Var.new_param(
    0.5,
    dist=lsl.Dist(tfd.HalfNormal, scale=1.0),
    bijector=tfb.Exp(),
    inference=gs.MCMCSpec(gs.NUTSKernel),
    name="sigma",
)
y = lsl.Var.new_obs(
    df["y"].to_numpy(),
    dist=lsl.Dist(tfd.Normal, loc=mu, scale=sigma),
    name="y",
)
model = lsl.Model(y)
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Model graph connecting x, the spline coefficients and smoothing variance
      to mu, and mu and sigma to the observed response y.
---
model.plot()
```

The graph exposes the smooth's basis, coefficients, and smoothing variance.
Liesel-GAM creates these variables and their inference specifications; you still
choose the response distribution in Liesel.

## Sample the posterior

The intercept and spline coefficients use the default untuned IWLS updates.
The default smoothing variance uses a conjugate Gibbs update. The positive
response scale is sampled on the log scale by NUTS, so we include adaptation.

```{code-cell} ipython3
results = gs.LieselMCMC(model).run_for_epochs(
    seed=2026,
    num_chains=4,
    adaptation=1000,
    posterior=1500,
    show_progress=False,
)
samples = results.get_posterior_samples()
```

## Check the fit

Check the fitted GAM before interpreting its effects:

```{code-cell} ipython3
summary = gs.Summary(results)
```

```{code-cell} ipython3
summary.aggregate_diagnostics().round({"ess_bulk": 0, "ess_tail": 0, "rhat": 3})
```

```{code-cell} ipython3
summary.error_df().reset_index().reindex(
    columns=["positions", "error_msg", "phase", "count"]
)
```

Inspect effective sample sizes and R-hat for each parameter. For the NUTS
response-scale update, distinguish divergences during warmup from those in
retained posterior draws. See the
[Goose diagnostics guide draft](https://github.com/liesel-devs/liesel/blob/docs/model-goose-guides/docs/source/goose-diagnostics.md)
for interpreting these checks and investigating divergences. Results can vary
by platform.

## Inspect the smooth

The term plot shows the centered effect, excluding the intercept. Its band is a
pointwise 90% credible interval for this effect.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Posterior mean of the centered spline effect with a pointwise 90 percent
      credible band.
---
gam.plot_1d_smooth(
    smooth,
    samples,
    ci_quantiles=(0.05, 0.95),
    show_n_samples=None,
)
```

## Predict the mean

Predict `mu` to include the intercept. New covariate values are supplied under
the training input name `x`; the fitted basis and centering are reused. Here we
stay inside the observed range.

```{code-cell} ipython3
x_grid = jnp.linspace(df["x"].min(), df["x"].max(), 100)
pred = model.predict(
    samples,
    predict=[mu.name],
    newdata=gs.Position({"x": x_grid}),
)
mean_summary = gs.SamplesSummary(pred).to_dataframe().reset_index()
mean_summary["x"] = np.asarray(x_grid)
mean_summary["truth"] = np.sin(2.0 * mean_summary["x"]) + 0.3 * mean_summary["x"]
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Fitted mean and pointwise 90 percent credible band over observations; a dashed
      orange line shows the known simulated mean.
---
(
    p9.ggplot(mean_summary, p9.aes("x", "mean"))
    + p9.geom_ribbon(p9.aes(ymin="q_0.05", ymax="q_0.95"), alpha=0.25)
    + p9.geom_line(color="#0072B2")
    + p9.geom_line(
        p9.aes(y="truth"),
        color="#D55E00",
        linetype="dashed",
    )
    + p9.geom_point(
        p9.aes("x", "y"),
        data=df,
        inherit_aes=False,
        alpha=0.35,
    )
    + p9.labs(
        x="Covariate",
        y="Conditional mean",
        subtitle="Dashed orange: true mean",
    )
    + p9.theme_minimal()
)
```

The fitted curve follows the known simulated mean (dashed orange), with greater
uncertainty near the edges. This is a comparison with the generating curve, not
an out-of-sample evaluation.

The band measures uncertainty about the mean, so observations need not lie
inside it. To describe a new observation, include response variation through
posterior predictive sampling. Continue with {doc}`location-scale` for changing
response spread, or see {doc}`../guides/prediction` for prediction tasks.
