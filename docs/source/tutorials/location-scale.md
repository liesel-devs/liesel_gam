---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Model location and scale

Let a covariate change both the expected response and its spread. This is a
normal location-scale model: $y_i \sim \mathcal{N}(\mu(x_i), \sigma(x_i)^2)$.
The same building blocks also work for other distribution parameters.

Read {doc}`smooth-curve` first for the initial GAM workflow. This
notebook is self-contained and uses fresh simulated observations.

## Generate changing spread

The conditional mean is curved, and the response standard deviation grows with
`x`. Both are unknown to the fitted model.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd
import plotnine as p9
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_gam as gam
```

```{code-cell} ipython3
rng = np.random.default_rng(2027)
x = np.sort(rng.uniform(-2.0, 2.0, 200))
true_mean = np.sin(2.0 * x) + 0.3 * x
true_sigma = np.exp(-0.8 + 0.35 * x)
df = pd.DataFrame(
    {
        "x": x,
        "y": true_mean + true_sigma * rng.normal(size=len(x)),
    },
)
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Simulated observations with a curved mean and increasing response spread
      from left to right.
---
(
    p9.ggplot(df, p9.aes("x", "y"))
    + p9.geom_point(alpha=0.5)
    + p9.labs(x="Covariate", y="Measurement")
    + p9.theme_minimal()
)
```

## Define two predictors

Use one registry so both predictors share the same covariate variable. Separate
builders give their terms distinct names. Each predictor has its own intercept
with a constant prior and its own centered smooth with the default inverse-gamma
prior on smoothing variance.

The exponential inverse link makes `sigma` positive. Its terms add on the
**log-standard-deviation** scale, while {attr}`sigma.value <liesel.model.Var.value>` is on the response scale.
`mu` uses the identity link.

```{code-cell} ipython3
registry = gam.PandasRegistry(df)
tb_mu = gam.TermBuilder(registry, prefix_names_by="mu.")
tb_sigma = gam.TermBuilder(registry, prefix_names_by="sigma.")

mu = gam.AdditivePredictor("mu")
sigma = gam.AdditivePredictor("sigma", inv_link=jnp.exp)
mu += tb_mu.ps("x", k=12)
sigma += tb_sigma.ps("x", k=8)

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
    alt: Model graph with a shared covariate x feeding separate spline predictors
      for mean mu and response standard deviation sigma.
---
model.plot()
```

The two smooths have separate coefficients and smoothing variances.
Their parameters are estimated in the same model through one likelihood. Fitting a
scale curve to residuals from a separate mean fit would omit that joint uncertainty.

## Fit and check sampling

All sampled parameters here carry default IWLS or Gibbs specifications.
These defaults need no step-size tuning, so a burn-in phase is sufficient here.

```{code-cell} ipython3
results = gs.LieselMCMC(model).run_for_epochs(
    seed=2027,
    num_chains=4,
    adaptation=0,
    burnin=1000,
    posterior=1500,
    show_progress=False,
)
samples = results.get_posterior_samples()
summary = gs.Summary(results)
```

```{code-cell} ipython3
summary.aggregate_diagnostics().round({"ess_bulk": 0, "ess_tail": 0, "rhat": 3})
```

```{code-cell} ipython3
summary.error_df()
```

Inspect effective sample sizes, R-hat, and sampler errors above. Use the [Goose diagnostics guide draft](https://github.com/liesel-devs/liesel/blob/docs/model-goose-guides/docs/source/goose-diagnostics.md)
for diagnostic definitions and next steps. These checks concern sampling, not
the adequacy of the normal response model.

## Predict both parameters

Requesting `sigma` applies the inverse link **within each posterior draw**.
Exponentiating an average log-scale curve would give a different summary.
The parameter bands below describe uncertainty about $\mu(x)$ and $\sigma(x)$;
the scale band is not a predictive interval for $y$.

```{code-cell} ipython3
x_grid = jnp.linspace(df["x"].min(), df["x"].max(), 100)
pred = model.predict(
    samples,
    predict=[mu.name, sigma.name],
    newdata=gs.Position({"x": x_grid}),
)
parameter_summary = gs.SamplesSummary(pred).to_dataframe().reset_index()
parameter_summary["x"] = np.tile(np.asarray(x_grid), len(pred))
parameter_summary["truth"] = np.where(
    parameter_summary["variable"] == "mu",
    np.sin(2.0 * parameter_summary["x"]) + 0.3 * parameter_summary["x"],
    np.exp(-0.8 + 0.35 * parameter_summary["x"]),
)
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Separate panels compare fitted mu and sigma with their known simulated curves,
      shown as dashed orange lines, and pointwise 90 percent credible bands.
---
(
    p9.ggplot(parameter_summary, p9.aes("x", "mean"))
    + p9.geom_ribbon(p9.aes(ymin="q_0.05", ymax="q_0.95"), alpha=0.25)
    + p9.geom_line(color="#0072B2")
    + p9.geom_line(
        p9.aes(y="truth"),
        color="#D55E00",
        linetype="dashed",
    )
    + p9.facet_wrap("~variable", scales="free_y", ncol=1)
    + p9.labs(
        x="Covariate",
        y="Response parameter",
        subtitle="Dashed orange: true curve",
    )
    + p9.theme_minimal()
)
```

The fitted curves recover the curved mean and increasing spread. Some boundary
estimates of the mean miss the true curve. Pointwise bands need not cover a whole
function; basis size and prior sensitivity also deserve checking in an application.

## Compare response intervals

Use the same fitted GAM to compare mean uncertainty with response variation.
The simulation below follows the [Liesel simulation guide draft](https://github.com/liesel-devs/liesel/blob/docs/model-goose-guides/docs/source/model-simulation.md);
the response placeholder matches the new grid length.

```{code-cell} ipython3
key = jax.random.key(2028)
key, draw_key = jax.random.split(key)
rep = model.sample(
    shape=(),
    seed=draw_key,
    posterior_samples=samples,
    newdata=gs.Position({"x": x_grid, "y": jnp.zeros_like(x_grid)}),
)
predictive_summary = gs.SamplesSummary({"y": rep["y"]}).to_dataframe().reset_index()
predictive_summary["x"] = np.asarray(x_grid)
mean_summary = parameter_summary.query("variable == 'mu'")
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Wide orange predictive interval grows with x; a narrower blue credible band
      describes uncertainty about the mean, with observed points overlaid.
---
(
    p9.ggplot(predictive_summary, p9.aes("x", "mean"))
    + p9.geom_ribbon(
        p9.aes(ymin="q_0.05", ymax="q_0.95"),
        fill="#E69F00",
        alpha=0.3,
    )
    + p9.geom_ribbon(
        p9.aes(ymin="q_0.05", ymax="q_0.95"),
        data=mean_summary,
        fill="#0072B2",
        alpha=0.6,
    )
    + p9.geom_line(data=mean_summary, color="#0072B2")
    + p9.geom_point(
        p9.aes("x", "y"),
        data=df,
        inherit_aes=False,
        alpha=0.35,
    )
    + p9.labs(
        x="Covariate",
        y="Response",
        subtitle="Orange: new response; blue: mean",
    )
    + p9.theme_minimal()
)
```

The orange pointwise 90% predictive interval includes parameter uncertainty
and response variation. The blue interval only describes the mean. Neither band
is a simultaneous statement about the whole curve. In particular, averaging
$\mu \pm \sigma$ over draws does not give a 90% predictive interval.

For posterior predictive checks, compare spread and tails across covariate values
at the observed design. See the [Liesel simulation guide draft](https://github.com/liesel-devs/liesel/blob/docs/model-goose-guides/docs/source/model-simulation.md)
for the general workflow, and the [Goose model-comparison draft](https://github.com/liesel-devs/liesel/blob/docs/model-goose-guides/docs/source/goose-model-comparison.md)
for predictive model comparison. Change GAM assumptions with
{doc}`../guides/priors`, or continue to {doc}`../guides/prediction` for group
predictions and effect summaries.
