---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Fit and initialize with Laplace

Fit a location-scale GAM with Liesel's `LaplaceLoss`, then use the fit to
initialize MCMC. The approximation integrates out the regression coefficients
while optimizing the smoothing parameters. You can use it for approximate
inference or as a starting point for sampling the full posterior.

```{important}
This is a development guide. It requires Liesel's published but **unmerged
`laplace-loss` branch**, tested at commit
`a68a8f4fe71001ebbf90e1e8872c8be0a3263914`. Released Liesel 0.5.2 does not
provide this workflow. Install the pinned development version in the same
environment as Liesel-GAM:

    pip install "liesel @ git+https://github.com/liesel-devs/liesel.git@a68a8f4fe71001ebbf90e1e8872c8be0a3263914"

The API may change before release. Graphviz is needed to display the model graph.
```

Read {doc}`location-scale` for the additive model. This notebook keeps the same
data and basis sizes, but changes the smoothing-variance parameterization for
optimization. For the algorithm and its curvature checks, see the
[Liesel Laplace guide draft](https://github.com/liesel-devs/liesel/blob/laplace-loss/docs/source/optimizer-laplace.md).

## Build the GAM

Enable 64-bit arithmetic before constructing arrays or the model. This helps
with the nested optimization and curvature calculations. The mean is nonlinear;
the response standard deviation increases with `x`.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import liesel.optim as opt
import numpy as np
import pandas as pd
import plotnine as p9
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_gam as gam

jax.config.update("jax_enable_x64", True)

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

Use {func}`scale_ig <liesel_gam.scale_ig>` for each smooth. It places an inverse-gamma
prior on the smoothing variance and makes **log variance** the writable
parameter. Optimizer steps can then range over the real line while the
variance stays positive. This smoothing variance controls the coefficient prior;
it is distinct from the response standard deviation `sigma`.

Set a NUTS specification on each log variance now for the later MCMC run.
The intercepts and smooth coefficients retain their default IWLS specifications.
The optimizer itself does not use these sampling specifications.

```{code-cell} ipython3
registry = gam.PandasRegistry(df)
tb_mu = gam.TermBuilder(registry, prefix_names_by="mu.")
tb_sigma = gam.TermBuilder(registry, prefix_names_by="sigma.")

mu = gam.AdditivePredictor("mu")
sigma = gam.AdditivePredictor("sigma", inv_link=jnp.exp)

mu_smooth = tb_mu.ps(
    "x",
    k=12,
    scale=gam.scale_ig(
        1.0,
        concentration=1.0,
        scale=0.005,
        inference=gs.MCMCSpec(gs.NUTSKernel),
    ),
)

sigma_smooth = tb_sigma.ps(
    "x",
    k=8,
    scale=gam.scale_ig(
        1.0,
        concentration=1.0,
        scale=0.005,
        inference=gs.MCMCSpec(gs.NUTSKernel),
    ),
)
mu += mu_smooth
sigma += sigma_smooth

y = lsl.Var.new_obs(
    df["y"].to_numpy(),
    dist=lsl.Dist(tfd.Normal, loc=mu, scale=sigma),
    name="y",
)
model = lsl.Model(y, to_float32=False)
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Location and log-scale predictors with separate centered smooths and log-variance
      parameters feed one normal response.
---
model.plot()
```

## Choose what to integrate

Include both intercepts and both coefficient vectors in `latent`. This also
integrates the unpenalized directions of the smooths. The only remaining
parameters are the two log smoothing variances, which the outer optimizer fits.
The printed names below make that split visible.

This optimizes an approximate marginal posterior density, including the variance
priors and transformation Jacobians. It is not a classical REML fit.

```{code-cell} ipython3
latent = [
    mu.intercept.name,
    mu_smooth.coef.name,
    sigma.intercept.name,
    sigma_smooth.coef.name,
]
loss = opt.LaplaceLoss(model, latent=latent)
```

```{code-cell} ipython3
pd.Series(
    {"latent": latent, "outer": loss.default_position_keys},
).to_frame("parameters")
```

## Fit the smoothing parameters

Use full data for both optimization and monitoring. `LaplaceLoss` does not
support minibatches. At each outer step, it solves for the conditional
coefficient mode and evaluates its curvature.

```{code-cell} ipython3
fit = opt.LieselOptim(
    model,
    loss=loss,
    optimizers="lbfgs",
    loss_monitor="train_full_data",
    stopper=opt.Stopper(epochs=300, patience=20, rtol=1e-10),
    show_progress=False,
).fit()
```

```{code-cell} ipython3
pd.Series(
    {
        "status": fit.status,
        "failure_reason": fit.failure_reason,
        "epochs": fit.n_epochs,
        "best_epoch": fit.min_monitor_epoch,
    }
).to_frame("value")
```

## Initialize from the fit

Inspect the status and failure reason above before using the fit. The best
monitored outer position and its stored inner
state belong together: the outer position alone omits the fitted coefficients.
Build a complete starting state from that pair.

```{code-cell} ipython3
outer = fit.position_min_monitor
inner = fit.loss_state_min_monitor
fitted_position = {**outer, **inner.latent_position}
fitted_state = model.update_state(fitted_position)
```

```{code-cell} ipython3
float(gs.LieselInterface(model).log_prob(fitted_state))
```

The fitted state has a finite log density. It combines fitted smoothing
parameters with their conditional coefficient mode; it is not the joint
posterior mode. `model.update_state` returns this state without changing the
model. To initialize the model itself, assign `model.state = fitted_state`.

For a deterministic MCMC start, pass `fitted_state` to
`builder.set_initial_values(fitted_state)`; Goose replicates it across chains.
This fit-only path needs no joint Gaussian approximation. Identical starts alone
provide limited evidence about convergence, so use distinct starts when possible.
The rest of this tutorial leaves the model unchanged and obtains distinct starts
from the joint approximation. Keep the data and fixed parameters unchanged too.

## Approximate curve uncertainty

Construct a joint Gaussian approximation that includes uncertainty in the
smoothing parameters and their dependence on the coefficients. Holding the fitted
smoothing parameters fixed would give a different, conditional approximation.
This call checks stationarity and curvature and raises if the approximation is
invalid. A successful check establishes local numerical validity, not accuracy
for a skewed or multimodal posterior.

```{code-cell} ipython3
posterior = loss.approximate_joint_posterior(
    fit,
    at="min_monitor",
    raise_on_failure=True,
    stationarity_tol=1e-4,
)
```

Draw from the approximation and predict both response parameters:

```{code-cell} ipython3
approx_samples = posterior.sample(sample_shape=(1, 4000), seed=jax.random.key(42))

x_grid = jnp.linspace(df["x"].min(), df["x"].max(), 100)
pred = model.predict(
    approx_samples,
    predict=[mu.name, sigma.name],
    newdata=gs.Position({"x": x_grid}),
)
curves = (
    gs.SamplesSummary(pred, which=("mean", "quantiles")).to_dataframe().reset_index()
)
curves["x"] = np.asarray(x_grid)[curves["var_index"].str[0].to_numpy(dtype=int)]
curves["truth"] = np.where(
    curves["variable"] == "mu",
    np.sin(2.0 * curves["x"]) + 0.3 * curves["x"],
    np.exp(-0.8 + 0.35 * curves["x"]),
)
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Approximate posterior mean and pointwise 90 percent bands for the mean and
      response standard deviation, compared with the true curves.
---
(
    p9.ggplot(curves, p9.aes("x", "mean"))
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

The curves follow the nonlinear mean and increasing spread, with wider
uncertainty near the right boundary. The mean curve still misses part of the
true curve there; the bands are pointwise and the approximation is local.

The leading size-one axis gives the sample dictionary the layout expected by
Liesel's prediction helpers. These are independent draws from a Gaussian
approximation, not an MCMC chain; R-hat and MCMC effective sample size do not
assess the approximation. The plot shows pointwise parameter uncertainty, not
predictive intervals for new observations. `sigma` is transformed within each
draw, so its band stays on the response-standard-deviation scale.

## Start distinct chains

For MCMC, use different starts for the chains. The joint approximation supplies
four complete parameter draws, including both smoothing parameters. Their
dependence with the coefficients is retained. Recompute each model state from
its draw; passing only the outer parameters would leave coefficients at the
wrong starting values.

Sample on the same transformed parameterization as the fitted model. In
particular, do not pass log variances into a second model whose writable
parameters are variances.

```{code-cell} ipython3
initial_positions = posterior.sample(sample_shape=(4,), seed=jax.random.key(43))
initial_states = jax.vmap(model.update_state)(initial_positions)
initial_log_probs = jax.vmap(gs.LieselInterface(model).log_prob)(initial_states)
```

```{code-cell} ipython3
np.asarray(initial_log_probs).round(2)
```

Check that each starting log density is finite before sampling. Here all four
draws give finite values. If Gaussian draws produce invalid states in another
model, inspect its constraints and the fit before using those draws as starts.

Pass these states to the builder with `multiple_chains=True`. Disable additional
jitter because the starts are already distinct. The builder's warning that no
jitter functions were provided is expected here. Unlike the Gibbs variance
updates in {doc}`location-scale`, the NUTS updates for log variance need an
adaptation phase.

```{code-cell} ipython3
builder = gs.LieselMCMC(model).get_engine_builder(
    seed=2027,
    num_chains=4,
    apply_jitter=False,
)
builder.set_initial_values(initial_states, multiple_chains=True)
builder.add_adaptation(1000)
builder.add_burnin(500)
builder.add_posterior(1500)
builder.show_progress = False
engine = builder.build()
```

```{code-cell} ipython3
engine.sample_all_epochs()
results = engine.get_results()

summary = gs.Summary(results)
```

```{code-cell} ipython3
summary.aggregate_diagnostics().round({"ess_bulk": 0, "ess_tail": 0, "rhat": 3})
```

```{code-cell} ipython3
summary.error_df()[["count", "relative"]]
```

Inspect R-hat, effective sample sizes, and errors for each kernel above.
Distinguish warmup divergences from errors in retained posterior draws.
A good initialization alone does not establish convergence.

These chains target the full model posterior, with smoothing parameters sampled
alongside the coefficients. The Laplace fit only supplied their starting states;
it did not replace the sampling target. Initialization does not remove the need
for adaptation or convergence checks, and nearby Gaussian starts can miss other
modes. See the [Goose diagnostics guide draft](https://github.com/liesel-devs/liesel/blob/docs/model-goose-guides/docs/source/goose-diagnostics.md)
for the checks and {doc}`../guides/prediction` to use
`results.get_posterior_samples()` for GAM predictions.

## When to use this workflow

Use Laplace fitting for a local approximation or to initialize a moderate-sized
GAM before MCMC. This implementation uses dense latent curvature: memory grows
quadratically and factorization work cubically in the number of integrated
coefficients. Large spatial effects or tensor products can make it expensive.
Prior choices and an identifiable model still matter; see {doc}`../guides/priors`
and the Liesel Laplace guide draft (`docs/source/optimizer-laplace.rst` in that checkout)
for optimization failures and approximation limits.
