---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Build interactions

An interaction lets the effect of one covariate depend on another. Build it
from marginal terms, then decide which effects belong in the predictor.
{meth}`~liesel_gam.TermBuilder.tx` returns the highest-order interaction;
{meth}`~liesel_gam.TermBuilder.tf` bundles main effects and interactions by default.
The graphs below show that difference for two centered P-spline marginals.

## Build an interaction with tx

Start with the interaction $h(x,y)$ alone. Plot the term to see its
coefficient and prior dependencies before adding it to a response model:

Use `gam` and a builder `tb` for `gam.demo_data_ta(n=200, seed=42)`, with
covariates `x`, `y` and response `z`. The
{doc}`smooth-curve tutorial <../tutorials/smooth-curve>` introduces builder setup.

```{code-cell} ipython3
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import plotnine as p9
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_gam as gam
```

```{code-cell} ipython3
:tags: [remove-cell]

df = gam.demo_data_ta(n=200, seed=42)
tb = gam.TermBuilder.from_df(df)
```

```{code-cell} ipython3
sx = tb.ps("x", k=8)
sy = tb.ps("y", k=8)
interaction = tb.tx(sx, sy)
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: The tx interaction depends on a tensor basis and its coefficient vector,
      whose prior uses both marginal smoothing scales.
---
interaction.plot(width=10, height=7)
```

The marginal bases and scales feed one interaction coefficient vector.
The main-effect coefficient vectors are absent from this subgraph.

`tx` adds only the highest-order interaction. The centered marginal bases
separate it from their main effects. With linear or categorical marginals,
contrast coding and identification need their own care: those design matrices
are reused without additional centering.

## Add the main effects explicitly

For $\mu(x,y) = \beta_0 + f(x) + g(y) + h(x,y)$, include the original
marginal terms alongside the interaction. Plotting the predictor shows all
three contributions and the intercept, without a likelihood or response node:

```{code-cell} ipython3
mu = gam.AdditivePredictor("mu")
mu += sx, sy, interaction
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: The mu predictor adds an intercept, the x and y main effects, and their tx
      interaction; the terms share marginal smoothing scales.
---
mu.plot(width=10, height=9)
```

Each main effect has its own coefficients. Its smoothing scale also enters
the interaction prior.

## Bundle all effects with tf

Use a fresh builder for the alternative construction. `tf` collects the two
main effects and their interaction in one term, which you add once to the
predictor:

```{code-cell} ipython3
full_tb = gam.TermBuilder.from_df(df)
full_sx = full_tb.ps("x", k=8)
full_sy = full_tb.ps("y", k=8)

full_surface = full_tb.tf(
    full_sx,
    full_sy,
    group_terms_by_order=True,
)

full_mu = gam.AdditivePredictor("mu")
full_mu += full_surface
```

`group_terms_by_order=True` groups main effects and the interaction in the
graph. It changes the display structure, not which effects are included.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: The tf term sums the x main effect, the y main effect, and their interaction,
      retaining distinct coefficient vectors and shared smoothing scales.
---
full_surface.plot(width=10, height=9)
```

The `tf` node collects the same three effect components that were added
separately above. The predictor supplies the intercept outside this term.

These are alternative model constructions. Do not add the same main effects
again alongside `tf`. With more than two marginals, its default includes the
lower-order interactions as well; `order` can select which orders to include.
For example, `order=(2, 3)` includes pairs and the three-way interaction for
three marginals, while omitting their main effects.

Choose the graph root for the question: `interaction.plot()` shows a single
interaction, `full_surface.plot()` shows the bundled surface, and `mu.plot()`
shows a predictor. After connecting a response distribution, `model.plot()`
shows the complete model, as in {doc}`../tutorials/location-scale`.

## Account for shared scales

By default, the tensor uses its marginal terms' scales. Reusing those terms
as main effects therefore shares smoothing parameters between the main effects
and the interaction. This is a model assumption. `common_scale` instead
constructs an isotropic tensor. For `tx` it leaves main-effect scales alone;
for `tf` it also changes the supplied structured main effects' scales.
See {meth}`~liesel_gam.TermBuilder.tx` and {meth}`~liesel_gam.TermBuilder.tf`.

The anisotropic prior generally needs nonconjugate scale updates. The constructors
replace default variance Gibbs samplers with HMC specifications on the log scale,
so use an adaptation phase. Tensor coefficient counts multiply across marginal
basis dimensions; start with modest bases and check runtime and fit.

## Predict slices

Fit the first construction to the simulated response `z`. Here the response
standard deviation is fixed at its generating value, 0.3, to focus on the
interaction. The two smoothing variances are estimated. Fit with adaptation
for the tensor's scale samplers:

```{code-cell} ipython3
z = lsl.Var.new_obs(
    df["z"].to_numpy(),
    dist=lsl.Dist(tfd.Normal, loc=mu, scale=0.3),
    name="z",
)
model = lsl.Model(z)
```

```{code-cell} ipython3
results = gs.LieselMCMC(model).run_for_epochs(
    seed=42,
    num_chains=4,
    adaptation=1000,
    burnin=1000,
    posterior=6000,
    show_progress=False,
)
samples = results.get_posterior_samples()
```

```{code-cell} ipython3
summary = gs.Summary(results)
```

```{code-cell} ipython3
summary.aggregate_diagnostics().round({"ess_bulk": 0, "ess_tail": 0, "rhat": 3})
```

```{code-cell} ipython3
summary.error_df()[["count", "relative"]]
```

Inspect the effective sample sizes alongside R-hat: a relatively small ESS
can reveal slow mixing of a smoothing variance despite a long run. Distinguish
warmup divergences from errors in retained samples. Use the [Goose diagnostics guide draft](https://github.com/liesel-devs/liesel/blob/docs/goose-guides/docs/source/goose-diagnostics.rst).

A slice holds one covariate fixed while varying the other. Here, sweep across
`x` three times, holding `y` at 0.2, 0.5, and 0.8. Add the main-effect and
interaction contributions within each draw before summarizing their sum:

```{code-cell} ipython3
x_grid = jnp.linspace(df["x"].min(), df["x"].max(), 100)
y_slices = jnp.array([0.2, 0.5, 0.8])
slice_grid = gs.Position(
    {
        "x": jnp.tile(x_grid, len(y_slices)),
        "y": jnp.repeat(y_slices, len(x_grid)),
    },
)
pred = model.predict(
    samples,
    predict=[sx.name, interaction.name],
    newdata=slice_grid,
)
slice_draws = pred[sx.name] + pred[interaction.name]
slice_summary = gs.SamplesSummary({"slice": slice_draws}).to_dataframe().reset_index()
slice_summary["x"] = np.asarray(slice_grid["x"])
slice_summary["y"] = [f"{value:.1f}" for value in np.asarray(slice_grid["y"])]

slice_colors = ["#0072B2", "#D55E00", "#009E73"]
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Blue, orange, and green curves show the posterior x contribution at fixed
      y values of 0.2, 0.5, and 0.8 on the same axes, with matching pointwise
      90 percent credible bands. Their different shapes show how the interaction
      modifies the x effect.
---
(
    p9.ggplot(slice_summary, p9.aes("x", "mean", color="y", fill="y"))
    + p9.geom_ribbon(
        p9.aes(ymin="q_0.05", ymax="q_0.95"),
        alpha=0.15,
        color=None,
    )
    + p9.geom_line()
    + p9.scale_color_manual(values=slice_colors)
    + p9.scale_fill_manual(values=slice_colors)
    + p9.labs(
        x="x",
        y="Contribution to the mean",
        color="Fixed y",
        fill="Fixed y",
        subtitle="f(x) + h(x, y); pointwise 90% credible bands",
    )
    + p9.theme_minimal()
    + p9.theme(figure_size=(8, 4.5), legend_position="top")
)
```

Each color is one slice: read left to right to vary `x` at the fixed `y`
shown in the legend. Compare the colored curves at the same `x`.
The main effect $f(x)$ is the same in all three; differences
come from the interaction $h(x,y)$. The bands describe uncertainty about these
contributions, not new response values.

Each slice omits the intercept and the main effect of `y`. Request `mu.name`
instead for the full mean. Add contributions within each draw before computing
intervals; adding marginal interval endpoints discards their posterior dependence.
For fitted surfaces, see the existing {doc}`tensor example <../notebooks/multivariate/test_tf>`.
