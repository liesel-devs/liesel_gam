---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Predict and summarise

Choose the quantity before choosing a plotting function. A term contribution,
a distribution parameter, and a new observation answer different questions.
For general input/shape rules, see the [Liesel prediction guide draft](https://github.com/liesel-devs/liesel/blob/docs/model-goose-guides/docs/source/model-prediction.md).
The {doc}`location-scale tutorial <../tutorials/location-scale>` shows mean and
scale predictions. The group example below is self-contained.

## Choose a prediction

```{list-table}
:header-rows: 1
:widths: 35 65

* - Request
  - Interpretation
* - A term's `name`
  - Its contribution on the predictor's link scale, excluding other terms.
* - `sigma.linear_predictor.name`
  - The full log-standard-deviation predictor, including its intercept.
* - `sigma.name`
  - The positive response standard deviation after the inverse link.
* - `mu.name`
  - The full conditional mean in this normal response model.
```

For an additive predictor with an inverse link, summarize after applying that
link to each draw. For an interaction slice, add its contributions within each
draw before computing intervals; see {doc}`interactions`.

## Reuse fitted terms

Use the fitted model to retain knots, constraints, and category mappings. Do not
rebuild a term on prediction data: that can change the model's meaning. Stay
within a scientifically defensible domain; a smooth extrapolation is not evidence
that an effect continues outside the observed range.

For a model with categorical inputs such as a random intercept, pass labels, for
example `newdata={"group": ["a", "b"]}`. The fitted `CatVar` converts labels
through its training mapping. Unknown labels are not automatically new random
effects. Levels known to the setup but without observations have no direct data
support; their uncertainty depends on the prior and any spatial structure.

## Predict for groups

For an existing group, prediction uses its posterior effect draws. For a group
without observations, prediction must include between-group variation as well
as uncertainty in its scale when that scale is estimated. A zero group effect
answers a different question;
with a nonlinear inverse link it is not the average over new groups.

Declare an unobserved level before fitting. Here groups `a` and `b` have
observations, while `unobserved` has none. Fix the group-effect standard
deviation at 1 and response standard deviation at 0.3 for this example.
Use a known population mean of zero (`intercept=False`), so the group effects
are deviations from zero. This keeps the example focused on group prediction.

```{code-cell} ipython3
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_gam as gam

rng = np.random.default_rng(8)
training = pd.DataFrame({"group": np.repeat(["a", "b"], 30)})
training["y"] = np.repeat([-0.5, 0.5], 30) + rng.normal(0, 0.3, 60)
training["group"] = pd.Categorical(
    training["group"], categories=["a", "b", "unobserved"]
)
```

Build the model with the known population mean of zero:

```{code-cell} ipython3
group_tb = gam.TermBuilder.from_df(training)
group_effect = group_tb.ri("group", scale=1.0)
mu = gam.AdditivePredictor("mu", intercept=False)
mu += group_effect

y = lsl.Var.new_obs(
    training["y"].to_numpy(),
    dist=lsl.Dist(tfd.Normal, loc=mu, scale=0.3),
    name="y",
)
model = lsl.Model(y)
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: A normal response depends on random group effects with a fixed prior scale
      and population mean zero, including one unobserved level.
---
model.plot()
```

Prediction assumes posterior draws `group_samples` from this model. Follow the
{doc}`smooth-curve tutorial <../tutorials/smooth-curve>` for the sampling workflow.
Here the default IWLS updates use burn-in without adaptation.

```{code-cell} ipython3
:tags: [remove-cell]

results = gs.LieselMCMC(model).run_for_epochs(
    seed=8,
    num_chains=4,
    adaptation=0,
    burnin=500,
    posterior=1000,
    show_progress=False,
)
group_samples = results.get_posterior_samples()
```

```{code-cell} ipython3
group_draws = group_effect.predict(
    group_samples,
    newdata={"group": ["a", "unobserved", "unobserved"]},
)
group_summary = gs.SamplesSummary({"group_effect": group_draws}).to_dataframe()
group_summary["group"] = ["a", "unobserved", "unobserved"]
```

```{code-cell} ipython3
group_summary[["group", "mean", "sd", "q_0.05", "q_0.95", "ess_bulk", "rhat"]].round(3)
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Group-effect intervals for observed groups a and b and the wider prior-informed
      interval for the unobserved group.
---
gam.plot_forest(group_effect, group_samples)
```

Repeated labels share one group effect within each draw. For an IID random
intercept with no observations, that effect is informed by the group-level
prior and its scale, fixed at 1 in this example. Do not draw an independent effect for every
observation in the same new group. Spatial effects instead depend on their
specified neighbourhood structure.

Undeclared labels are rejected by the usual fitted category mapping. A deliberately
configured {class}`~liesel_gam.CatVar` catch-all category has different semantics;
it does not create an independent random effect for each new label.

For mean curves and response intervals in a distributional GAM, see the
{doc}`location-scale tutorial <../tutorials/location-scale>`. General response
simulation, sample shapes, and conditioning rules belong to the
[Liesel simulation guide draft](https://github.com/liesel-devs/liesel/blob/docs/model-goose-guides/docs/source/model-simulation.md).

## Use effect helpers

Use {func}`~liesel_gam.plot_1d_smooth` for a centered one-dimensional smooth,
{func}`~liesel_gam.plot_2d_smooth` for a surface,
{func}`~liesel_gam.plot_forest` for linear or group effects, and
{func}`~liesel_gam.plot_regions` for regional effects. Their corresponding
`summarise_*` functions return tables for your own plotnine plots.

For a varying-coefficient term, `plot_1d_smooth` without `newdata` shows the
coefficient curve at multiplier one. Supply both inputs to show the multiplied
contribution. This distinction matters when interpreting the vertical axis.
