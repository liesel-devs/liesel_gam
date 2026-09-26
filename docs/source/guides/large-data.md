---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Prepare large data

A representative setup sample can reduce the cost of choosing a smooth basis.
It does not replace the data used for inference. Prepare the basis on training
rows, then encode all required observations with that fitted registry.

## Choose setup rows

This complete setup uses all rows as training data. If you hold out observations,
replace `train_indices` with your training split before creating the basis.

```{code-cell} ipython3
import liesel.model as lsl
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_gam as gam

df = gam.demo_data(n=1000, seed=1)
train_indices = list(range(len(df)))
setup_df = gam.basis_setup_sample(
    df,
    indices=train_indices,
    continuous=["x_nonlin"],
    categorical=["x_cat"],
    n=200,
    seed=2,
)
```

Build the model on these representative rows:

```{code-cell} ipython3
tb = gam.TermBuilder.from_df(setup_df)

mu = gam.AdditivePredictor("mu")
mu += tb.ps("x_nonlin", k=20)
mu += tb.ri("x_cat")

y = lsl.Var.new_obs(
    setup_df["y"].to_numpy(),
    dist=lsl.Dist(tfd.Normal, loc=mu, scale=1.0),
    name="y",
)
model = lsl.Model(y)
```

The sample preserves eligible continuous boundaries and categorical metadata.
It fixes knots, constraints, penalty scaling, and category mappings for this
model. These choices may differ from those obtained by building the basis on
all training rows. Choose a sample large and representative enough for the
intended model. Here the response standard deviation is fixed at 1 solely to
keep this setup example small.

Encode the full observations using the fitted registry:

```{code-cell} ipython3
full_position = tb.registry.observed_position(model, df)
```

`full_position` contains the raw observed variables required by `model`:
`x_nonlin`, encoded `x_cat`, and `y`, each with 1,000 entries. It excludes
unused columns and preserves the fitted category mapping. This call does not
update the model: it still holds the 200 setup rows. Pass the full position to
an inference workflow that supports it, or update the model before full-data MCMC:

```{code-cell} ipython3
model.state = model.update_state(full_position)
```

Confirm that each observed variable now has 1,000 rows:

```{code-cell} ipython3
pd.DataFrame(
    {"rows": {name: len(var.value) for name, var in model.observed.items()}},
)
```

After this update, the model's likelihood uses all 1,000 observations. In a
train/validation workflow, use the training position for fitting and a separate
validation position for evaluation.

## Keep categories represented

Use {func}`~liesel_gam.category_coverage_indices` when designing a split that
should retain every observed category in training. It returns positional indices
for a split policy; it does not perform the split itself. Do not use held-out
responses to select basis settings or tune the model. Known category metadata
does not imply that a level has training observations.

## Consider approximation

The example uses exact basis evaluation. `TermBuilder(..., approximation=True)`
enables the default approximation policy for eligible smooths. Use
{class}`~liesel_gam.ApproximationSpec` to set tolerances and grid limits, and check
whether approximation changes predictions enough to matter for your use case.
Reducing setup cost and reducing repeated evaluation cost are separate decisions.

For batching, optimization losses, and train/validation execution, see
[Liesel's optimization guides on optim-base](https://github.com/liesel-devs/liesel/blob/optim-base/docs/source/optimization.md).
The optimizer API from that branch is the baseline for these guides. See
{doc}`initialization` for selecting coefficients while holding smoothing variances
fixed, or the {doc}`large-data notebook <../notebooks/large_data/test_large_data>`
for an executed minibatch example.
