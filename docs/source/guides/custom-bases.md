---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Use a custom basis

Use {meth}`liesel_gam.TermBuilder.f <liesel_gam.TermBuilder.f>` when the built-in constructors do not
represent your effect. Supply a basis function and penalty; Liesel-GAM adds the
coefficient prior and inference specification.

## Define the basis

A basis function takes a two-dimensional covariate array and returns one row per
observation, with one column per coefficient. Here a polynomial basis illustrates
the interface. For ordinary polynomial regression, use `tb.lin("x + {x**2}")`.

The example uses JAX as `jnp`, Liesel-GAM as `gam`, and a builder `tb` for
`gam.demo_data(n=100, seed=1)`. See the
{doc}`smooth-curve tutorial <../tutorials/smooth-curve>` for builder setup.

```{code-cell} ipython3
import jax.numpy as jnp

import liesel_gam as gam
```

```{code-cell} ipython3
:tags: [remove-cell]

df = gam.demo_data(n=100, seed=1)
tb = gam.TermBuilder.from_df(df)
```

```{code-cell} ipython3
def polynomial_basis(x):
    values = x[:, 0]
    return jnp.column_stack((jnp.ones_like(values), values, values**2))


term = tb.f(
    "x_nonlin",
    basis_fn=polynomial_basis,
    penalty=jnp.eye(3),
    use_callback=False,
    row_wise=True,
    name="polynomial",
)
term.constrain("sumzero_term")

custom_mu = gam.AdditivePredictor("mu")
custom_mu += term
```

The identity penalty assigns equal prior precision to the original coefficients.
That is a modeling choice, not the usual P-spline difference penalty. A custom
penalty must match the coefficient dimension and be positive semidefinite.

Plot this term on its own to inspect the basis and coefficient prior without
the rest of the model:

```{code-cell} ipython3
---
mystnb:
  image:
    alt: The polynomial term combines its basis and coefficient vector; a smoothing
      scale controls the coefficient prior.
---
term.plot(width=10, height=6)
```

The basis maps the covariate to the term, while the scale controls its
coefficient prior. This graph shows dependencies; it does not display the
constraint matrix or establish that the chosen penalty is appropriate.

## Identify the effect

Apply constraints before building and fitting the model. `sumzero_term` centers
the effect over the training observations, separating it from the predictor's
intercept. `sumzero_coef` instead constrains the coefficients to sum to zero;
these are generally different constraints. A custom constraint matrix expresses
$A\beta = 0$; see {meth}`constrain <liesel_gam.StrctTerm.constrain>`.

The fitted transformation is reused for prediction, so the effect need not sum
to zero on a new covariate grid. Re-centering at prediction time would change the
intercept/effect decomposition.

## Keep prediction valid

The example uses JAX operations and `use_callback=False` so the basis can be
evaluated inside JAX transformations. `row_wise=True` states that each output
row depends only on the corresponding input row. Do not set it for a function
that recomputes a mean or other sample-wide statistic on every call.

If a basis needs fitted quantities such as knots or training means, fix them at
setup and reuse them. With `use_callback=True`, a Python callback can evaluate the fitted basis on
new prediction rows, but the basis must stay constant during estimation and JAX
cannot differentiate through its inputs.
Formulaic-based linear design matrices likewise require care with dynamic inputs;
native smooth bases and arbitrary Python callbacks do not have identical JAX
capabilities. See {class}`Basis <liesel_gam.Basis>` for callback and caching rules.
