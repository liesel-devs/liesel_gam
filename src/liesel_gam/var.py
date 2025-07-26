from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import tensorflow_probability.substrates.jax.distributions as tfd

from .dist import MultivariateNormalSingular
from .kernel import init_star_ig_gibbs
from .roles import Roles

InferenceTypes = Any
Array = Any
type Term = SmoothTerm | LinearTerm | Intercept


class SmoothTerm(lsl.Var):
    def __init__(
        self,
        basis: Basis | lsl.Var,
        penalty: lsl.Var | Array,
        scale: lsl.Var,
        name: str,
        inference: InferenceTypes = None,
        coef_name: str | None = None,
    ):
        coef_name = f"{name}_coef" if coef_name is None else coef_name

        if not jnp.asarray(basis.value).ndim == 2:
            raise ValueError(f"basis must have 2 dimensions, got {basis.value.ndim}.")

        nbases = jnp.shape(basis.value)[-1]

        prior = lsl.Dist(
            MultivariateNormalSingular,
            loc=0.0,
            scale=scale,
            penalty=penalty,
            penalty_rank=jnp.linalg.matrix_rank(penalty),
        )

        self.scale = scale
        self.nbases = nbases
        self.basis = basis
        self.coef = lsl.Var.new_param(
            jnp.zeros(nbases), prior, inference=inference, name=coef_name
        )
        calc = lsl.Calc(jnp.dot, basis, self.coef)

        super().__init__(calc, name=name)
        self.coef.update()
        self.update()
        self.coef.role = Roles.coef_smooth
        self.role = Roles.term_smooth

    @classmethod
    def new_ig(
        cls,
        basis: Basis | lsl.Var,
        penalty: Array,
        name: str,
        ig_concentration: float = 0.01,
        ig_scale: float = 0.01,
        inference: InferenceTypes = None,
        variance_value: float | None = None,
        variance_name: str | None = None,
        variance_jitter_dist: tfd.Distribution | None = None,
        coef_name: str | None = None,
    ) -> Self:
        variance_name = f"{name}_variance" if variance_name is None else variance_name

        variance = lsl.Var.new_param(
            value=1.0,
            distribution=lsl.Dist(
                tfd.InverseGamma,
                concentration=ig_concentration,
                scale=ig_scale,
            ),
            name=variance_name,
        )
        variance.role = Roles.variance_smooth

        scale = lsl.Var.new_calc(jnp.sqrt, variance, name=f"{variance_name}_root")
        scale.role = Roles.scale_smooth

        if variance_value is None:
            ig_median = variance.dist_node.init_dist().quantile(0.5)  # type: ignore
            variance.value = min(ig_median, 10.0)
        else:
            variance.value = variance_value

        term = cls(
            basis=basis,
            scale=scale,
            penalty=penalty,
            inference=inference,
            name=name,
            coef_name=coef_name,
        )

        variance.inference = gs.MCMCSpec(
            init_star_ig_gibbs,
            kernel_kwargs={"coef": term.coef},
            jitter_dist=variance_jitter_dist,
        )

        return term


class LinearTerm(lsl.Var):
    def __init__(
        self,
        x: lsl.Var | Array,
        name: str,
        distribution: lsl.Dist | None = None,
        inference: InferenceTypes = None,
        add_intercept: bool = False,
        coef_name: str | None = None,
        basis_name: str | None = None,
    ):
        coef_name = f"{name}_coef" if coef_name is None else coef_name
        basis_name = f"B({name})" if basis_name is None else basis_name

        def _matrix(x):
            x = jnp.atleast_1d(x)
            if len(jnp.shape(x)) == 1:
                x = jnp.expand_dims(x, -1)
            if add_intercept:
                ones = jnp.ones(x.shape[0])
                x = jnp.c_[ones, x]
            return x

        if not isinstance(x, lsl.Var):
            x = lsl.Var.new_obs(x, name=f"{name}_input")

        basis = lsl.Var(lsl.TransientCalc(_matrix, x=x), name=basis_name)
        basis.role = Roles.basis

        nbases = jnp.shape(basis.value)[-1]

        self.nbases = nbases
        self.basis = basis
        self.coef = lsl.Var.new_param(
            jnp.zeros(nbases), distribution, inference=inference, name=coef_name
        )
        calc = lsl.Calc(jnp.dot, basis, self.coef)

        super().__init__(calc, name=name)
        self.coef.role = Roles.coef_linear
        self.role = Roles.term_linear


class Intercept(lsl.Var):
    def __init__(
        self,
        name: str,
        value: Array | float = 0.0,
        distribution: lsl.Dist | None = None,
        inference: InferenceTypes = None,
    ) -> None:
        super().__init__(
            value=value, distribution=distribution, name=name, inference=inference
        )
        self.parameter = True
        self.role = Roles.intercept


class Basis(lsl.Var):
    def __init__(
        self,
        value: lsl.Var | lsl.Node,
        basis_fn: Callable[[Array], Array] | Callable[..., Array],
        *args,
        name: str | None = None,
        **kwargs,
    ) -> None:
        try:
            value_ar = jnp.asarray(value.value)
        except AttributeError:
            raise TypeError(f"{value=} should be a liesel.model.Var instance.")

        dtype = value_ar.dtype

        input_shape = jnp.shape(basis_fn(value_ar, *args, **kwargs))
        if len(input_shape):
            k = input_shape[-1]

        def fn(x):
            n = jnp.shape(jnp.atleast_1d(x))[0]
            if len(input_shape) == 2:
                shape = (n, k)
            elif len(input_shape) == 1:
                shape = (n,)
            elif not len(input_shape):
                shape = ()
            else:
                raise RuntimeError(
                    "Return shape of 'basis_fn(value)' must"
                    " have <= dimensions, got {input_shape}"
                )
            result_shape = jax.ShapeDtypeStruct(shape, dtype)
            result = jax.pure_callback(
                basis_fn, result_shape, x, *args, vmap_method="sequential", **kwargs
            )
            return result

        if not value.name:
            raise ValueError(f"{value=} must be named.")

        if name is None:
            name_ = f"B({value.name})"
        else:
            name_ = name

        super().__init__(lsl.Calc(fn, value, _name=name_ + "_calc"), name=name_)
        self.update()
        self.role = Roles.basis
