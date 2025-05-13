from __future__ import annotations

from typing import Any, Self

import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import tensorflow_probability.substrates.jax.distributions as tfd

from .dist import MultivariateNormalSingular
from .kernel import init_star_ig_gibbs
from .roles import Roles

InferenceTypes = Any
Array = Any


class SmoothTerm(lsl.Var):
    def __init__(
        self,
        basis: lsl.Var | Array,
        penalty: Array,
        scale: lsl.Var,
        name: str,
        inference: InferenceTypes = None,
        coef_name: str | None = None,
        basis_name: str | None = None,
    ):
        coef_name = f"{name}_coef" if coef_name is None else coef_name
        basis_name = f"{name}_basis" if basis_name is None else basis_name

        if not isinstance(basis, lsl.Var):
            basis = Basis(basis, name=basis_name)

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
        self.coef.role = Roles.coef_smooth
        self.role = Roles.term_smooth

    @classmethod
    def new_ig(
        cls,
        basis: Basis | lsl.Var | Array,
        penalty: Array,
        name: str,
        ig_concentration: float,
        ig_scale: float,
        inference: InferenceTypes = None,
        variance_value: float | None = None,
        variance_name: str | None = None,
        variance_jitter_dist: tfd.Distribution | None = None,
        coef_name: str | None = None,
        basis_name: str | None = None,
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
            basis_name=basis_name,
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
        distribution: lsl.Dist | None = None,
        inference: InferenceTypes = None,
        add_intercept: bool = False,
        name: str = "",
        coef_name: str | None = None,
        basis_name: str | None = None,
    ):
        coef_name = f"{name}_coef" if coef_name is None else coef_name
        basis_name = f"{name}_basis" if basis_name is None else basis_name

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

        basis = Basis(lsl.TransientCalc(_matrix, x=x), name=basis_name)

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
        value: Array | float = 0.0,
        distribution: lsl.Dist | None = None,
        name: str = "",
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
        value: Array,
        name: str = "",
    ) -> None:
        super().__init__(value=value, name=name)
        self.role = Roles.basis
