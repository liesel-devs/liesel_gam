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


class UserVar(lsl.Var):
    @classmethod
    def new_calc(cls, *args, **kwargs) -> None:  # type: ignore
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )

    @classmethod
    def new_obs(cls, *args, **kwargs) -> None:  # type: ignore
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )

    @classmethod
    def new_param(cls, *args, **kwargs) -> None:  # type: ignore
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )

    @classmethod
    def new_value(cls, *args, **kwargs) -> None:  # type: ignore
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )


class Term(UserVar):
    def __init__(
        self,
        basis: Basis,
        penalty: lsl.Var | Array,
        scale: lsl.Var | Array,
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
        basis: Basis,
        penalty: Array,
        name: str,
        ig_concentration: float = 1.0,
        ig_scale: float = 0.005,
        inference: InferenceTypes = None,
        variance_value: float = 100.0,
        variance_name: str | None = None,
        variance_jitter_dist: tfd.Distribution | None = None,
        coef_name: str | None = None,
    ) -> Self:
        variance_name = variance_name or f"{name}_variance"

        variance = lsl.Var.new_param(
            value=variance_value,
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


SmoothTerm = Term


class LinearTerm(Term):
    """Kept for backwards-compatibility of the interface."""

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
        if not isinstance(x, lsl.Var):
            x = lsl.Var.new_obs(x, name=f"{name}_input")

        if not x.name:
            # to ensure sensible basis name
            raise ValueError(f"{x=} must be named.")

        coef_name = coef_name or f"{name}_coef"
        basis_name = basis_name or f"B({name})"
        basis = Basis.new_linear(value=x, name=basis_name, add_intercept=add_intercept)

        nbases = jnp.shape(basis.value)[-1]
        penalty = jnp.eye(nbases)
        # just a temporary variable to satisfy the api of Term
        scale = lsl.Var(1.0, name=f"_{name}_scale_tmp")

        super().__init__(
            basis=basis,
            penalty=penalty,
            scale=scale,
            name=name,
            inference=inference,
            coef_name=coef_name,
        )
        self.coef.dist_node = distribution
        self.coef.role = Roles.coef_linear
        self.role = Roles.term_linear


class LinearTerm2(Term):
    """New version of LinearTerm, with interface consistent with the Term base class."""

    def __init__(
        self,
        value: lsl.Var | lsl.Node | Array,
        name: str,
        scale: lsl.Var | Array = 1000.0,
        inference: InferenceTypes = None,
        add_intercept: bool = False,
        coef_name: str | None = None,
        basis_name: str | None = None,
    ):
        if not isinstance(value, lsl.Var | lsl.Node):
            x: lsl.Var | lsl.Node = lsl.Var.new_obs(value, name=f"{name}_input")
        else:
            x = value

        if not x.name:
            # to ensure sensible basis name
            raise ValueError(f"{value=} must be named.")

        coef_name = coef_name or f"{name}_coef"
        basis_name = basis_name or f"B({name})"
        basis = Basis.new_linear(value=x, name=basis_name, add_intercept=add_intercept)

        nbases = jnp.shape(basis.value)[-1]
        penalty = jnp.eye(nbases)
        super().__init__(
            basis=basis,
            penalty=penalty,
            coef_name=coef_name,
            name=name,
            inference=inference,
            scale=scale,
        )

        self.coef.role = Roles.coef_linear
        self.role = Roles.term_linear


class Intercept(UserVar):
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


def make_callback(function, input_shape, dtype, *args, **kwargs):
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
                f" have <= dimensions, got {input_shape}"
            )
        result_shape = jax.ShapeDtypeStruct(shape, dtype)
        result = jax.pure_callback(
            function, result_shape, x, *args, vmap_method="sequential", **kwargs
        )
        return result

    return fn


class Basis(UserVar):
    def __init__(
        self,
        value: lsl.Var | lsl.Node | Array,
        basis_fn: Callable[[Array], Array] | Callable[..., Array] = lambda x: x,
        name: str | None = None,
        xname: str | None = None,
        use_callback: bool = True,
        cache_basis: bool = True,
        **basis_kwargs,
    ) -> None:
        if isinstance(value, lsl.Var | lsl.Node):
            value_var = value
        else:
            if not xname:
                raise ValueError(
                    "When supplying an array to `value`, `xname` must be defined."
                )
            value_var = lsl.Var.new_obs(value, name=xname)

        if not value_var.name:
            # to ensure sensible basis name
            raise ValueError(f"{value=} must be named.")

        if use_callback:
            value_ar = jnp.asarray(value_var.value)
            basis_ar = basis_fn(value_ar, **basis_kwargs)
            dtype = basis_ar.dtype
            input_shape = jnp.shape(basis_ar)
            fn = make_callback(basis_fn, input_shape, dtype, **basis_kwargs)
        else:
            fn = basis_fn

        name_ = name or f"B({value_var.name})"

        if cache_basis:
            calc = lsl.Calc(fn, value_var, _name=name_ + "_calc")
        else:
            calc = lsl.TransientCalc(fn, value_var, _name=name_ + "_calc")

        super().__init__(calc, name=name_)
        self.update()
        self.role = Roles.basis
        self.observed = True
        self.x = value_var
        basis_shape = jnp.shape(self.value)
        if len(basis_shape) >= 1:
            self.nbases = basis_shape[-1]
        else:
            self.nbases = 1  # scalar case

    @classmethod
    def new_linear(
        cls,
        value: lsl.Var | lsl.Node | Array,
        name: str | None = None,
        xname: str | None = None,
        add_intercept: bool = False,
    ):
        def as_matrix(x):
            x = jnp.atleast_1d(x)
            if len(jnp.shape(x)) == 1:
                x = jnp.expand_dims(x, -1)
            if add_intercept:
                ones = jnp.ones(x.shape[0])
                x = jnp.c_[ones, x]
            return x

        basis = cls(
            value=value,
            basis_fn=as_matrix,
            name=name,
            xname=xname,
            use_callback=False,
            cache_basis=False,
            includes_intercept=add_intercept,
        )

        return basis
