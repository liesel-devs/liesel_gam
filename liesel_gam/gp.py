from collections.abc import Callable
from typing import Generic, ParamSpec, TypeVar

import jax.numpy as jnp
import liesel.model as lsl
import tensorflow_probability.substrates.jax.math.psd_kernels as tpk
from jax import Array
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalDiag,
    MultivariateNormalFullCovariance,
)

VarOrArray = lsl.Var | Array
P = ParamSpec("P")
TKernel = TypeVar("TKernel", bound=tpk.PositiveSemidefiniteKernel)


class GaussianProcessTerm(lsl.Var, Generic[P, TKernel]):
    def __init__(
        self,
        inputs: VarOrArray,
        kernel_constructor: Callable[P, TKernel],
        kernel_diag: float = 1e-3,
        name: str = "",
        *kernel_args: P.args,
        **kernel_kwargs: P.kwargs,
    ):
        if not isinstance(inputs, lsl.Var):
            input_name = f"{name}_inputs"
            inputs = lsl.Var.new_obs(inputs, name=input_name)

        def kernel_matrix_fn(
            inputs: Array, *kargs: P.args, **kkwargs: P.kwargs
        ) -> Array:
            krn = kernel_constructor(*kargs, **kkwargs)
            mat = krn.matrix(inputs, inputs)
            mat = mat.at[jnp.diag_indices_from(mat)].add(kernel_diag)
            return mat

        c00 = lsl.Var.new_calc(
            kernel_matrix_fn,
            inputs,
            *kernel_args,
            **kernel_kwargs,
            name=f"{name}_cov_mat",
            distribution=None,
            _needs_seed=False,
            _update_on_init=True,
        )

        prior = lsl.Dist(
            MultivariateNormalFullCovariance,
            loc=jnp.zeros(c00.value.shape[0]),
            covariance_matrix=c00,
        )

        super().__init__(
            jnp.zeros(inputs.value.shape[0]),
            prior,
            name=name,
        )
        self._parameter = True

        self._prior_cov_mat = c00
        self._kernel_constructor = kernel_constructor
        self._kernel_diag = kernel_diag
        self._kernel_args = kernel_args
        self._kernel_kwargs = kernel_kwargs
        self._inputs = inputs
        self._pred_dict_counter = 0
        self._pred_var_counter = 0

    def predict_dist(
        self, predict_inputs: lsl.Var, name: str, full_cov: bool, make_param_vars: bool
    ) -> lsl.Dist:
        def pred_params(
            prior_cov: Array,
            inputs: Array,
            inputs_pred: Array,
            values: Array,
            *kargs: P.args,
            **kkwargs: P.kwargs,
        ):
            krn = self._kernel_constructor(*kargs, **kkwargs)
            c01 = krn.matrix(inputs, inputs_pred)
            c11 = krn.matrix(inputs_pred, inputs_pred)
            c11 = c11.at[jnp.diag_indices_from(c11)].add(self._kernel_diag)

            pred_mean = 0 + c01.T @ jnp.linalg.solve(prior_cov, values - 0)
            pred_cov = c11 - c01.T @ jnp.linalg.solve(prior_cov, c01)
            if not full_cov:
                pred_cov = jnp.diag(pred_cov)
            return pred_mean, pred_cov

        if not make_param_vars:

            def pred_dist(
                prior_cov: Array,
                inputs: Array,
                inputs_pred: Array,
                values: Array,
                *kargs: P.args,
                **kkwargs: P.kwargs,
            ):
                pred_mean, pred_cov = pred_params(
                    prior_cov, inputs, inputs_pred, values, *kargs, **kkwargs
                )

                if full_cov:
                    return MultivariateNormalFullCovariance(
                        loc=pred_mean,
                        covariance_matrix=pred_cov,
                    )
                else:
                    return MultivariateNormalDiag(
                        loc=pred_mean, scale_diag=jnp.sqrt(pred_cov)
                    )

            dist = lsl.Dist(
                pred_dist,
                self._prior_cov_mat,
                self._inputs,
                predict_inputs,
                self,
                *self._kernel_args,
                **self._kernel_kwargs,
                _name="",
                _needs_seed=False,
            )
            return dist
        else:
            params = lsl.Var.new_calc(
                pred_params,
                self._prior_cov_mat,
                self._inputs,
                predict_inputs,
                self,
                *self._kernel_args,
                **self._kernel_kwargs,
                name=f"{name}_params",
                distribution=None,
                _needs_seed=False,
                _update_on_init=True,
            )

            pred_mean = lsl.Var.new_calc(
                lambda x: x[0],
                params,
                name=f"{name}_mean",
            )

            pred_cov = lsl.Var.new_calc(
                lambda x: x[1],
                params,
                name=f"{name}_cov",
            )

            def dist_fn(pred_mean, pred_cov):
                if full_cov:
                    return MultivariateNormalFullCovariance(
                        loc=pred_mean,
                        covariance_matrix=pred_cov,
                    )
                else:
                    return MultivariateNormalDiag(
                        loc=pred_mean, scale_diag=jnp.sqrt(pred_cov)
                    )

            dist = lsl.Dist(dist_fn, pred_mean=pred_mean, pred_cov=pred_cov)
            return dist

    def predict_var(
        self,
        predict_inputs: VarOrArray,
        name: str = "",
        full_cov: bool = False,
        make_param_vars: bool = False,
    ):
        if not name:
            name = f"{self.name}_predict_{self._pred_var_counter}"
            self._pred_var_counter += 1

        if not isinstance(predict_inputs, lsl.Var):
            input_name = f"{name}_inputs"
            predict_inputs = lsl.Var.new_obs(predict_inputs, name=input_name)

        n = predict_inputs.value.shape[0]

        dist = self.predict_dist(predict_inputs, name, full_cov, make_param_vars)
        var = lsl.Var.new_param(jnp.zeros(n), dist, name=f"{name}")
        return var
