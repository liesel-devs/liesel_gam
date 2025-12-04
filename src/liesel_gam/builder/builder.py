from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Literal, get_args

import formulaic as fo
import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd
import smoothcon as scon
from ryp import r, to_py, to_r

from ..var import (
    Basis,
    LinBasis,
    LinTerm,
    MRFBasis,
    MRFSpec,
    MRFTerm,
    RITerm,
    ScaleIG,
    Term,
    TPTerm,
    VarIGPrior,
)
from .registry import CategoryMapping, PandasRegistry

InferenceTypes = Any

Array = jax.Array
ArrayLike = jax.typing.ArrayLike

BasisTypes = Literal["tp", "ts", "cr", "cs", "cc", "bs", "ps", "cp", "gp"]


logger = logging.getLogger(__name__)


def _validate_bs(bs):
    if isinstance(bs, str):
        bs = [bs]
    allowed = get_args(BasisTypes)
    for bs_str in bs:
        if bs_str not in allowed:
            raise ValueError(f"Allowed values for 'bs' are: {allowed}; got {bs=}.")


def _margin_penalties(smooth: scon.SmoothCon):
    """Extracts the marginal penalty matrices from a ti() smooth."""
    # this should go into smoothcon, but it works here for now
    r(
        f"penalties_list <- lapply({smooth._smooth_r_name}"
        "[[1]]$margin, function(x) x$S[[1]])"
    )
    pens = to_py("penalties_list")
    return [pen.to_numpy() for pen in pens]


def _tp_penalty(K1, K2) -> Array:
    """Computes the full tensor product penalty from the marginals."""
    # this should go into smoothcon, but it works here for now
    D1 = np.shape(K1)[1]
    D2 = np.shape(K2)[1]
    I1 = np.eye(D1)
    I2 = np.eye(D2)

    return jnp.asarray(jnp.kron(K1, I2) + jnp.kron(I1, K2))


def labels_to_integers(newdata: dict, mappings: dict[str, CategoryMapping]) -> dict:
    # replace categorical inputs with their index representation
    # create combined input matrices from individual variables, if desired
    newdata = newdata.copy()

    # replace categorical variables by their integer representations
    for name, mapping in mappings.items():
        if name in newdata:
            newdata[name] = mapping.labels_to_integers(newdata[name])

    return newdata


def assert_intercept_in_spec(spec: fo.ModelSpec) -> fo.ModelSpec:
    """
    Uses the degrees of the terms in the spec's formula to find intercepts.
    The degree of a term indicates how many columns of the input data are referenced
    by the term, so a degree of zero can be used to identify an intercept.
    """
    terms = list(spec.formula)
    terms_with_degree_zero = [term for term in terms if term.degree == 0]

    if len(terms_with_degree_zero) > 1:
        raise RuntimeError(f"Too many intercepts: {len(terms_with_degree_zero)}.")
    if len(terms_with_degree_zero) == 0:
        raise RuntimeError(
            "No intercept found in formula. Did you explicitly remove an "
            "intercept by including '0' or '-1'? This breaks model matrix setup."
        )

    return spec


def validate_formula(formula: str) -> None:
    if "~" in formula:
        raise ValueError("'~' in formulas is not supported.")

    terms = ["".join(x.split()) for x in formula.split("+")]
    for term in terms:
        if term == "1":
            raise ValueError(
                "Using '1 +' is not supported. To add an intercept, use the "
                "argument 'include_intercept'."
            )
        if term == "0" or term == "-1":
            raise ValueError(
                "Using '0 +' or '-1' is not supported. Intercepts are not included "
                "by default and can be added manually with the argument "
                "'include_intercept'."
            )


def validate_penalty_order(penalty_order: int):
    if not isinstance(penalty_order, int):
        raise TypeError(
            f"'penalty_order' must be int or None, got {type(penalty_order)}"
        )
    if not penalty_order > 0:
        raise ValueError(f"'penalty_order' must be >0, got {penalty_order}")


class BasisBuilder:
    def __init__(
        self, registry: PandasRegistry, names: NameManager | None = None
    ) -> None:
        self.registry = registry
        self.mappings: dict[str, CategoryMapping] = {}
        self.names = NameManager() if names is None else names

    @property
    def data(self) -> pd.DataFrame:
        return self.registry.data

    def basis(
        self,
        *x: str,
        basis_fn: Callable[[Array], Array] = lambda x: x,
        use_callback: bool = True,
        cache_basis: bool = True,
        penalty: ArrayLike | lsl.Value | None = None,
        basis_name: str = "B",
    ) -> Basis:
        if isinstance(penalty, lsl.Value):
            penalty.value = jnp.asarray(penalty.value)
        elif penalty is not None:
            penalty = jnp.asarray(penalty)

        x_vars = []
        for x_name in x:
            x_var = self.registry.get_numeric_obs(x_name)
            x_vars.append(x_var)

        Xname = self.registry.prefix + ",".join(x)

        Xvar = lsl.TransientCalc(
            lambda *x: jnp.column_stack(x),
            *x_vars,
            _name=Xname,
        )

        basis = Basis(
            value=Xvar,
            basis_fn=basis_fn,
            name=self.names.create_lazily(basis_name + "(" + Xname + ")"),
            use_callback=use_callback,
            cache_basis=cache_basis,
            penalty=jnp.asarray(penalty),
        )

        return basis

    def te(
        self,
        x1: str,
        x2: str,
        bs: BasisTypes | tuple[BasisTypes, BasisTypes] = "ps",
        k: int | tuple[int, int] = -1,
        m: str = "NA",
        knots: ArrayLike | None = None,
        basis_name: str = "B",
    ) -> Basis:
        if knots is not None:
            knots = np.asarray(knots)
        _validate_bs(bs)
        absorb_cons: bool = True
        diagonal_penalty: bool = True
        scale_penalty: bool = True

        x1_array = jnp.asarray(self.registry.data[x1].to_numpy())
        x2_array = jnp.asarray(self.registry.data[x2].to_numpy())

        if isinstance(k, int):
            if k == -1:
                k1: str | int = "NA"
                k2: str | int = "NA"
            else:
                k1 = k
                k2 = k
        elif len(k) == 2:
            k1, k2 = k
        else:
            raise ValueError(f"{k=} not supported.")

        if isinstance(bs, str):
            bs_arg = f"'{bs}'"
        elif len(bs) == 2:
            bs_arg = f"c('{bs[0]}', '{bs[1]}')"
        else:
            raise ValueError(f"{bs=} not supported.")

        spec = f"te({x1}, {x2}, bs={bs_arg}, k=c({k1}, {k2}), m={m})"

        smooth = scon.SmoothCon(
            spec,
            data={x1: x1_array, x2: x2_array},
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )

        x1_var = self.registry.get_numeric_obs(x1)
        x2_var = self.registry.get_numeric_obs(x2)
        x1x2_name = self.registry.prefix + f"{x1},{x2}"
        x1_x2_var = lsl.TransientCalc(
            lambda x1, x2: jnp.stack((x1, x2), axis=1),
            x1=x1_var,
            x2=x2_var,
            _name=x1x2_name,
        )

        K1 = smooth.single_penalty(0, 0)
        K2 = smooth.single_penalty(0, 1)

        basis = Basis(
            x1_x2_var,
            name=self.names.create(basis_name) + "(" + x1 + "," + x2 + ")",
            basis_fn=lambda x: jnp.asarray(smooth.predict({x1: x[:, 0], x2: x[:, 1]})),
            penalty=K1 + K2,
            use_callback=True,
            cache_basis=True,
        )
        basis._constraint = "absorbed_via_mgcv"
        return basis

    def ti(
        self,
        x1: str,
        x2: str,
        bs: BasisTypes | tuple[BasisTypes, BasisTypes] = "ps",
        k: int | tuple[int, int] = -1,
        m: str = "NA",
        knots: ArrayLike | None = None,
        basis_name: str = "B",
    ) -> Basis:
        if knots is not None:
            knots = np.asarray(knots)
        _validate_bs(bs)
        absorb_cons: bool = True
        diagonal_penalty: bool = True
        scale_penalty: bool = True

        x1_array = jnp.asarray(self.registry.data[x1].to_numpy())
        x2_array = jnp.asarray(self.registry.data[x2].to_numpy())

        if isinstance(k, int):
            if k == -1:
                k1: str | int = "NA"
                k2: str | int = "NA"
            else:
                k1 = k
                k2 = k
        elif len(k) == 2:
            k1, k2 = k
        else:
            raise ValueError(f"{k=} not supported.")

        if isinstance(bs, str):
            bs_arg = f"'{bs}'"
        elif len(bs) == 2:
            bs_arg = f"c('{bs[0]}', '{bs[1]}')"
        else:
            raise ValueError(f"{bs=} not supported.")

        spec = f"ti({x1}, {x2}, bs={bs_arg}, k=c({k1}, {k2}), m={m})"

        smooth = scon.SmoothCon(
            spec,
            data={x1: x1_array, x2: x2_array},
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )

        x1_var = self.registry.get_numeric_obs(x1)
        x2_var = self.registry.get_numeric_obs(x2)

        x1x2_name = self.registry.prefix + f"{x1},{x2}"

        x1_x2_var = lsl.TransientCalc(
            lambda x1, x2: jnp.c_[x1, x2],
            x1=x1_var,
            x2=x2_var,
            _name=x1x2_name,
        )

        penalty = _tp_penalty(*_margin_penalties(smooth))

        basis = Basis(
            x1_x2_var,
            name=self.names.create(basis_name) + "(" + x1 + "," + x2 + ")",
            basis_fn=lambda x: jnp.asarray(smooth.predict({x1: x[:, 0], x2: x[:, 1]})),
            penalty=penalty,
            use_callback=True,
            cache_basis=True,
        )
        basis._constraint = "absorbed_via_mgcv"
        return basis

    def ps(
        self,
        x: str,
        *,
        k: int,
        basis_degree: int = 3,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
    ) -> Basis:
        validate_penalty_order(penalty_order)
        if knots is not None:
            knots = np.asarray(knots)

        spec = f"s({x}, bs='ps', k={k}, m=c({basis_degree - 1}, {penalty_order}))"
        x_array = jnp.asarray(self.registry.data[x].to_numpy())
        smooth = scon.SmoothCon(
            spec,
            data={x: x_array},
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )

        x_var = self.registry.get_numeric_obs(x)
        basis = Basis(
            x_var,
            name=self.names.create_lazily(basis_name + "(" + x_var.name + ")"),
            basis_fn=lambda x_: jnp.asarray(smooth.predict({x: x_})),
            penalty=smooth.penalty,
            use_callback=True,
            cache_basis=True,
        )

        if absorb_cons:
            basis._constraint = "absorbed_via_mgcv"
        return basis

    def cr(
        self,
        x: str,
        *,
        k: int,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
    ) -> Basis:
        validate_penalty_order(penalty_order)
        if knots is not None:
            knots = np.asarray(knots)
        spec = f"s({x}, bs='cr', k={k}, m=c({penalty_order}))"
        x_array = jnp.asarray(self.registry.data[x].to_numpy())
        smooth = scon.SmoothCon(
            spec,
            data={x: x_array},
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )

        x_var = self.registry.get_numeric_obs(x)
        basis = Basis(
            x_var,
            name=self.names.create_lazily(basis_name + "(" + x_var.name + ")"),
            basis_fn=lambda x_: jnp.asarray(smooth.predict({x: x_})),
            penalty=smooth.penalty,
            use_callback=True,
            cache_basis=True,
        )

        if absorb_cons:
            basis._constraint = "absorbed_via_mgcv"
        return basis

    def cs(
        self,
        x: str,
        *,
        k: int,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
    ) -> Basis:
        """
        s(x,bs="cs") specifies a penalized cubic regression spline which has had its
        penalty modified to shrink towards zero at high enough smoothing parameters (as
        the smoothing parameter goes to infinity a normal cubic spline tends to a
        straight line.)
        """
        validate_penalty_order(penalty_order)
        if knots is not None:
            knots = np.asarray(knots)
        spec = f"s({x}, bs='cs', k={k}, m=c({penalty_order}))"
        x_array = jnp.asarray(self.registry.data[x].to_numpy())
        smooth = scon.SmoothCon(
            spec,
            data={x: x_array},
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )

        x_var = self.registry.get_numeric_obs(x)
        basis = Basis(
            x_var,
            name=self.names.create_lazily(basis_name + "(" + x_var.name + ")"),
            basis_fn=lambda x_: jnp.asarray(smooth.predict({x: x_})),
            penalty=smooth.penalty,
            use_callback=True,
            cache_basis=True,
        )

        if absorb_cons:
            basis._constraint = "absorbed_via_mgcv"
        return basis

    def cc(
        self,
        x: str,
        *,
        k: int,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
    ) -> Basis:
        validate_penalty_order(penalty_order)
        if knots is not None:
            knots = np.asarray(knots)
        spec = f"s({x}, bs='cc', k={k}, m=c({penalty_order}))"
        x_array = jnp.asarray(self.registry.data[x].to_numpy())
        smooth = scon.SmoothCon(
            spec,
            data={x: x_array},
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )

        x_var = self.registry.get_numeric_obs(x)
        basis = Basis(
            x_var,
            name=self.names.create_lazily(basis_name + "(" + x_var.name + ")"),
            basis_fn=lambda x_: jnp.asarray(smooth.predict({x: x_})),
            penalty=smooth.penalty,
            use_callback=True,
            cache_basis=True,
        )

        if absorb_cons:
            basis._constraint = "absorbed_via_mgcv"
        return basis

    def bs(
        self,
        x: str,
        *,
        k: int,
        basis_degree: int = 3,
        penalty_order: int | Sequence[int] = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
    ) -> Basis:
        """
        The integrated square of the m[2]th derivative is used as the penalty. So
        m=c(3,2) is a conventional cubic spline. Any further elements of m, after the
        first 2, define the order of derivative in further penalties. If m is supplied
        as a single number, then it is taken to be m[1] and m[2]=m[1]-1, which is only a
        conventional smoothing spline in the m=3, cubic spline case.
        """
        if knots is not None:
            knots = np.asarray(knots)
        if isinstance(penalty_order, int):
            validate_penalty_order(penalty_order)
            penalty_order_seq: Sequence[str] = [str(penalty_order)]
        else:
            [validate_penalty_order(p) for p in penalty_order]
            penalty_order_seq = [str(p) for p in penalty_order]

        spec = (
            f"s({x}, bs='bs', k={k}, "
            f"m=c({basis_degree}, {', '.join(penalty_order_seq)}))"
        )
        x_array = jnp.asarray(self.registry.data[x].to_numpy())
        smooth = scon.SmoothCon(
            spec,
            data={x: x_array},
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )

        x_var = self.registry.get_numeric_obs(x)
        basis = Basis(
            x_var,
            name=self.names.create_lazily(basis_name + "(" + x_var.name + ")"),
            basis_fn=lambda x_: jnp.asarray(smooth.predict({x: x_})),
            penalty=smooth.penalty,
            use_callback=True,
            cache_basis=True,
        )

        if absorb_cons:
            basis._constraint = "absorbed_via_mgcv"
        return basis

    def cp(
        self,
        x: str,
        *,
        k: int,
        basis_degree: int = 3,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
    ) -> Basis:
        validate_penalty_order(penalty_order)
        if knots is not None:
            knots = np.asarray(knots)
        spec = f"s({x}, bs='cp', k={k}, m=c({basis_degree - 1}, {penalty_order}))"
        x_array = jnp.asarray(self.registry.data[x].to_numpy())
        smooth = scon.SmoothCon(
            spec,
            data={x: x_array},
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )

        x_var = self.registry.get_numeric_obs(x)
        basis = Basis(
            x_var,
            name=self.names.create_lazily(basis_name + "(" + x_var.name + ")"),
            basis_fn=lambda x_: jnp.asarray(smooth.predict({x: x_})),
            penalty=smooth.penalty,
            use_callback=True,
            cache_basis=True,
        )

        if absorb_cons:
            basis._constraint = "absorbed_via_mgcv"
        return basis

    def s(
        self,
        *x: str,
        k: int,
        bs: BasisTypes,
        m: str = "NA",
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
    ) -> Basis:
        if knots is not None:
            knots = np.asarray(knots)
        _validate_bs(bs)
        bs_arg = f"'{bs}'"
        spec = f"s({','.join(x)}, bs={bs_arg}, k={k}, m={m})"

        obs_vars = {}
        for xname in x:
            obs_vars[xname] = self.registry.get_numeric_obs(xname)
        obs_values = {k: np.asarray(v.value) for k, v in obs_vars.items()}

        smooth = scon.SmoothCon(
            spec,
            data=pd.DataFrame.from_dict(obs_values),
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )

        xname = ",".join([v.name for v in obs_vars.values()])

        if len(obs_vars) > 1:
            xvar: lsl.Var | lsl.TransientCalc = (
                lsl.TransientCalc(  # for memory-efficiency
                    lambda *args: jnp.vstack(args).T,
                    *list(obs_vars.values()),
                    _name=self.names.create(xname),
                )
            )
        else:
            xvar = obs_vars[xname]

        def basis_fn(x):
            df = pd.DataFrame(x, columns=list(obs_vars))
            return jnp.asarray(smooth.predict(df))

        basis = Basis(
            xvar,
            name=self.names.create_lazily(basis_name + "(" + xname + ")"),
            basis_fn=basis_fn,
            penalty=smooth.penalty,
            use_callback=True,
            cache_basis=True,
        )
        if absorb_cons:
            basis._constraint = "absorbed_via_mgcv"
        return basis

    def tp(
        self,
        *x: str,
        k: int,
        penalty_order: int | None = None,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
        remove_null_space_completely: bool = False,
    ) -> Basis:
        """
        For penalty_order:
        m = penalty_order
        Quote from MGCV docs
        The default is to set m (the order of derivative in the thin plate spline
        penalty) to the smallest value satisfying 2m > d+1 where d is the number of
        covariates of the term: this yields ‘visually smooth’ functions.
        In any case 2m>d must be satisfied.
        """
        d = len(x)
        m_args = []
        if penalty_order is None:
            penalty_order_default = ceil((d + 1) / 2)
            i = 0
            while not 2 * penalty_order_default > (d + 1) and i < 20:
                penalty_order_default += 1
                i += 1

            m_args.append(str(penalty_order_default))
        else:
            validate_penalty_order(penalty_order)
            m_args.append(str(penalty_order))

        if remove_null_space_completely:
            m_args.append("0")
        m_str = "c(" + ", ".join(m_args) + ")"

        basis = self.s(
            *x,
            k=k,
            bs="tp",
            m=m_str,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name=basis_name,
        )
        return basis

    def ts(
        self,
        *x: str,
        k: int,
        penalty_order: int | None = None,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
    ) -> Basis:
        """
        For penalty_order:
        m = penalty_order
        Quote from MGCV docs
        The default is to set m (the order of derivative in the thin plate spline
        penalty) to the smallest value satisfying 2m > d+1 where d is the number of
        covariates of the term: this yields ‘visually smooth’ functions.
        In any case 2m>d must be satisfied.
        """
        d = len(x)
        m_args = []
        if not penalty_order:
            m_args.append(str(ceil((d + 1) / 2)))
        else:
            validate_penalty_order(penalty_order)
            m_args.append(str(penalty_order))

        m_str = "c(" + ", ".join(m_args) + ")"

        basis = self.s(
            *x,
            k=k,
            bs="ts",
            m=m_str,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name=basis_name,
        )
        return basis

    def kriging(
        self,
        *x: str,
        k: int,
        kernel_name: Literal[
            "spherical",
            "power_exponential",
            "matern1.5",
            "matern2.5",
            "matern3.5",
        ] = "matern1.5",
        linear_trend: bool = True,
        range: float | None = None,
        power_exponential_power: float = 1.0,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
    ) -> Basis:
        """

        - If range=None, the range parameter will be estimated as in Kammann and \
            Wand (2003)
        """
        m_kernel_dict = {
            "spherical": 1,
            "power_exponential": 2,
            "matern1.5": 3,
            "matern2.5": 4,
            "matern3.5": 5,
        }
        m_linear = 1.0 if linear_trend else -1.0

        m_args = []
        m_kernel = str(int(m_linear * m_kernel_dict[kernel_name]))
        m_args.append(m_kernel)
        if range:
            m_range = str(range)
            m_args.append(m_range)
        if power_exponential_power:
            if not range:
                m_args.append(str(-1.0))
            if not 0.0 < power_exponential_power <= 2.0:
                raise ValueError(
                    "'power_exponential_power' must be in (0, 2.0], "
                    f"got {power_exponential_power}"
                )
            m_args.append(str(power_exponential_power))

        m_str = "c(" + ", ".join(m_args) + ")"

        basis = self.s(
            *x,
            k=k,
            bs="gp",
            m=m_str,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name=basis_name,
        )

        return basis

    def lin(
        self,
        formula: str,
        xname: str = "",
        basis_name: str = "X",
        include_intercept: bool = False,
        context: dict[str, Any] | None = None,
    ) -> LinBasis:
        validate_formula(formula)
        spec = fo.ModelSpec(formula, output="numpy")

        if not include_intercept:
            # because we do our own intercept handling with the full model matrix
            # it may be surprising to assert that there is an intercept only if
            # the plan is to remove it.
            # But in order to safely remove it, we first have to ensure that it is
            # present.
            assert_intercept_in_spec(spec)

        # evaluate model matrix once to get a spec with structure information
        # also necessary to populate spec with the correct information for
        # transformations like center, scale, standardize
        spec = spec.get_model_matrix(self.data, context=context).model_spec

        # get column names. There may be a more efficient way to do it
        # that does not require building the model matrix a second time, but this
        # works robustly for now: we take the names that formulaic creates
        column_names = list(
            fo.ModelSpec(formula, output="pandas")
            .get_model_matrix(self.data, context=context)
            .columns
        )[1:]

        required = sorted(str(var) for var in spec.required_variables)
        df_subset = self.data.loc[:, required]
        df_colnames = df_subset.columns

        variables = dict()

        mappings = {}
        for col in df_colnames:
            result = self.registry.get_obs_and_mapping(col)
            variables[col] = result.var

            if result.mapping is not None:
                self.mappings[col] = result.mapping
                mappings[col] = result.mapping

        xvar = lsl.TransientCalc(  # for memory-efficiency
            lambda *args: jnp.vstack(args).T,
            *list(variables.values()),
            _name=self.names.create_lazily(xname) if xname else xname,
        )

        def basis_fn(x):
            df = pd.DataFrame(x, columns=df_colnames)

            # for categorical variables: convert integer representation back to
            # labels
            for col in df_colnames:
                if col in self.mappings:
                    integers = df[col].to_numpy()
                    df[col] = self.mappings[col].integers_to_labels(integers)

            basis = np.asarray(spec.get_model_matrix(df, context=context))
            if not include_intercept:
                basis = basis[:, 1:]
            return jnp.asarray(basis, dtype=float)

        if xname:
            bname = self.names.create_lazily(basis_name + "(" + xvar.name + ")")
        else:
            bname = self.names.create_lazily(basis_name)

        basis = LinBasis(
            xvar,
            basis_fn=basis_fn,
            use_callback=True,
            cache_basis=True,
            name=bname,
        )

        basis.model_spec = spec
        basis.mappings = mappings
        basis.column_names = column_names

        return basis

    def ri(
        self,
        cluster: str,
        basis_name: str = "B",
        penalty: ArrayLike | None = None,
    ) -> Basis:
        if penalty is not None:
            penalty = jnp.asarray(penalty)
        result = self.registry.get_obs_and_mapping(cluster)
        if result.mapping is None:
            raise TypeError(f"{cluster=} must be categorical.")

        self.mappings[cluster] = result.mapping
        nparams = len(result.mapping.labels_to_integers_map)

        if penalty is None:
            penalty = jnp.eye(nparams)

        basis = Basis(
            value=result.var,
            basis_fn=lambda x: x,
            name=self.names.create_lazily(basis_name + "(" + cluster + ")"),
            use_callback=False,
            cache_basis=False,
            penalty=jnp.asarray(penalty) if penalty is not None else penalty,
        )

        return basis

    def mrf(
        self,
        x: str,
        k: int = -1,
        polys: dict[str, ArrayLike] | None = None,
        nb: Mapping[str, ArrayLike | list[str] | list[int]] | None = None,
        penalty: ArrayLike | None = None,
        penalty_labels: Sequence[str] | None = None,
        absorb_cons: bool = False,
        diagonal_penalty: bool = False,
        scale_penalty: bool = False,
        basis_name: str = "B",
    ) -> MRFBasis:
        """
        Polys: Dictionary of arrays. The keys of the dict are the region labels.
            The corresponding values define the region by defining polygons.
        nb: Dictionary of array. The keys of the dict are the region labels.
            The corresponding values indicate the neighbors of the region.
            If it is a list or array of strings, the values are the labels of the
            neighbors.
            If it is a list or array of integers, the values are the indices of the
            neighbors.


        mgcv does not concern itself with your category ordering. It *will* order
        categories alphabetically. Penalty columns have to take this into account.

        Comments on return value:

        - If either polys or nb are supplied, the returned container will contain nb.
        - If only a penalty matrix is supplied, the returned container will *not*
          contain nb.
        - Returning the label order only makes sense if the basis is *not*
          reparameterized, because only then we have a clear correspondence of
          parameters to labels.
          If the basis is reparameterized, there's no such correspondence in a clear
          way, so the returned label order is None.

        """

        if not isinstance(k, int):
            raise TypeError(f"'k' must be int, got {type(k)}.")
        if k < -1:
            raise ValueError(f"'k' cannot be smaller than -1, got {k=}.")

        if polys is None and nb is None and penalty is None:
            raise ValueError("At least one of polys, nb, or penalty must be provided.")

        var, mapping = self.registry.get_categorical_obs(x)
        self.mappings[x] = mapping

        labels = set(list(mapping.labels_to_integers_map))

        if penalty is not None:
            if penalty_labels is None:
                raise ValueError(
                    "If 'penalty' is supplied, 'penalty_labels' must also be supplied."
                )
            if len(penalty_labels) != len(labels):
                raise ValueError(
                    f"Variable {x} has {len(labels)} unique entries, but "
                    f"'penalty_labels' has {len(penalty_labels)}. Both must match."
                )

        xt_args = []
        pass_to_r: dict[str, np.typing.NDArray | dict[str, np.typing.NDArray]] = {}
        if polys is not None:
            xt_args.append("polys=polys")
            if not labels == set(list(polys)):
                raise ValueError(
                    "Names in 'poly' must correspond to the levels of 'x'."
                )
            pass_to_r["polys"] = {key: np.asarray(val) for key, val in polys.items()}

        if nb is not None:
            xt_args.append("nb=nb")
            if not labels == set(list(nb)):
                raise ValueError("Names in 'nb' must correspond to the levels of 'x'.")

            nb_processed = {}
            for key, val in nb.items():
                val_arr = np.asarray(val)
                if np.isdtype(val_arr.dtype, np.dtype("int")):
                    # add one to convert to 1-based indexing for R
                    # and cast to float for R
                    val_arr = np.astype(val_arr + 1, float)
                    # val_arr = np.astype(val_arr, float)
                elif np.isdtype(val_arr.dtype, np.dtype("float")):
                    # add one to convert to 1-based indexing for R
                    val_arr = np.astype(np.astype(val_arr, int) + 1, float)
                elif val_arr.dtype.kind == "U":  # must be unicode strings then
                    pass
                else:
                    raise TypeError(f"Unsupported dtype: {val_arr.dtype!r}")

                nb_processed[key] = val_arr

            pass_to_r["nb"] = nb_processed

        if penalty is not None:
            penalty = np.asarray(penalty)
            pen_rank = np.linalg.matrix_rank(penalty)
            pen_dim = penalty.shape[-1]
            if (pen_dim - pen_rank) != 1:
                logger.warning(
                    f"Supplied penalty has dimension {penalty.shape} and rank "
                    f"{pen_rank}. The expected rank deficiency is 1. "
                    "This may indicate a problem. There might be disconnected sets "
                    "of regions in the data represented by this penalty. "
                    "In this case, you probably need more elaborate constraints "
                    "than the ones provided here. You might consider splitting the "
                    "disconnected regions into several mrf terms. "
                    "Otherwise, please only continue if you are certain that you "
                    "know what is happening."
                )

            xt_args.append("penalty=penalty")
            if not np.shape(penalty)[0] == np.shape(penalty)[1]:
                raise ValueError(f"Penalty must be square, got {np.shape(penalty)=}")

            if not np.shape(penalty)[1] == len(labels):
                raise ValueError(
                    "Dimensions of 'penalty' must correspond to the levels of 'x'."
                )
            pass_to_r["penalty"] = penalty

        xt = "list("
        xt += ",".join(xt_args)
        xt += ")"

        if penalty is not None:
            # removing penalty from the pass_to_r dict, because we are giving it
            # special treatment here.
            # specifically, we have to equip it with row and column names to make
            # sure that penalty entries get correctly matched to clusters by mgcv
            penalty_prelim_arr = np.asarray(pass_to_r.pop("penalty"))
            to_r(penalty_prelim_arr, "penalty")
            to_r(np.array(penalty_labels), "penalty_labels")
            r("colnames(penalty) <- penalty_labels")
            r("rownames(penalty) <- penalty_labels")

        spec = f"s({x}, k={k}, bs='mrf', xt={xt})"

        # disabling warnings about "mrf should be a factor"
        # since even turning data into a pandas df and x_array into
        # a categorical series did not satisfy mgcv in that regard.
        # Things still seem to work, and we ensure further above
        # that we are actually dealing with a categorical variable
        # so I think turning the warnings off temporarily here is fine
        # r("old_warn <- getOption('warn')")
        # r("options(warn = -1)")
        observed = mapping.integers_to_labels(var.value)
        regions = list(mapping.labels_to_integers_map)
        df = pd.DataFrame({x: pd.Categorical(observed, categories=regions)})

        smooth = scon.SmoothCon(
            spec,
            data=df,
            diagonal_penalty=diagonal_penalty,
            absorb_cons=absorb_cons,
            scale_penalty=scale_penalty,
            pass_to_r=pass_to_r,
        )
        # r("options(warn = old_warn)")

        x_name = x

        def basis_fun(x):
            """
            The array outputted by this smooth contains column names.
            Here, we remove these column names and convert to jax.
            """
            # disabling warnings about "mrf should be a factor"
            r("old_warn <- getOption('warn')")
            r("options(warn = -1)")
            labels = mapping.integers_to_labels(x)
            df = pd.DataFrame({x_name: pd.Categorical(labels, categories=regions)})
            basis = jnp.asarray(np.astype(smooth.predict(df)[:, 1:], "float"))
            r("options(warn = old_warn)")
            return basis

        smooth_penalty = smooth.penalty
        if np.shape(smooth_penalty)[1] > len(labels):
            smooth_penalty = smooth_penalty[:, 1:]

        penalty_arr = jnp.asarray(np.astype(smooth_penalty, "float"))

        basis = MRFBasis(
            value=var,
            basis_fn=basis_fun,
            name=self.names.create_lazily(basis_name + "(" + x + ")"),
            cache_basis=True,
            use_callback=True,
            penalty=penalty_arr,
        )
        if absorb_cons:
            basis._constraint = "absorbed_via_mgcv"

        try:
            nb_out = to_py(f"{smooth._smooth_r_name}[[1]]$xt$nb", format="numpy")
        except TypeError:
            nb_out = None
        # nb_out = {key: np.astype(val, "int") for key, val in nb_out.items()}

        if absorb_cons:
            label_order = None
        else:
            label_order = list(
                to_py(f"{smooth._smooth_r_name}[[1]]$X", format="pandas").columns
            )
            label_order = [lab[1:] for lab in label_order]  # removes leading x from R

        if nb_out is not None:

            def to_label(code):
                try:
                    label_array = mapping.integers_to_labels(code - 1)
                except TypeError:
                    label_array = code
                return np.atleast_1d(label_array).tolist()

            nb_out = {k: to_label(v) for k, v in nb_out.items()}

        basis.mrf_spec = MRFSpec(mapping, nb_out, label_order)

        return basis


@dataclass
class NameManager:
    prefix: str = ""
    created_names: dict[str, int] = field(default_factory=dict)

    def create(self, name: str, apply_prefix: bool = True) -> str:
        """
        Appends a counter to the given name for uniqueness.
        There is an individual counter for each name.

        If a prefix was passed to the builder on init, the prefix is applied to the
        name.
        """
        if apply_prefix:
            name = self.prefix + name

        i = self.created_names.get(name, 0)

        name_indexed = name + str(i)

        self.created_names[name] = i + 1

        return name_indexed

    def create_lazily(self, name: str, apply_prefix: bool = True) -> str:
        if apply_prefix:
            name = self.prefix + name

        i = self.created_names.get(name, 0)

        if i > 0:
            name_indexed = name + str(i)
        else:
            name_indexed = name

        self.created_names[name] = i + 1

        return name_indexed

    def fname(self, f: str, basis: Basis) -> str:
        return self.create_lazily(f"{f}({basis.x.name})")

    def create_param_name(self, term_name: str, param_name: str) -> str:
        if term_name:
            param_name = f"${param_name}" + "_{" + f"{term_name}" + "}$"
            return self.create_lazily(param_name, apply_prefix=False)
        else:
            param_name = f"${param_name}$"
            return self.create_lazily(param_name, apply_prefix=True)

    def create_beta_name(self, term_name: str) -> str:
        return self.create_param_name(term_name=term_name, param_name="\\beta")

    def create_tau_name(self, term_name: str) -> str:
        return self.create_param_name(term_name=term_name, param_name="\\tau")

    def create_tau2_name(self, term_name: str) -> str:
        return self.create_param_name(term_name=term_name, param_name="\\tau^2")


class TermBuilder:
    def __init__(self, registry: PandasRegistry, prefix_names_by: str = "") -> None:
        self.registry = registry
        self.names = NameManager(prefix=prefix_names_by)
        self.bases = BasisBuilder(registry, names=self.names)

    def _init_default_scale(
        self,
        concentration: float | Array,
        scale: float | Array,
        value: float | Array = 1.0,
        term_name: str = "",
    ) -> ScaleIG:
        scale_name = self.names.create_tau_name(term_name)
        variance_name = self.names.create_tau2_name(term_name)
        scale_var = ScaleIG(
            value=value,
            concentration=concentration,
            scale=scale,
            name=scale_name,
            variance_name=variance_name,
        )
        return scale_var

    @classmethod
    def from_dict(
        cls, data: dict[str, ArrayLike], prefix_names_by: str = ""
    ) -> TermBuilder:
        return cls.from_df(pd.DataFrame(data), prefix_names_by=prefix_names_by)

    @classmethod
    def from_df(cls, data: pd.DataFrame, prefix_names_by: str = "") -> TermBuilder:
        registry = PandasRegistry(
            data, na_action="drop", prefix_names_by=prefix_names_by
        )
        return cls(registry, prefix_names_by=prefix_names_by)

    def labels_to_integers(self, newdata: dict) -> dict:
        return labels_to_integers(newdata, self.bases.mappings)

    # formula
    def lin(
        self,
        formula: str,
        prior: lsl.Dist | None = None,
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        include_intercept: bool = False,
        context: dict[str, Any] | None = None,
    ) -> LinTerm:
        r"""
        Supported:
        - {a+1} for quoted Python
        - `weird name` backtick-strings for weird names
        - (a + b)**n for n-th order interactions
        - a:b for simple interactions
        - a*b for expanding to a + b + a:b
        - a / b for nesting
        - b %in% a for inverted nesting
        - Python functions
        - bs
        - cr
        - cs
        - cc
        - hashed

        .. warning:: If you use bs, cr, cs, or cc, be aware that these will not
            lead to terms that include a penalty. In most cases, you probably want
            to use :meth:`~.TermBuilder.s`, :meth:`~.TermBuilder.ps`, and so on
            instead.

        Not supported:

        - String literals
        - Numeric literals
        - Wildcard "."
        - \| for splitting a formula
        - "te" tensor products

        - "~" in formula
        - 1 + in formula
        - 0 + in formula
        - -1 in formula

        """

        basis = self.bases.lin(
            formula,
            xname="",
            basis_name="X",
            include_intercept=include_intercept,
            context=context,
        )

        if basis.x.name:
            term_name = self.names.create_lazily("lin" + "(" + basis.x.name + ")")
        else:
            term_name = self.names.create_lazily("lin" + "(" + basis.name + ")")

        coef_name = self.names.create_beta_name(term_name)

        term = LinTerm(
            basis, prior=prior, name=term_name, inference=inference, coef_name=coef_name
        )

        term.model_spec = basis.model_spec
        term.mappings = basis.mappings
        term.column_names = basis.column_names

        return term

    def cr(
        self,
        x: str,
        *,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        noncentered: bool = False,
    ) -> Term:
        basis = self.bases.cr(
            x=x,
            k=k,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("cr", basis)

        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale, term_name=fname
            )

        coef_name = self.names.create_beta_name(fname)
        term = Term(
            basis=basis,
            penalty=basis.penalty,
            scale=scale,
            name=fname,
            inference=inference,
            coef_name=coef_name,
        )
        if noncentered:
            term.reparam_noncentered()
        return term

    def cs(
        self,
        x: str,
        *,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        noncentered: bool = False,
    ) -> Term:
        basis = self.bases.cs(
            x=x,
            k=k,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("cs", basis)

        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale, term_name=fname
            )

        coef_name = self.names.create_beta_name(fname)
        term = Term(
            basis=basis,
            penalty=basis.penalty,
            scale=scale,
            name=fname,
            inference=inference,
            coef_name=coef_name,
        )
        if noncentered:
            term.reparam_noncentered()
        return term

    def cc(
        self,
        x: str,
        *,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        noncentered: bool = False,
    ) -> Term:
        basis = self.bases.cc(
            x=x,
            k=k,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("cc", basis)

        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale, term_name=fname
            )

        coef_name = self.names.create_beta_name(fname)
        term = Term(
            basis=basis,
            penalty=basis.penalty,
            scale=scale,
            name=fname,
            inference=inference,
            coef_name=coef_name,
        )
        if noncentered:
            term.reparam_noncentered()
        return term

    def bs(
        self,
        x: str,
        *,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        basis_degree: int = 3,
        penalty_order: int | Sequence[int] = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        noncentered: bool = False,
    ) -> Term:
        basis = self.bases.bs(
            x=x,
            k=k,
            basis_degree=basis_degree,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("bs", basis)

        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale, term_name=fname
            )

        coef_name = self.names.create_beta_name(fname)
        term = Term(
            basis=basis,
            penalty=basis.penalty,
            scale=scale,
            name=fname,
            inference=inference,
            coef_name=coef_name,
        )
        if noncentered:
            term.reparam_noncentered()
        return term

    # P-spline
    def ps(
        self,
        x: str,
        *,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        basis_degree: int = 3,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        noncentered: bool = False,
    ) -> Term:
        basis = self.bases.ps(
            x=x,
            k=k,
            basis_degree=basis_degree,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("ps", basis)

        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale, term_name=fname
            )

        coef_name = self.names.create_beta_name(fname)
        term = Term(
            basis=basis,
            penalty=basis.penalty,
            scale=scale,
            name=fname,
            inference=inference,
            coef_name=coef_name,
        )
        if noncentered:
            term.reparam_noncentered()
        return term

    def cp(
        self,
        x: str,
        *,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        basis_degree: int = 3,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        noncentered: bool = False,
    ) -> Term:
        basis = self.bases.cp(
            x=x,
            k=k,
            basis_degree=basis_degree,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("cp", basis)

        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale, term_name=fname
            )

        coef_name = self.names.create_beta_name(fname)
        term = Term(
            basis=basis,
            penalty=basis.penalty,
            scale=scale,
            name=fname,
            inference=inference,
            coef_name=coef_name,
        )
        if noncentered:
            term.reparam_noncentered()
        return term

    # ANOVA part of isotropic tensor product interaction
    # allows MGCV bases
    def ti(
        self,
        x1: str,
        x2: str,
        bs: BasisTypes | tuple[BasisTypes, BasisTypes] = "ps",
        k: int | tuple[int, int] = -1,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        m: str = "NA",
        knots: ArrayLike | None = None,
        noncentered: bool = False,
    ) -> Term:
        basis = self.bases.ti(
            x1=x1,
            x2=x2,
            bs=bs,
            k=k,
            m=m,
            knots=knots,
            basis_name="B",
        )

        fname = self.names.create_lazily(name="ti")
        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale, term_name=fname
            )

        term = Term.f(
            basis,
            fname=fname,
            scale=scale,
            inference=inference,
            coef_name=None,
            noncentered=noncentered,
        )
        return term

    def te(
        self,
        x1: str,
        x2: str,
        bs: BasisTypes | tuple[BasisTypes, BasisTypes] = "ps",
        k: int | tuple[int, int] = -1,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        m: str = "NA",
        knots: ArrayLike | None = None,
        noncentered: bool = False,
    ) -> Term:
        basis = self.bases.te(
            x1=x1,
            x2=x2,
            bs=bs,
            k=k,
            m=m,
            knots=knots,
            basis_name="B",
        )

        fname = self.names.create_lazily(name="te")
        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale, term_name=fname
            )

        term = Term.f(
            basis,
            fname=fname,
            scale=scale,
            inference=inference,
            coef_name=None,
            noncentered=noncentered,
        )
        return term

    # random intercept
    def ri(
        self,
        cluster: str,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        penalty: ArrayLike | None = None,
        noncentered: bool = False,
    ) -> RITerm:
        basis = self.bases.ri(cluster=cluster, basis_name="B", penalty=penalty)

        fname = self.names.fname("ri", basis)
        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale, term_name=fname
            )

        coef_name = self.names.create_beta_name(fname)

        term = RITerm(
            basis=basis,
            penalty=basis.penalty,
            coef_name=coef_name,
            inference=inference,
            scale=scale,
            name=fname,
        )

        if noncentered:
            term.reparam_noncentered()

        mapping = self.bases.mappings[cluster]
        term.mapping = mapping
        term.labels = list(mapping.labels_to_integers_map)

        return term

    # random scaling
    def rs(
        self,
        x: str | Term | LinTerm,
        cluster: str,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        penalty: ArrayLike | None = None,
        noncentered: bool = False,
    ) -> lsl.Var:
        ri = self.ri(
            cluster=cluster,
            scale=scale,
            inference=inference,
            penalty=penalty,
            noncentered=noncentered,
        )

        if isinstance(x, str):
            x_var = self.registry.get_numeric_obs(x)
            xname = x
        else:
            x_var = x
            xname = x_var.basis.x.name

        fname = self.names.create_lazily("rs(" + xname + "|" + cluster + ")")
        term = lsl.Var.new_calc(
            lambda x, cluster: x * cluster,
            x=x_var,
            cluster=ri,
            name=fname,
        )
        return term

    # varying coefficient
    def vc(
        self,
        x: str,
        by: Term,
    ) -> lsl.Var:
        fname = self.names.create_lazily(x + "*" + by.name)
        x_var = self.registry.get_obs(x)

        term = lsl.Var.new_calc(
            lambda x, by: x * by,
            x=x_var,
            by=by,
            name=fname,
        )
        return term

    # general smooth with MGCV bases
    def s(
        self,
        *x: str,
        k: int,
        bs: BasisTypes,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        m: str = "NA",
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        noncentered: bool = False,
    ) -> Term:
        """
        Works:
        - tp (thin plate splines)
        - ts (thin plate splines with slight null space penalty)

        - cr (cubic regression splines)
        - cs (shrinked cubic regression splines)
        - cc (cyclic cubic regression splines)

        - bs (B-splines)
        - ps (P-splines)
        - cp (cyclic P-splines)

        Works, but not here:
        - re (use .ri instead)
        - mrf (used .mrf instead)
        - te (use .te instead) (with the bases above)
        - ti (use .ti instead) (with the bases above)

        Does not work:
        - ds (Duchon splines)
        - sos (splines on the sphere)
        - gp (gaussian process)
        - so (soap film smooths)
        - ad (adaptive smooths)

        Probably disallow manually:
        - fz (factor smooth interaction)
        - fs (random factor smooth interaction)
        """
        basis = self.bases.s(
            *x,
            k=k,
            bs=bs,
            m=m,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname(bs, basis=basis)

        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale, term_name=fname
            )

        coef_name = self.names.create_beta_name(fname)
        term = Term(
            basis,
            penalty=basis.penalty,
            name=fname,
            coef_name=coef_name,
            scale=scale,
            inference=inference,
        )
        if noncentered:
            term.reparam_noncentered()
        return term

    # markov random field
    def mrf(
        self,
        x: str,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        k: int = -1,
        polys: dict[str, ArrayLike] | None = None,
        nb: Mapping[str, ArrayLike | list[str] | list[int]] | None = None,
        penalty: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        noncentered: bool = False,
    ) -> MRFTerm:
        """
        Polys: Dictionary of arrays. The keys of the dict are the region labels.
            The corresponding values define the region by defining polygons.
        nb: Dictionary of array. The keys of the dict are the region labels.
            The corresponding values indicate the neighbors of the region.
            If it is a list or array of strings, the values are the labels of the
            neighbors.
            If it is a list or array of integers, the values are the indices of the
            neighbors.


        mgcv does not concern itself with your category ordering. It *will* order
        categories alphabetically. Penalty columns have to take this into account.
        """
        basis = self.bases.mrf(
            x=x,
            k=k,
            polys=polys,
            nb=nb,
            penalty=penalty,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("mrf", basis)
        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale, term_name=fname
            )
        coef_name = self.names.create_beta_name(fname)
        term = MRFTerm(
            basis,
            penalty=basis.penalty,
            name=fname,
            scale=scale,
            inference=inference,
            coef_name=coef_name,
        )
        if noncentered:
            term.reparam_noncentered()

        term.polygons = polys
        term.neighbors = basis.mrf_spec.nb
        if basis.mrf_spec.ordered_labels is not None:
            term.ordered_labels = basis.mrf_spec.ordered_labels

        term.labels = list(basis.mrf_spec.mapping.labels_to_integers_map)
        term.mapping = basis.mrf_spec.mapping

        return term

    # general basis function + penalty smooth
    def f(
        self,
        *x: str,
        basis_fn: Callable[[Array], Array],
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        use_callback: bool = True,
        cache_basis: bool = True,
        penalty: ArrayLike | None = None,
        noncentered: bool = False,
    ) -> Term:
        basis = self.bases.basis(
            *x,
            basis_fn=basis_fn,
            use_callback=use_callback,
            cache_basis=cache_basis,
            penalty=penalty,
            basis_name="B",
        )

        fname = self.names.fname("f", basis)
        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale, term_name=fname
            )

        coef_name = self.names.create_beta_name(fname)
        term = Term(
            basis,
            penalty=basis.penalty,
            name=fname,
            scale=scale,
            inference=inference,
            coef_name=coef_name,
        )
        if noncentered:
            term.reparam_noncentered()
        return term

    def kriging(
        self,
        *x: str,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        kernel_name: Literal[
            "spherical",
            "power_exponential",
            "matern1.5",
            "matern2.5",
            "matern3.5",
        ] = "matern1.5",
        linear_trend: bool = True,
        range: float | None = None,
        power_exponential_power: float = 1.0,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        noncentered: bool = False,
    ) -> Term:
        basis = self.bases.kriging(
            *x,
            k=k,
            kernel_name=kernel_name,
            linear_trend=linear_trend,
            range=range,
            power_exponential_power=power_exponential_power,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("kriging", basis)
        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale, term_name=fname
            )
        coef_name = self.names.create_beta_name(fname)
        term = Term(
            basis,
            penalty=basis.penalty,
            name=fname,
            scale=scale,
            inference=inference,
            coef_name=coef_name,
        )
        if noncentered:
            term.reparam_noncentered()
        return term

    def tp(
        self,
        *x: str,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        penalty_order: int | None = None,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        noncentered: bool = False,
        remove_null_space_completely: bool = False,
    ) -> Term:
        basis = self.bases.tp(
            *x,
            k=k,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
            remove_null_space_completely=remove_null_space_completely,
        )

        fname = self.names.fname("tp", basis)
        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale, term_name=fname
            )

        coef_name = self.names.create_beta_name(fname)
        term = Term(
            basis,
            penalty=basis.penalty,
            name=fname,
            scale=scale,
            inference=inference,
            coef_name=coef_name,
        )
        if noncentered:
            term.reparam_noncentered()
        return term

    def ts(
        self,
        *x: str,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        penalty_order: int | None = None,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        noncentered: bool = False,
    ) -> Term:
        basis = self.bases.ts(
            *x,
            k=k,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("ts", basis)
        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale, term_name=fname
            )

        coef_name = self.names.create_beta_name(fname)
        term = Term(
            basis,
            penalty=basis.penalty,
            name=fname,
            scale=scale,
            inference=inference,
            coef_name=coef_name,
        )
        if noncentered:
            term.reparam_noncentered()
        return term

    def ta(
        self,
        *marginals: Term,
        common_scale: ScaleIG | lsl.Var | float | VarIGPrior | None = None,
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        include_main_effects: bool = False,
    ) -> TPTerm:
        inputs = ",".join(list(TPTerm._input_obs([t.basis for t in marginals])))
        fname = self.names.create_lazily("ta(" + inputs + ")")
        coef_name = self.names.create_beta_name(fname)

        if isinstance(common_scale, VarIGPrior):
            common_scale = self._init_default_scale(
                concentration=common_scale.concentration,
                scale=common_scale.scale,
                term_name=fname,
            )

        term = TPTerm(
            *marginals,
            common_scale=common_scale,
            name=fname,
            inference=inference,
            coef_name=coef_name,
            include_main_effects=include_main_effects,
        )
        return term
