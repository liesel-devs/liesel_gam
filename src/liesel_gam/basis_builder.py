from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from math import ceil
from typing import Any, Literal, get_args

import formulaic as fo
import jax
import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import pandas as pd
import smoothcon as scon
from ryp import r, to_py, to_r

from .basis import Basis, LinBasis, MRFBasis, MRFSpec
from .names import NameManager
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
            name=self.names.create(basis_name + "(" + Xname + ")"),
            use_callback=use_callback,
            cache_basis=cache_basis,
            penalty=jnp.asarray(penalty),
        )

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
            name=self.names.create(basis_name + "(" + x_var.name + ")"),
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
            name=self.names.create(basis_name + "(" + x_var.name + ")"),
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
            name=self.names.create(basis_name + "(" + x_var.name + ")"),
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
            name=self.names.create(basis_name + "(" + x_var.name + ")"),
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
            name=self.names.create(basis_name + "(" + x_var.name + ")"),
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
            name=self.names.create(basis_name + "(" + x_var.name + ")"),
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
            name=self.names.create(basis_name + "(" + xname + ")"),
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

        # evaluate model matrix once to get a spec with structure information
        # also necessary to populate spec with the correct information for
        # transformations like center, scale, standardize
        try:
            spec = spec.get_model_matrix(self.data, context=context).model_spec
        except Exception as e:
            raise RuntimeError(
                "Could not build model matrix. This could be caused by "
                "unsupported data dtypes like dates. Please check your input data. "
                "Also check the original error message, included above."
            ) from e

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
            _name=self.names.create(xname) if xname else xname,
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
            bname = self.names.create(basis_name + "(" + xvar.name + ")")
        else:
            bname = self.names.create(basis_name)

        basis = LinBasis(
            xvar,
            basis_fn=basis_fn,
            use_callback=True,
            cache_basis=True,
            name=bname,
            penalty=None,
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

        if not result.is_categorical:
            raise TypeError(f"{cluster=} must be categorical.")

        if result.mapping is not None:
            self.mappings[cluster] = result.mapping

        basis = Basis(
            value=result.var,
            basis_fn=lambda x: x,
            name=self.names.create(basis_name + "(" + cluster + ")"),
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
            neighbors. Indices correspond to regions based on an alphabetical ordering
            of regions.
        penalty: If penalty is supplied, it takes precedence over both nb and polys,
            and the arguments '


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
                    "Names in 'polys' must correspond to the levels of 'x'."
                )
            pass_to_r["polys"] = {key: np.asarray(val) for key, val in polys.items()}

        if nb is not None:
            xt_args.append("nb=nb")
            if not labels == set(list(nb)):
                raise ValueError("Names in 'nb' must correspond to the levels of 'x'.")

            nb_processed = {}
            for key, val in nb.items():
                val_arr = np.asarray(val)
                if val_arr.ndim != 1:
                    raise ValueError(
                        f"Expected 1d arrays in 'nb', got {val_arr.ndim=} for {key}."
                    )
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

        if "nb" in pass_to_r and "penalty" in pass_to_r:
            logger.warning(
                "Both 'nb' and 'penalty' were supplied. 'penalty' will be used to "
                "setup this basis."
            )

        if "polys" in pass_to_r and "penalty" in pass_to_r:
            logger.warning(
                "Both 'polys' and 'penalty' were supplied. 'penalty' will be used "
                "to setup this basis."
            )

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
        elif np.shape(smooth_penalty)[0] < np.shape(smooth_penalty)[1]:
            smooth_penalty = smooth_penalty[:, 1:]

        try:
            penalty_arr = jnp.asarray(np.astype(smooth_penalty, "float"))
        except ValueError:
            penalty_arr = jnp.asarray(np.astype(smooth_penalty[:, 1:], "float"))

        basis = MRFBasis(
            value=var,
            basis_fn=basis_fun,
            name=self.names.create(basis_name + "(" + x + ")"),
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
