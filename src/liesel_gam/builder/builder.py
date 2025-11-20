from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, get_args

import formulaic as fo
import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd
import smoothcon as scon
from ryp import r, to_py

from ..var import Basis, BasisDot, IndexingTerm, MRFTerm, ScaleIG, Term
from .registry import CategoryMapping, PandasRegistry

InferenceTypes = Any

Array = jax.Array
ArrayLike = jax.typing.ArrayLike

BasisTypes = Literal["tp", "ts", "cr", "cs", "cc", "bs", "ps", "cp"]


logger = logging.getLogger(__name__)


class MRFSpec(NamedTuple):
    basis: Basis
    nb: dict[str, np.typing.NDArray[np.int_]] | None
    labels: list[str] | None


class VarIGPrior(NamedTuple):
    concentration: float | Array
    scale: float | Array


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


def _tp_penalty(K1, K2) -> np.typing.NDArray:
    """Computes the full tensor product penalty from the marginals."""
    # this should go into smoothcon, but it works here for now
    D1 = np.shape(K1)[1]
    D2 = np.shape(K2)[1]
    I1 = np.eye(D1)
    I2 = np.eye(D2)

    return np.kron(K1, I2) + np.kron(I1, K2)


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


class BasisBuilder:
    def __init__(self, registry: PandasRegistry) -> None:
        self.registry = registry
        self.mappings: dict[str, CategoryMapping] = {}

    @property
    def data(self) -> pd.DataFrame:
        return self.registry.data

    def basis(
        self,
        *x: str,
        basis_fn: Callable[[Array], Array] = lambda x: x,
        use_callback: bool = True,
        cache_basis: bool = True,
        penalty: np.typing.ArrayLike | lsl.Value | None = None,
        Bname: str = "B",
    ) -> Basis:
        x_vars = []
        for x_name in x:
            x_var = self.registry.get_numeric_obs(x_name)
            x_vars.append(x_var)

        Xname = ",".join(x)

        Xvar = lsl.TransientCalc(
            lambda *x: jnp.column_stack(x),
            *x_vars,
            _name=Xname,
        )

        basis = Basis(
            value=Xvar,
            basis_fn=basis_fn,
            name=Bname + "(" + Xname + ")",
            use_callback=use_callback,
            cache_basis=cache_basis,
            penalty=jnp.asarray(penalty),
        )

        return basis

    def te(
        self,
        x1: str,
        x2: str,
        bs: BasisTypes | tuple[BasisTypes, BasisTypes] = "tp",
        k: int | tuple[int, int] = -1,
        m: str = "NA",
        knots: np.typing.ArrayLike | None = None,
        Bname: str = "B",
    ) -> Basis:
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
        x1x2_name = f"{x1},{x2}"
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
            name=Bname + "(" + x1 + "," + x2 + ")",
            basis_fn=lambda x: jnp.asarray(smooth.predict({x1: x[:, 0], x2: x[:, 1]})),
            penalty=K1 + K2,
            use_callback=True,
            cache_basis=True,
        )
        return basis

    def ti(
        self,
        x1: str,
        x2: str,
        bs: BasisTypes | tuple[BasisTypes, BasisTypes] = "tp",
        k: int | tuple[int, int] = -1,
        m: str = "NA",
        knots: np.typing.ArrayLike | None = None,
        Bname: str = "B",
    ) -> Basis:
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

        x1x2_name = f"{x1},{x2}"

        x1_x2_var = lsl.TransientCalc(
            lambda x1, x2: jnp.c_[x1, x2],
            x1=x1_var,
            x2=x2_var,
            _name=x1x2_name,
        )

        penalty = _tp_penalty(*_margin_penalties(smooth))

        basis = Basis(
            x1_x2_var,
            name=Bname + "(" + x1 + "," + x2 + ")",
            basis_fn=lambda x: jnp.asarray(smooth.predict({x1: x[:, 0], x2: x[:, 1]})),
            penalty=penalty,
            use_callback=True,
            cache_basis=True,
        )
        return basis

    def ps(
        self,
        x: str,
        k: int = 20,
        basis_degree: int = 3,
        penalty_order: int = 2,
        knots: np.typing.ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        Bname: str = "B",
    ) -> Basis:
        spec = f"s({x}, bs='ps', k={k}, m=c({basis_degree}, {penalty_order}))"
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
            name=Bname + "(" + x + ")",
            basis_fn=lambda x_: jnp.asarray(smooth.predict({x: x_})),
            penalty=smooth.penalty,
            use_callback=True,
            cache_basis=True,
        )
        return basis

    def s(
        self,
        x: str,
        k: int = -1,
        bs: BasisTypes = "tp",
        m: str = "NA",
        knots: np.typing.ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        Bname: str = "B",
    ) -> Basis:
        _validate_bs(bs)
        bs_arg = f"'{bs}'"
        spec = f"s({x}, bs={bs_arg}, k={k}, m={m})"
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
            name=Bname + "(" + x + ")",
            basis_fn=lambda x_: jnp.asarray(smooth.predict({x: x_})),
            penalty=smooth.penalty,
            use_callback=True,
            cache_basis=True,
        )
        return basis

    def fo(
        self,
        formula: str,
        name: str = "",
        include_intercept: bool = False,
        context: dict[str, Any] | None = None,
    ) -> Basis:
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

        Not supported:

        - String literals
        - Numeric literals
        - Wildcard "."
        - \| for splitting a formula
        - "te" tensor products

        - "~" in formula

        """
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

        required = sorted(str(var) for var in spec.required_variables)
        df_subset = self.data.loc[:, required]
        df_colnames = df_subset.columns

        variables = dict()

        for col in df_colnames:
            result = self.registry.get_obs_and_mapping(col)
            variables[col] = result.var

            if result.mapping is not None:
                self.mappings[col] = result.mapping

        xvar = lsl.TransientCalc(  # for memory-efficiency
            lambda *args: jnp.vstack(args).T,
            *list(variables.values()),
            _name=name,
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

        basis = Basis(
            xvar,
            basis_fn=basis_fn,
            name=None,  # to use automatic naming based on xvar.name.
            use_callback=True,
            cache_basis=True,
        )

        return basis

    def ri(
        self, cluster: str, Bname: str = "B", penalty: np.typing.ArrayLike | None = None
    ) -> Basis:
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
            name=Bname + "(" + cluster + ")",
            use_callback=False,
            cache_basis=False,
            penalty=jnp.asarray(penalty) if penalty is not None else penalty,
        )

        return basis

    def mrf(
        self,
        x: str,
        k: int = -1,
        polys: dict[str, np.typing.ArrayLike] | None = None,
        nb: dict[str, np.typing.ArrayLike | list[str] | list[int]] | None = None,
        penalty: np.typing.ArrayLike | None = None,
        absorb_cons: bool = False,
        diagonal_penalty: bool = False,
        scale_penalty: bool = False,
        Bname: str = "B",
    ) -> MRFSpec:
        """
        Polys: Dictionary of arrays. The keys of the dict are the region labels.
            The corresponding values define the region by defining polygons.
        nb: Dictionary of array. The keys of the dict are the region labels.
            The corresponding values indicate the neighbors of the region.
            If it is a list or array of strings, the values are the labels of the
            neighbors.
            If it is a list or array of integers, the values are the indices of the
            neighbors.

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
        if polys is None and nb is None and penalty is None:
            raise ValueError("At least one of polys, nb, or penalty must be provided.")

        var, mapping = self.registry.get_categorical_obs(x)
        self.mappings[x] = mapping

        labels = set(list(mapping.labels_to_integers_map))

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

        spec = f"s({x}, k={k}, bs='mrf', xt={xt})"

        x_array = np.asarray(var.value)

        # disabling warnings about "mrf should be a factor"
        # since even turning data into a pandas df and x_array into
        # a categorical series did not satisfy mgcv in that regard.
        # Things still seem to work, and we ensure further above
        # that we are actually dealing with a categorical variable
        # so I think turning the warnings off temporarily here is fine
        r("old_warn <- getOption('warn')")
        r("options(warn = -1)")
        smooth = scon.SmoothCon(
            spec,
            data={x: x_array},
            diagonal_penalty=diagonal_penalty,
            absorb_cons=absorb_cons,
            scale_penalty=scale_penalty,
            pass_to_r=pass_to_r,
        )
        r("options(warn = old_warn)")

        def basis_fun(x):
            """
            The array outputted by this smooth contains column names.
            Here, we remove these column names and convert to jax.
            """
            # disabling warnings about "mrf should be a factor"
            r("old_warn <- getOption('warn')")
            r("options(warn = -1)")
            basis = jnp.asarray(np.astype(smooth(x)[:, 1:], "float"))
            r("options(warn = old_warn)")
            return basis

        smooth_penalty = smooth.penalty
        if np.shape(smooth_penalty)[1] > len(labels):
            smooth_penalty = smooth_penalty[:, 1:]

        penalty_arr = jnp.asarray(np.astype(smooth_penalty, "float"))

        basis = Basis(
            value=var,
            basis_fn=basis_fun,
            name=Bname + "(" + x + ")",
            cache_basis=True,
            use_callback=True,
            penalty=penalty_arr,
        )

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
            # switch to zero-based indexing as expected in Python
            nb_out = {k: np.astype(v - 1, int) for k, v in nb_out.items()}

        return MRFSpec(basis, nb_out, label_order)


class TermBuilder:
    def __init__(self, registry: PandasRegistry) -> None:
        self.registry = registry
        self.bases = BasisBuilder(registry)

        self._automatically_assigned_xnames: list[str] = []
        self._automatically_assigned_fnames: dict[str, list[str]] = dict()

    def _auto_xname(self) -> str:
        name = "x" + str(len(self._automatically_assigned_xnames) + 1)
        self._automatically_assigned_xnames.append(name)
        return name

    def _auto_fname(self, fname: str) -> str:
        max_i = 10_000
        i = 1
        fname_indexed = fname + str(i)
        if fname not in self._automatically_assigned_fnames:
            self._automatically_assigned_fnames[fname] = []
        names_with_this_prefix = self._automatically_assigned_fnames[fname]
        while fname_indexed in names_with_this_prefix:
            i += 1
            fname_indexed = fname + str(i)
            if i > max_i:
                raise RuntimeError("Maximum number of iterations reached.")

        self._automatically_assigned_fnames[fname].append(fname_indexed)
        return fname_indexed

    def _init_default_scale(
        self,
        concentration: float | Array,
        scale: float | Array,
        value: float | Array = 1.0,
    ) -> ScaleIG:
        scale_name = self._auto_fname("$\\tau$")
        variance_name = self._auto_fname("$\\tau^2$")
        scale_var = ScaleIG(
            value=value,
            concentration=concentration,
            scale=scale,
            name=scale_name,
            variance_name=variance_name,
        )
        return scale_var

    @classmethod
    def from_dict(cls, data: dict[str, np.typing.ArrayLike]) -> TermBuilder:
        return cls.from_df(pd.DataFrame(data))

    @classmethod
    def from_df(cls, data: pd.DataFrame) -> TermBuilder:
        registry = PandasRegistry(data, na_action="drop")
        return cls(registry)

    def labels_to_integers(self, newdata: dict) -> dict:
        return labels_to_integers(newdata, self.bases.mappings)

    # formula
    def fo(
        self,
        formula: str,
        prior: lsl.Dist | None = None,
        name: str = "",
        xname: str = "",
        coef_name: str = "",
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        include_intercept: bool = False,
        context: dict[str, Any] | None = None,
    ) -> BasisDot:
        if xname == "":
            xname = self._auto_xname()

        if name == "":
            name = "fo(" + xname + ")"

        basis = self.bases.fo(
            formula, name=xname, include_intercept=include_intercept, context=context
        )

        term = BasisDot(
            basis, prior=prior, name=name, inference=inference, coef_name=coef_name
        )

        return term

    # P-spline
    def ps(
        self,
        x: str,
        k: int = 20,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        basis_degree: int = 3,
        penalty_order: int = 2,
        knots: np.typing.ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        noncentered: bool = False,
    ) -> Term:
        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale
            )

        basis = self.bases.ps(
            x=x,
            k=k,
            basis_degree=basis_degree,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            Bname=self._auto_fname(fname="B"),
        )

        fname = self._auto_fname(fname="ps")
        term = Term.f(
            basis,
            fname=fname,
            scale=scale,
            inference=inference,
            coef_name=None,
            noncentered=noncentered,
        )
        return term

    # ANOVA part of isotropic tensor product interaction
    # allows MGCV bases
    def ti(
        self,
        x1: str,
        x2: str,
        bs: BasisTypes | tuple[BasisTypes, BasisTypes] = "tp",
        k: int | tuple[int, int] = -1,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        m: str = "NA",
        knots: np.typing.ArrayLike | None = None,
        noncentered: bool = False,
    ) -> Term:
        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale
            )

        basis = self.bases.ti(
            x1=x1,
            x2=x2,
            bs=bs,
            k=k,
            m=m,
            knots=knots,
            Bname=self._auto_fname(fname="B"),
        )

        fname = self._auto_fname(fname="ti")
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
        bs: BasisTypes | tuple[BasisTypes, BasisTypes] = "tp",
        k: int | tuple[int, int] = -1,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        m: str = "NA",
        knots: np.typing.ArrayLike | None = None,
        noncentered: bool = False,
    ) -> Term:
        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale
            )

        basis = self.bases.te(
            x1=x1,
            x2=x2,
            bs=bs,
            k=k,
            m=m,
            knots=knots,
            Bname=self._auto_fname(fname="B"),
        )

        fname = self._auto_fname(fname="te")
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
        penalty: np.typing.ArrayLike | None = None,
        noncentered: bool = False,
    ) -> IndexingTerm:
        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale
            )

        basis = self.bases.ri(
            cluster=cluster, Bname=self._auto_fname(fname="RI"), penalty=penalty
        )

        fname = self._auto_fname(fname="ri")
        term = IndexingTerm.f(
            basis=basis,
            scale=scale,
            inference=inference,
            noncentered=noncentered,
            fname=fname,
        )

        return term

    # random scaling
    def rs(
        self,
        x: str | Term,
        cluster: str,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        penalty: np.typing.ArrayLike | None = None,
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

        fname = self._auto_fname(fname="rs")
        term = lsl.Var.new_calc(
            lambda x, ri: x * ri,
            x=x_var,
            ri=ri,
            name=fname + "(" + xname + "|" + cluster + ")",
        )
        return term

    # varying coefficient
    def vc(
        self,
        x: str,
        by: Term,
    ) -> lsl.Var:
        fname = self._auto_fname(fname="rs")
        x_var = self.registry.get_obs(x)

        term = lsl.Var.new_calc(
            lambda x, by: x * by,
            x=x_var,
            by=by,
            name=fname + "(" + x + "*" + by.name + ")",
        )
        return term

    # general smooth with MGCV bases
    def s(
        self,
        x: str,
        k: int = -1,
        bs: BasisTypes = "tp",
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        m: str = "NA",
        knots: np.typing.ArrayLike | None = None,
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
        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale
            )

        basis = self.bases.s(
            x=x,
            k=k,
            bs=bs,
            m=m,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            Bname=self._auto_fname(fname="B"),
        )

        fname = self._auto_fname(fname="s")
        term = Term.f(
            basis,
            fname=fname,
            scale=scale,
            inference=inference,
            coef_name=None,
            noncentered=noncentered,
        )
        return term

    # markov random field
    def mrf(
        self,
        x: str,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        k: int = -1,
        polys: dict[str, np.typing.ArrayLike] | None = None,
        nb: dict[str, np.typing.ArrayLike | list[str] | list[int]] | None = None,
        penalty: np.typing.ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        noncentered: bool = False,
    ) -> MRFTerm:
        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale
            )

        spec = self.bases.mrf(
            x=x,
            k=k,
            polys=polys,
            nb=nb,
            penalty=penalty,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            Bname=self._auto_fname(fname="B"),
        )

        fname = self._auto_fname(fname="mrf")
        term = MRFTerm.f(
            spec.basis,
            fname=fname,
            scale=scale,
            inference=inference,
            coef_name=None,
            noncentered=noncentered,
        )

        term.polygons = polys
        term.neighbors = spec.nb
        if spec.labels is not None:
            term.labels = spec.labels

        return term

    # general basis function + penalty smooth
    def f(
        self,
        *x: str,
        basis_fn: Callable[[Array], Array] = lambda x: x,
        scale: ScaleIG | lsl.Var | float | VarIGPrior = VarIGPrior(1.0, 0.005),
        inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel),
        use_callback: bool = True,
        cache_basis: bool = True,
        penalty: np.typing.ArrayLike | None = None,
        noncentered: bool = False,
    ) -> Term:
        if isinstance(scale, VarIGPrior):
            scale = self._init_default_scale(
                concentration=scale.concentration, scale=scale.scale
            )

        basis = self.bases.basis(
            *x,
            basis_fn=basis_fn,
            use_callback=use_callback,
            cache_basis=cache_basis,
            penalty=penalty,
            Bname=self._auto_fname(fname="B"),
        )

        fname = self._auto_fname(fname="f")
        term = Term.f(
            basis,
            fname=fname,
            scale=scale,
            inference=inference,
            coef_name=None,
            noncentered=noncentered,
        )
        return term
