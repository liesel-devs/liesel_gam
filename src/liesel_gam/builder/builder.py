from __future__ import annotations

from typing import Any

import formulaic as fo
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd

from ..var import Basis, BasisDot
from .registry import CategoryMapping, PandasRegistry

InferenceTypes = Any


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


class BasisBuilder:
    def __init__(self, registry: PandasRegistry) -> None:
        self.registry = registry
        self.mappings: dict[str, CategoryMapping] = {}

    @property
    def data(self) -> pd.DataFrame:
        return self.registry.data

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
        - a*b for expanding to a + b + a*b
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


class TermBuilder:
    def __init__(self, registry: PandasRegistry) -> None:
        self.registry = registry
        self.bases = BasisBuilder(registry)

        self._automatically_assigned_xnames: list[str] = []

    def _auto_xname(self) -> str:
        name = "x" + str(len(self._automatically_assigned_xnames) + 1)
        self._automatically_assigned_xnames.append(name)
        return name

    @classmethod
    def from_dict(cls, data: dict[str, np.typing.ArrayLike]) -> TermBuilder:
        return cls.from_df(pd.DataFrame(data))

    @classmethod
    def from_df(cls, data: pd.DataFrame) -> TermBuilder:
        registry = PandasRegistry(data, na_action="drop")
        return cls(registry)

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
