# TODO: this file is deprecated and will be removed in the future

from __future__ import annotations

import abc
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import pandas as pd
import smoothcon
from liesel.contrib import splines

from .predictor import AdditivePredictor
from .var import Basis, Intercept, LinearTerm, SmoothTerm


class FormulaComponent(abc.ABC):
    """Base class for formula components."""

    @property
    @abc.abstractmethod
    def term_name(self) -> str:
        """Return the name of the term represented by this component."""
        pass


@dataclass
class InterceptComponent(FormulaComponent):
    """Represents an intercept specification (0 or 1)."""

    @property
    def term_name(self) -> str:
        return "Intercept"


@dataclass
class LinearComponent(FormulaComponent):
    """Represents linear variables that should be grouped together."""

    variables: list[str]
    include_intercept: bool

    @property
    def term_name(self) -> str:
        return "linear(" + ("+".join(self.variables)) + ")"


@dataclass
class FunctionComponent(FormulaComponent):
    """Represents a function call like s() or ls()."""

    function: str  # function name (e.g., "s", "ls")
    variable: str  # variable name
    kwargs: dict  # function arguments

    @property
    def term_name(self) -> str:
        return f"{self.function}({self.variable})"


def _build_smoothcon_string(var_name: str, **kwargs) -> str:
    """Build smoothcon string for s() terms."""
    # put strings in quotes
    kwargs = {k: f"'{v}'" if isinstance(v, str) else v for k, v in kwargs.items()}

    # build the string of the kwargs
    kwstr = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    s_str = f"s({var_name}"
    if kwstr:
        s_str += f", {kwstr}"
    s_str += ")"
    return s_str


class Constructor:
    """Formula-based constructor for structured additive models.

    Provides R-like formula syntax for building structured additive models or parts
    of them. On the predictor level, it supports intercepts, linear terms, and smooth
    terms (either s or ls). Smooth terms with s are handled by mgcv via the smoothcon
    package, while ls uses liesel.contrib.splines for basis construction.

    Additionally, user specific handlers can be provided for custom functions
    (e.g., custom smooth terms).

    Examples:
        >>> constructor = Constructor(df)
        >>> predictor = constructor.predictor("x1 + x2 + s(x3, k=20)")
        >>> y = constructor.response("y", tfd.Normal, loc=predictor, scale=1.0)
    """

    def __init__(self, data: pd.DataFrame, handlers: dict[str, Callable] | None = None):
        """Initialize Constructor with data and optional custom handlers.

        Args:
            data: pandas DataFrame containing model variables
            handlers: dict mapping function names to builder functions
                     e.g., {'foo': custom_builder_function}
        """
        self.data = data
        self._var_cache: dict[str, lsl.Var] = {}
        self._name_counter = 0  # counter for unique auto-generated names
        self._current_predictor_prefix: str | None = (
            None  # current predictor name context
        )

        # default handlers
        self.handlers = {
            "s": self._handle_smoothcon,  # use smoothcon for s()
            "ls": self._handle_liesel_splines,  # use liesel splines for ls()
        }
        if handlers:
            self.handlers.update(handlers)

    def _get_obs(self, name: str) -> lsl.Var:
        """Get or create a liesel Var for a data column.

        Caches the Var to avoid re-creating it multiple times.
        """
        if name not in self._var_cache:
            if name not in self.data.columns:
                raise ValueError(f"Variable '{name}' not found in data")
            self._var_cache[name] = lsl.Var.new_obs(
                self.data[name].to_numpy(), name=name
            )
        return self._var_cache[name]

    def _get_unique_name(self, base_name: str) -> str:
        """Generate a unique name by appending counter to base name."""
        self._name_counter += 1
        return f"{base_name}_{self._name_counter}"

    def _get_semantic_name(self, base_name: str, unique: bool = False) -> str:
        """Generate semantic name using predictor prefix if available."""
        if self._current_predictor_prefix:
            name = f"{self._current_predictor_prefix}_{base_name}"
        else:
            name = base_name

        if unique:
            name = self._get_unique_name(name)

        return name

    def _handle_smoothcon(self, var_name: str, **kwargs) -> SmoothTerm:
        """Handler for s() smooth terms using R's mgcv via smoothcon."""

        input_var = self._get_obs(var_name)
        # pop parameters that can't be handled by smoothcon
        term_name = kwargs.pop("name", self._get_semantic_name(f"s({var_name})"))
        basis_name = kwargs.pop("basis_name", f"{term_name}_B({var_name})")
        ig_concentration = kwargs.pop("ig_concentration", 0.01)
        ig_scale = kwargs.pop("ig_scale", 0.01)

        # use smoothcon to build basis and penalty matrices
        s_str = _build_smoothcon_string(var_name, **kwargs)
        smooth = smoothcon.SmoothCon(s_str, data=self.data)

        # generate semantic names if not provided by user
        basis = Basis(input_var, basis_fn=smooth, name=basis_name)

        # create smooth term with IWLS kernel as sampler
        term = SmoothTerm.new_ig(
            basis=basis,
            penalty=smooth.penalty,
            name=term_name,
            inference=gs.MCMCSpec(gs.IWLSKernel),
            ig_concentration=ig_concentration,
            ig_scale=ig_scale,
        )

        return term

    def _handle_liesel_splines(self, var_name: str, **kwargs) -> SmoothTerm:
        """Handler for ls() smooth terms using liesel.contrib.splines."""
        input_var = self._get_obs(var_name)
        input_values = self.data[var_name].to_numpy()

        # extract parameters with defaults
        k = kwargs.get("k", 10)
        bs = kwargs.get("bs", "ps")

        if bs != "ps":
            raise NotImplementedError(
                f"Basis type '{bs}' not implemented for ls(), only 'ps' supported"
            )

        # create basis matrix using liesel.contrib.splines
        knots = splines.equidistant_knots(input_values, n_param=k, order=3)
        penalty = splines.pspline_penalty(d=k, diff=2)

        # create basis variable
        term_name = kwargs.get("name", self._get_semantic_name(f"ls({var_name})"))
        basis_name = kwargs.get("basis_name", f"{term_name}_B({var_name})")
        basis = Basis(
            value=input_var,
            basis_fn=splines.basis_matrix,
            knots=knots,
            name=basis_name,
        )

        # create smooth term with IWLS kernel as sampler
        term = SmoothTerm.new_ig(
            basis=basis,
            penalty=penalty,
            name=term_name,
            inference=gs.MCMCSpec(gs.IWLSKernel),
            ig_concentration=kwargs.get("ig_concentration", 0.01),
            ig_scale=kwargs.get("ig_scale", 0.01),
        )

        return term

    def _parse_formula_string(self, formula: str) -> list[str]:
        """Split formula string on + but not inside parentheses."""
        formula = formula.strip()
        terms = []
        paren_depth = 0
        current_term = ""

        for char in formula:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            elif char == "+" and paren_depth == 0:
                terms.append(current_term.strip())
                current_term = ""
                continue
            current_term += char

        if current_term.strip():
            terms.append(current_term.strip())

        return terms

    def _parse_function_kwargs(self, kwargs_str: str) -> dict[str, Any]:
        """Parse function keyword arguments from string."""
        kwargs: dict[str, Any] = {}
        if not kwargs_str:
            return kwargs

        # simple parsing of k=value, bs='value' etc.
        for arg in re.split(r",\s*", kwargs_str):
            key_val = arg.split("=", 1)
            if len(key_val) == 2:
                key = key_val[0].strip()
                val = key_val[1].strip()
                # try to parse as int, float, or keep as string
                if (val.startswith("'") and val.endswith("'")) or (
                    val.startswith('"') and val.endswith('"')
                ):
                    kwargs[key] = val[1:-1]
                else:
                    try:
                        if "." in val:
                            kwargs[key] = float(val)
                        else:
                            kwargs[key] = int(val)
                    except ValueError:
                        kwargs[key] = val
        return kwargs

    def parse_formula(
        self, formula: str, default_intercept: bool = True
    ) -> list[FormulaComponent]:
        """Parse formula string into structured components.

        Args:
            formula: Formula string to parse
            default_intercept: Whether to include intercept by default

        Returns:
            List of FormulaComponent objects representing the parsed formula
        """
        terms = self._parse_formula_string(formula)
        components: list[FormulaComponent] = []
        linear_vars: list[str] = []
        explicit_intercept = None

        for term in terms:
            term = term.strip()

            # handle explicit intercept specification
            if term == "0":
                if explicit_intercept is not None:
                    raise ValueError(
                        "Intercept can only be specified as '0' or '1' once in formula"
                    )
                explicit_intercept = False
                continue
            elif term == "1":
                if explicit_intercept is not None:
                    raise ValueError(
                        "Intercept can only be specified as '0' or '1' once in formula"
                    )
                explicit_intercept = True
                continue

            # check if it's a function call
            func_match = re.match(r"(\w+)\s*\(\s*([^,)]+)(?:,\s*(.+))?\s*\)", term)

            if func_match:
                func_name = func_match.group(1)
                var_name = func_match.group(2).strip()
                kwargs_str = func_match.group(3)
                kwargs = self._parse_function_kwargs(kwargs_str or "")

                components.append(
                    FunctionComponent(
                        function=func_name, variable=var_name, kwargs=kwargs
                    )
                )
            else:
                # simple variable name
                if term and term not in ["0", "1"]:
                    linear_vars.append(term)
                else:
                    raise ValueError(f"Invalid term '{term}' in formula")

        # linear variables
        if linear_vars:
            include_intercept = (
                explicit_intercept
                if explicit_intercept is not None
                else default_intercept
            )
            components.append(
                LinearComponent(
                    variables=linear_vars, include_intercept=include_intercept
                )
            )
        elif explicit_intercept is True:
            # explicit intercept requested but no linear variables
            components.append(InterceptComponent())

        # sort components, first InterceptComponent, then LinearComponents,
        # then FunctionComponents
        components.sort(
            key=lambda c: (
                not isinstance(c, InterceptComponent),
                not isinstance(c, LinearComponent),
                not isinstance(c, FunctionComponent),
            )
        )

        return components

    def _create_term_from_component(
        self, component: FormulaComponent, name: str | None = None
    ) -> LinearTerm | SmoothTerm | Intercept:
        """Create a term object from a formula component."""
        name = name or self._get_unique_name(component.term_name)

        if isinstance(component, LinearComponent):
            # get all variables and stack them
            vars = [self._get_obs(var_name) for var_name in component.variables]

            def stack_vars(*vars):
                stacked = jnp.column_stack([jnp.atleast_1d(v) for v in vars])
                return stacked

            stacked_var = lsl.Var.new_calc(
                stack_vars,
                *vars,
                name=self._get_unique_name(f"stacked({'+'.join(component.variables)})"),
            )

            return LinearTerm(
                stacked_var,
                name=name,
                add_intercept=component.include_intercept,
                inference=gs.MCMCSpec(gs.IWLSKernel),
            )

        elif isinstance(component, InterceptComponent):
            return Intercept(
                name=name,
                inference=gs.MCMCSpec(gs.IWLSKernel),
            )

        elif isinstance(component, FunctionComponent):
            if component.function not in self.handlers:
                raise ValueError(
                    f"Unknown function '{component.function}'. Available: "
                    f"{list(self.handlers.keys())}"
                )

            # unless name is provided, use semantic name
            if "name" not in component.kwargs:
                component.kwargs["name"] = name

            return self.handlers[component.function](
                component.variable, **component.kwargs
            )

        else:
            raise ValueError(f"Unknown component type: {type(component)}")

    def get_term_keys(self, formula: str) -> list[str]:
        """Get the term keys that can be used in term_names mapping."""

        components = self.parse_formula(formula, default_intercept=False)
        keys = [comp.term_name for comp in components]
        return keys

    def _terms(
        self,
        formula: str,
        default_intercept: bool,
        term_names: dict[str, str] | None = None,
        semantic_term_names: bool = False,
    ) -> list[LinearTerm | SmoothTerm | Intercept]:
        """Parse formula and return list of terms."""
        components = self.parse_formula(formula, default_intercept)
        terms = []
        term_names = term_names or {}

        for component in components:
            # Get custom name or use semantic naming
            name = term_names.get(component.term_name)
            if name is None:
                if semantic_term_names or self._current_predictor_prefix:
                    name = self._get_semantic_name(component.term_name)
                else:
                    name = None

            term = self._create_term_from_component(component, name)
            terms.append(term)

        return terms

    def term(
        self, formula: str, name: str | None = None
    ) -> LinearTerm | SmoothTerm | Intercept:
        """Parse a single term from formula and return the corresponding Term object.

        This methods adds an intercept column to linear terms by default but does not
        add an intercept to smooth terms. If you want to create a term without
        intercept, use the `0 + ` formula syntax.

        Args:
            formula: formula string for single term (must result in exactly one term)
            name: optional custom name for the term

        Examples:
            >>> constructor.term("x1")  # Single linear term (with intercept)
            >>> constructor.term(
            ...     "x1 + x2"
            ... )  # Multiple variables in one linear term (with intercept)
            >>> constructor.term("ls(x1)")  # Single smooth term (no intercept)
            >>> constructor.term("x1 + ls(x2)")  # Multiple terms -> ERROR
        """

        keys = self.get_term_keys(formula)
        if len(keys) != 1:
            raise ValueError(
                f"term() expects exactly one term, but formula '{formula}' produces"
                f" {len(keys)} terms: {keys}"
            )
        term_names = {keys[0]: name if name else self._get_unique_name(keys[0])}
        terms = self._terms(formula, default_intercept=True, term_names=term_names)

        assert len(terms) == 1, "Expected exactly one term from formula parsing"
        term = terms[0]

        return term

    def predictor(
        self,
        formula: str,
        inv_link=None,
        name: str | None = None,
        term_names: dict[str, str] | None = None,
    ) -> AdditivePredictor:
        """Create AdditivePredictor with terms from formula string.

        Args:
            formula: R-like formula string (e.g., "x1 + x2 + s(x3, k=20)")
            inv_link: optional inverse link function (bijector)
            name: optional custom name for the predictor
            term_names: optional mapping from term keys to custom names.
                       Use get_term_keys() to see available keys.

        Returns:
            AdditivePredictor with terms added according to formula

        Examples:
            >>> predictor = constructor.predictor("x1 + s(x2)", name="mu")

            >>> # Custom term names
            >>> predictor = constructor.predictor(
            ...     "x1 + s(x2)",
            ...     name="mu",
            ...     term_names={"x1": "offset", "s(x2)": "trend"},
            ... )
        """
        if not formula:
            raise ValueError("Predictor must have at least one term")

        predictor_name = (
            name if name else self._get_semantic_name("predictor()", unique=True)
        )

        # set predictor context for semantic naming
        old_prefix = self._current_predictor_prefix
        self._current_predictor_prefix = name

        try:
            # get all terms from formula
            terms = self._terms(
                formula,
                default_intercept=True,
                term_names=term_names,
                semantic_term_names=True,
            )

            # create predictor
            predictor = AdditivePredictor(predictor_name, inv_link=inv_link)

            # add all terms to predictor
            for term in terms:
                predictor += term

            return predictor
        finally:
            # restore previous prefix context
            self._current_predictor_prefix = old_prefix

    def response(
        self, var_name: str, distribution, name: str | None = None, **params
    ) -> lsl.Var:
        """Create response variable with specified distribution.

        Args:
            var_name: name of response variable in data
            distribution: TensorFlow Probability distribution class
            name: optional custom name for the response variable
            **params: distribution parameters (e.g., predictors or liesel.Var)

        Returns:
            liesel.Var representing the response variable
        """
        if var_name not in self.data.columns:
            raise ValueError(f"Response variable '{var_name}' not found in data")

        response_data = self.data[var_name].to_numpy()
        response_name = name if name else var_name

        return lsl.Var.new_obs(
            value=response_data,
            distribution=lsl.Dist(distribution, **params),
            name=response_name,
        )
