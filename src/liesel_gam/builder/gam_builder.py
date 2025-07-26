"""GamBuilder for constructing structured additive models using new component system."""

from __future__ import annotations

import jax.numpy as jnp
import liesel.model as lsl
import pandas as pd

from ..predictor import AdditivePredictor
from .components import FormulaComponent
from .formula import FormulaParser
from .registry import VariableRegistry


class GamBuilder:
    """Formula-based constructor for structured additive models.

    Provides R-like formula syntax for building structured additive models or parts
    of them.

    Examples:
        >>> builder = GamBuilder(df)
        >>> predictor = builder.predictor("x1 + x2 + s(x3, k=20)")
        >>> y = builder.response("y", tfd.Normal, loc=predictor, scale=1.0)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        handlers: dict[str, type[FormulaComponent]] | None = None,
    ) -> None:
        """Initialize GamBuilder with data and optional custom handlers.

        Args:
            data: pandas DataFrame containing model variables
            handlers: dict mapping function names to FormulaComponent classes
                     e.g., {'custom_smooth': CustomSmoothComponent}
        """
        self.registry = VariableRegistry(data)
        self.parser = FormulaParser(self.registry, handlers)
        self._name_counter = 0
        self._current_predictor_prefix: str | None = None

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

    def terms(
        self,
        formula: str,
    ) -> list:
        """Parse formula and return list of terms.

        This method does not add an default intercept term.

        Args:
            formula: R-like formula string (e.g., "x1 + x2 + s(x3, k=20)")

        Returns:
            List of Term objects from the parsed formula

        Examples:
            >>> terms = builder.terms("x1 + s(x2)")
        """
        return self.parser.parse_to_terms(
            formula, default_intercept=False, name_prefix=""
        )

    def predictor(
        self,
        formula: str,
        inv_link=None,
        name: str | None = None,
    ) -> AdditivePredictor:
        """Create AdditivePredictor with terms from formula string.

        Args:
            formula: R-like formula string (e.g., "x1 + x2 + s(x3, k=20)")
            inv_link: optional inverse link function (bijector)
            name: optional custom name for the predictor
            default_intercept: whether to include an intercept term by default

        Returns:
            AdditivePredictor with terms added according to formula

        Examples:
            >>> predictor = builder.predictor("x1 + s(x2)", name="mu")
        """
        predictor_name = (
            name if name else self._get_semantic_name("predictor", unique=True)
        )

        # set predictor context for semantic naming
        old_prefix = self._current_predictor_prefix
        self._current_predictor_prefix = name

        try:
            # parse formula to terms using new parser system
            terms = self.parser.parse_to_terms(
                formula, default_intercept=True, name_prefix=name or ""
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
        if var_name not in self.registry.columns:
            raise ValueError(f"Response variable '{var_name}' not found in data")

        response_data = jnp.array(self.registry.data[var_name].to_numpy())
        response_name = name if name else var_name

        return lsl.Var.new_obs(
            value=response_data,
            distribution=lsl.Dist(distribution, **params),
            name=response_name,
        )
