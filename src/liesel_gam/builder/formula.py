from __future__ import annotations

from typing import Any

import lark

from ..var import Term
from .components import (
    FactorComponent,
    FormulaComponent,
    InterceptComponent,
    LieselSplineComponent,
    LinearComponent,
    MGCVComponent,
)
from .grammer import make_lark_formula_parser
from .registry import VariableRegistry


class FormulaParser:
    """Parser for formula strings to create components."""

    def __init__(
        self,
        registry: VariableRegistry,
        handlers: dict[str, type[FormulaComponent]] | None = None,
    ):
        self.registry = registry
        self.lark_parser = make_lark_formula_parser()

        self.handlers: dict[str, type[FormulaComponent]] = {
            "linear": LinearComponent,
            "factor": FactorComponent,
            "ls": LieselSplineComponent,
            "s": MGCVComponent,
        }
        if handlers:
            self.handlers.update(handlers)

    def parse(
        self, formula: str, default_intercept: bool, merge: bool = True
    ) -> list[FormulaComponent]:
        """Parse a formula string into components."""
        components = self._parse_formula(formula)

        # add default intercept if needed
        self._add_default_intercept_if_needed(components, default_intercept)

        # merge linear components and handle intercepts
        if merge:
            components = self.merge_components(components)

        return components

    def convert_to_terms(
        self, components: list[FormulaComponent], name_prefix: str = ""
    ) -> list[Term]:
        """Convert parsed components into Term objects."""
        terms = []
        for comp in components:
            terms.extend(comp.create_terms(self.registry, name_prefix=name_prefix))
        return terms

    def parse_to_terms(
        self,
        formula: str,
        default_intercept: bool,
        merge: bool = True,
        name_prefix: str = "",
    ) -> list[Term]:
        """Parse formula directly to terms."""
        components = self.parse(
            formula, default_intercept=default_intercept, merge=merge
        )
        self.validate_components(components)
        return self.convert_to_terms(components, name_prefix=name_prefix)

    def validate_components(self, components: list[FormulaComponent]) -> None:
        """Validate that all components can be created with current registry."""
        from .errors import VariableNotFoundError

        for comp in components:
            missing_vars = comp.depends_on_vars - set(self.registry.columns)
            if missing_vars:
                raise VariableNotFoundError(
                    list(missing_vars)[0], self.registry.columns
                )

    def _parse_formula(self, formula: str) -> list[FormulaComponent]:
        """Parse formula string into structured components using Lark.

        Args:
            formula: Formula string to parse

        Returns:
            List of FormulaComponent objects representing the parsed formula
        """
        formula = formula.strip()
        if not formula:
            raise ValueError("Formula cannot be empty")

        try:
            ast_dict: dict[str, Any] = self.lark_parser.parse(formula)  # type: ignore
            return self._ast_to_components(ast_dict)
        except lark.exceptions.LarkError as e:
            raise ValueError(f"Failed to parse formula '{formula}': {e}") from e

    def _ast_to_components(self, ast_dict: dict[str, Any]) -> list[FormulaComponent]:
        """Convert AST dictionary to FormulaComponent objects."""
        components: list[FormulaComponent] = []
        intercept_components: list[InterceptComponent] = []

        for term in ast_dict["terms"]:
            if term["type"] == "intercept":
                intercept_comp = InterceptComponent(has_intercept=bool(term["value"]))
                components.append(intercept_comp)
                intercept_components.append(intercept_comp)

            elif term["type"] == "var":
                components.append(LinearComponent(vars=[term["name"]], intercept=False))

            elif term["type"] == "func_call":
                func_name = term["name"]
                pos_args = term["positional"]
                kwargs = term["keyword"]

                if func_name in self.handlers:
                    handler = self.handlers[func_name]
                    original_term = _original_term_string(func_name, pos_args, kwargs)
                    component = handler.from_formula(
                        func_name, pos_args, kwargs, original_term
                    )
                    components.append(component)
                else:
                    raise ValueError(f"Unknown function '{func_name}' in formula")

        # check for conflicting intercept specifications
        if len(intercept_components) > 1:
            keys = [comp.formula_key(simplified=False) for comp in intercept_components]
            raise ValueError(f"Multiple intercept components found: {keys}")

        return components

    def _merge_linear_components(
        self, components: list[FormulaComponent]
    ) -> LinearComponent | None:
        """Merge all linear components into a single component."""
        linear_components = [
            comp for comp in components if isinstance(comp, LinearComponent)
        ]
        if linear_components:
            return LinearComponent.merge_linear_components(linear_components)
        else:
            return None

    def _extract_intercept_components(
        self, components: list[FormulaComponent]
    ) -> InterceptComponent | None:
        """Extract and validate intercept components."""
        intercept_components = [
            comp for comp in components if isinstance(comp, InterceptComponent)
        ]
        if len(intercept_components) > 1:
            keys = [comp.formula_key(simplified=False) for comp in intercept_components]
            raise ValueError(f"Multiple intercept components found: {keys}")

        return intercept_components[0] if intercept_components else None

    def _apply_intercept_to_linear(
        self,
        merged_linear: LinearComponent | None,
        intercept_component: InterceptComponent | None,
    ) -> FormulaComponent | None:
        """Apply intercept settings to linear component."""
        if intercept_component:
            if merged_linear is not None:
                # apply intercept setting to merged linear component
                intercept_setting = intercept_component.includes_intercept
                merged_linear._intercept = intercept_setting
                return merged_linear
            else:
                # no linear components, just add intercept component
                return intercept_component
        elif merged_linear is not None:
            # no intercept components but have linear components
            return merged_linear
        else:
            return None

    def _filter_other_components(
        self, components: list[FormulaComponent]
    ) -> list[FormulaComponent]:
        """Filter out linear and intercept components, return the rest."""
        return [
            comp
            for comp in components
            if not isinstance(comp, LinearComponent | InterceptComponent)
        ]

    def _validate_single_intercept(self, components: list[FormulaComponent]) -> None:
        """Validate that at most one component includes an intercept."""
        intercept_count = sum(1 for c in components if c.includes_intercept)
        if intercept_count > 1:
            intercept_comps = [c for c in components if c.includes_intercept]
            raise ValueError(
                "Multiple components include intercepts: "
                f"{[c.formula_key(simplified=False) for c in intercept_comps]}"
            )

    def _add_default_intercept_if_needed(
        self, components: list[FormulaComponent], default_intercept: bool
    ) -> None:
        """Add default intercept if needed and none exists."""
        if not default_intercept:
            return

        # check if any component already includes an intercept
        has_intercept = any(comp.includes_intercept for comp in components)

        # check if there's an explicit intercept component
        has_intercept_component = any(
            isinstance(comp, InterceptComponent) for comp in components
        )

        if not has_intercept and not has_intercept_component:
            components.append(InterceptComponent(has_intercept=True))

    def merge_components(
        self, components: list[FormulaComponent]
    ) -> list[FormulaComponent]:
        """Merge linear components and handle intercept specifications."""

        new_components = []

        # merge linear components
        merged_linear = self._merge_linear_components(components)

        # extract and validate intercept components
        intercept_component = self._extract_intercept_components(components)

        # apply intercept settings to linear component
        combined_component = self._apply_intercept_to_linear(
            merged_linear, intercept_component
        )
        if combined_component:
            new_components.append(combined_component)

        # add all other components (smooth terms, etc.)
        other_components = self._filter_other_components(components)
        new_components.extend(other_components)

        # validate only one component includes an intercept
        self._validate_single_intercept(new_components)

        return new_components


def _original_term_string(
    func_name: str, pos_args: list, kwargs: dict[str, Any]
) -> str:
    pos_args_str = ", ".join(pos_args)
    if kwargs:
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    else:
        kwargs_str = ""

    args_str = f"{pos_args_str}{', ' if pos_args and kwargs else ''}{kwargs_str}"
    return f"{func_name}({args_str})"
