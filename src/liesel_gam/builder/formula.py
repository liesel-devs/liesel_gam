"""
Formula parsing for GAM models.

Design Note: ParsedFormula class consideration
--------------------------------------------
Currently returns list[FormulaComponent] from parse(), but could return a
ParsedFormula class for better encapsulation:

Benefits of ParsedFormula:
- Immutable components list (defensive copy)
- Built-in validation (single intercept guarantee)
- Convenience methods: has_intercept(), get_linear_terms()
- Stores original formula string for debugging
- Natural workflow: formula.to_terms() vs parser.convert_to_terms(components)

Current approach (list) is simpler and more Pythonic, but ParsedFormula
would provide stronger guarantees and better UX for complex operations.
"""

from __future__ import annotations

import ast
import re

from ..var import Term
from .components import (
    FactorComponent,
    FormulaComponent,
    InterceptComponent,
    LieselSplineComponent,
    LinearComponent,
    MGCVComponent,
)
from .registry import VariableRegistry


class FormulaParser:
    """Parser for formula strings to create components."""

    def __init__(
        self,
        registry: VariableRegistry,
        handlers: dict[str, type[FormulaComponent]] | None = None,
    ):
        self.registry = registry

        self.handlers: dict[str, type[FormulaComponent]] = {
            "ls": LieselSplineComponent,
            "s": MGCVComponent,
            "factor": FactorComponent,
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

    def _parse_formula_string(self, formula: str) -> list[str]:
        """Split formula string on + but not inside parentheses."""
        formula = formula.strip()
        if not formula:
            raise ValueError("Formula cannot be empty")
        terms = []
        paren_depth = 0
        current_term = ""

        for char in formula:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            elif char == "+" and paren_depth == 0:
                if current_term.strip():
                    terms.append(current_term.strip())
                else:
                    raise ValueError(
                        f"Formula cannot have empty terms separated by '+': {formula}"
                    )
                current_term = ""
                continue
            current_term += char

        if current_term.strip():
            terms.append(current_term.strip())
        else:
            raise ValueError(f"Formula cannot end with '+': {formula}")

        return terms

    def _parse_formula(self, formula: str) -> list[FormulaComponent]:
        """Parse formula string into structured components.

        Args:
            formula: Formula string to parse

        Returns:
            List of FormulaComponent objects representing the parsed formula
        """
        terms = self._parse_formula_string(formula)
        components: list[FormulaComponent] = []
        intercept_components: list[InterceptComponent] = []

        for term in terms:
            term = term.strip()

            # handle explicit intercept specification
            if term == "0":
                intercept_comp = InterceptComponent(has_intercept=False)
                components.append(intercept_comp)
                intercept_components.append(intercept_comp)
                continue
            elif term == "1":
                intercept_comp = InterceptComponent(has_intercept=True)
                components.append(intercept_comp)
                intercept_components.append(intercept_comp)
                continue

            # check for a function call using AST parsing
            if self._looks_like_function_call(term):
                try:
                    func_name, args, kwargs = self._parse_function_call(term)
                    if func_name in self.handlers:
                        handler = self.handlers[func_name]
                        component = handler.from_formula(func_name, args, kwargs, term)
                        components.append(component)
                    else:
                        raise ValueError(
                            f"Unknown function '{func_name}' in formula: {term}"
                        )
                except ValueError:
                    # Re-raise parsing errors for malformed function calls
                    raise
            # the rest are linear terms
            else:
                components.append(LinearComponent(vars=[term], intercept=False))

        # check for conflicting intercept specifications during parsing
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

    def _looks_like_function_call(self, term: str) -> bool:
        """Check if term looks like a function call using simple regex."""
        return bool(re.match(r"^\s*\w+\s*\(.*\)\s*$", term.strip()))

    def _is_function_call(self, term: str) -> bool:
        """Check if term looks like a function call using AST."""
        try:
            tree = ast.parse(term.strip(), mode="eval")
            return isinstance(tree.body, ast.Call)
        except Exception:
            return False

    def _parse_function_call(self, term: str) -> tuple[str, list, dict]:
        """Parse function call using AST to handle nested parentheses correctly."""
        try:
            tree = ast.parse(term.strip(), mode="eval")
            if not isinstance(tree.body, ast.Call):
                raise ValueError("Not a function call")

            call = tree.body
            func_name = ast.unparse(call.func)
            args = [
                ast.literal_eval(arg)
                if isinstance(arg, ast.Constant)
                else ast.unparse(arg)
                for arg in call.args
            ]
            kwargs = {
                kw.arg: ast.literal_eval(kw.value)
                if isinstance(kw.value, ast.Constant)
                else ast.unparse(kw.value)
                for kw in call.keywords
            }
            return func_name, args, kwargs
        except Exception as e:
            raise ValueError(f"Failed to parse function call '{term}': {e}") from e


def _parse_args_in_formula(arg_str: str, formula: str):
    try:
        expr = ast.parse(f"f({arg_str})", mode="eval")
        call = expr.body
        args = [
            ast.literal_eval(a) if isinstance(a, ast.Constant) else ast.unparse(a)
            for a in call.args  # type: ignore
        ]
        kwargs = {
            kw.arg: ast.literal_eval(kw.value)
            if isinstance(kw.value, ast.Constant)
            else ast.unparse(kw.value)
            for kw in call.keywords  # type: ignore
        }
        return args, kwargs
    except Exception as e:
        raise ValueError(
            f"Failed to parse arguments in formula {formula}({arg_str})"
        ) from e
