"""Custom exception classes for the GAM builder module."""

from __future__ import annotations


class GAMBuilderError(Exception):
    """Base exception class for all GAM builder related errors."""

    pass


class VariableNotFoundError(GAMBuilderError):
    """Raised when a requested variable is not found in the data."""

    def __init__(self, variable_name: str, available_vars: list[str] | None = None):
        self.variable_name = variable_name
        self.available_vars = available_vars

        msg = f"Variable '{variable_name}' not found in data"
        if available_vars:
            msg += f". Available variables: {sorted(available_vars)}"

        super().__init__(msg)


class TypeMismatchError(GAMBuilderError):
    """Raised when a variable type doesn't match the expected type."""

    def __init__(self, variable_name: str, expected_type: str, actual_type: str):
        self.variable_name = variable_name
        self.expected_type = expected_type
        self.actual_type = actual_type

        msg = (
            f"Type mismatch for variable '{variable_name}': "
            f"expected {expected_type}, got {actual_type}"
        )

        super().__init__(msg)


class JAXCompatibilityError(GAMBuilderError):
    """Raised when a variable cannot be converted to JAX-compatible format."""

    def __init__(self, variable_name: str, reason: str):
        self.variable_name = variable_name
        self.reason = reason

        msg = f"Variable '{variable_name}' is not JAX-compatible: {reason}"

        super().__init__(msg)


class FormulaParseError(GAMBuilderError):
    """Raised when a formula string cannot be parsed correctly."""

    def __init__(self, formula: str, reason: str, position: int | None = None):
        self.formula = formula
        self.reason = reason
        self.position = position

        msg = f"Failed to parse formula '{formula}': {reason}"
        if position is not None:
            msg += f" at position {position}"

        super().__init__(msg)


class ComponentError(GAMBuilderError):
    """Raised when a formula component encounters an error during processing."""

    def __init__(self, component_type: str, reason: str):
        self.component_type = component_type
        self.reason = reason

        msg = f"Error in {component_type} component: {reason}"

        super().__init__(msg)


class VariableTransformError(GAMBuilderError):
    """Raised when a variable transformation fails."""

    def __init__(self, variable_name: str, transform_name: str, reason: str):
        self.variable_name = variable_name
        self.transform_name = transform_name
        self.reason = reason

        msg = (
            f"Failed to apply transformation '{transform_name}' "
            f"to variable '{variable_name}': {reason}"
        )

        super().__init__(msg)
