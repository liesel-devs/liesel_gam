"""GAM builder module for improved formula parsing and variable management."""

from .errors import (
    ComponentError,
    FormulaParseError,
    GAMBuilderError,
    JAXCompatibilityError,
    TypeMismatchError,
    VariableNotFoundError,
    VariableTransformError,
)
from .registry import VariableRegistry

__all__ = [
    "GAMBuilderError",
    "VariableNotFoundError",
    "TypeMismatchError",
    "JAXCompatibilityError",
    "FormulaParseError",
    "ComponentError",
    "VariableTransformError",
    "VariableRegistry",
]
