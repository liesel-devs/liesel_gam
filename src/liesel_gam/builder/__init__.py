"""Builder module for constructing GAM models."""

from .builder import BasisBuilder as BasisBuilder
from .builder import TermBuilder as TermBuilder
from .builder import VarIGPrior as VarIGPrior
from .category_mapping import CategoryMapping as CategoryMapping
from .category_mapping import series_is_categorical as series_is_categorical
from .registry import PandasRegistry as PandasRegistry
