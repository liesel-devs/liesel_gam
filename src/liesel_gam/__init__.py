from .__about__ import __version__ as __version__
from .builder.gam_builder import GamBuilder
from .dist import MultivariateNormalSingular as MultivariateNormalSingular
from .kernel import init_star_ig_gibbs as init_star_ig_gibbs
from .kernel import star_ig_gibbs as star_ig_gibbs
from .predictor import AdditivePredictor as AdditivePredictor
from .var import Basis as Basis
from .var import Intercept as Intercept
from .var import LinearTerm as LinearTerm
from .var import SmoothTerm as SmoothTerm

__all__ = [
    "__version__",
    "GamBuilder",
    "MultivariateNormalSingular",
    "init_star_ig_gibbs",
    "star_ig_gibbs",
    "AdditivePredictor",
    "Basis",
    "Intercept",
    "LinearTerm",
    "SmoothTerm",
]
