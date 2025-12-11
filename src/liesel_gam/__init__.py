import pandas as pd
from ryp import r, to_r

from .__about__ import __version__ as __version__
from .basis import Basis as Basis
from .basis_builder import BasisBuilder as BasisBuilder
from .category_mapping import CategoryMapping as CategoryMapping
from .category_mapping import series_is_categorical as series_is_categorical
from .dist import MultivariateNormalSingular as MultivariateNormalSingular
from .kernel import init_star_ig_gibbs as init_star_ig_gibbs
from .kernel import star_ig_gibbs as star_ig_gibbs
from .plots import plot_1d_smooth as plot_1d_smooth
from .plots import plot_1d_smooth_clustered as plot_1d_smooth_clustered
from .plots import plot_2d_smooth as plot_2d_smooth
from .plots import plot_forest as plot_forest
from .plots import plot_polys as plot_polys
from .plots import plot_regions as plot_regions
from .plots import polys_to_df as polys_to_df
from .plots import summarise_1d_smooth as summarise_1d_smooth
from .plots import summarise_1d_smooth_clustered as summarise_1d_smooth_clustered
from .plots import summarise_cluster as summarise_cluster
from .plots import summarise_lin as summarise_lin
from .plots import summarise_nd_smooth as summarise_nd_smooth
from .plots import summarise_regions as summarise_regions
from .predictor import AdditivePredictor as AdditivePredictor
from .registry import PandasRegistry as PandasRegistry
from .term import BasisDot as BasisDot
from .term import IndexingTerm as IndexingTerm
from .term import Intercept as Intercept
from .term import LinTerm as LinTerm
from .term import MRFTerm as MRFTerm
from .term import RITerm as RITerm
from .term import SmoothTerm as SmoothTerm
from .term import StrctTensorProdTerm as StrctTensorProdTerm
from .term import StrctTerm as StrctTerm
from .term_builder import TermBuilder as TermBuilder
from .var import ScaleIG as ScaleIG
from .var import VarIGPrior as VarIGPrior

try:
    to_r(pd.DataFrame({"a": [1.0, 2.0]}), "___test___")
    r("rm('___test___')")
except ImportError as e:
    raise ImportError(
        "Testing communication between R and Python failed. "
        "Probably, you need to install the R package 'arrow' using "
        "install.packages('arrow')."
        "Also, please consider the original traceback from ryp above."
    ) from e
