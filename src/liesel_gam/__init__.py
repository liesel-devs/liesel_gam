import os

from .__about__ import __version__ as __version__
from .basis import Basis as Basis
from .basis import LinBasis as LinBasis
from .basis import MRFBasis as MRFBasis
from .basis import MRFSpec as MRFSpec
from .basis_builder import BasisBuilder as BasisBuilder
from .category_mapping import CategoryMapping as CategoryMapping
from .category_mapping import series_is_categorical as series_is_categorical
from .demo_data import demo_data as demo_data
from .demo_data import demo_data_ta as demo_data_ta
from .dist import MultivariateNormalSingular as MultivariateNormalSingular
from .dist import MultivariateNormalStructured as MultivariateNormalStructured
from .dist import StructuredPenaltyOperator as StructuredPenaltyOperator
from .kernel import init_star_ig_gibbs as init_star_ig_gibbs
from .kernel import star_ig_gibbs as star_ig_gibbs
from .names import NameManager as NameManager
from .plots import plot_1d_smooth as plot_1d_smooth
from .plots import plot_1d_smooth_clustered as plot_1d_smooth_clustered
from .plots import plot_2d_smooth as plot_2d_smooth
from .plots import plot_forest as plot_forest
from .plots import plot_polys as plot_polys
from .plots import plot_regions as plot_regions
from .predictor import AdditivePredictor as AdditivePredictor
from .registry import PandasRegistry as PandasRegistry
from .summary import polys_to_df as polys_to_df
from .summary import summarise_1d_smooth as summarise_1d_smooth
from .summary import summarise_1d_smooth_clustered as summarise_1d_smooth_clustered
from .summary import summarise_by_samples as summarise_by_samples
from .summary import summarise_cluster as summarise_cluster
from .summary import summarise_lin as summarise_lin
from .summary import summarise_nd_smooth as summarise_nd_smooth
from .summary import summarise_regions as summarise_regions
from .term import BasisDot as BasisDot
from .term import IndexingTerm as IndexingTerm
from .term import LinMixin as LinMixin
from .term import LinTerm as LinTerm
from .term import MRFTerm as MRFTerm
from .term import RITerm as RITerm
from .term import SmoothTerm as SmoothTerm
from .term import StrctLinTerm as StrctLinTerm
from .term import StrctTensorProdTerm as StrctTensorProdTerm
from .term import StrctTerm as StrctTerm
from .term_builder import TermBuilder as TermBuilder
from .var import ScaleIG as ScaleIG
from .var import VarIGPrior as VarIGPrior

on_rtd = os.environ.get("READTHEDOCS", "False") == "True"
# safeguard because R is not installed in the readthedocs build environment
if not on_rtd:
    import pandas as pd
    from ryp import r, to_r

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
