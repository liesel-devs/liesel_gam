import os
import re
import warnings
from collections.abc import Callable

from . import experimental as experimental
from . import io as io
from .__about__ import __version__ as __version__
from .basis import Basis as Basis
from .basis import LinBasis as LinBasis
from .basis import MRFBasis as MRFBasis
from .basis import MRFSpec as MRFSpec
from .basis_builder import BasisBuilder as BasisBuilder
from .category_mapping import CategoryMapping as CategoryMapping
from .category_mapping import series_is_categorical as series_is_categorical
from .constraint import LinearConstraintEVD as LinearConstraintEVD
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
from .term import StrctInteractionTerm as StrctInteractionTerm
from .term import StrctLinTerm as StrctLinTerm
from .term import StrctTensorProdTerm as StrctTensorProdTerm
from .term import StrctTerm as StrctTerm
from .term_builder import TermBuilder as TermBuilder
from .var import ScaleIG as ScaleIG
from .var import UserVar as UserVar
from .var import VarIGPrior as VarIGPrior

_R_46_PANDAS_ISSUE_URL = "https://github.com/liesel-devs/liesel_gam/issues/67"


def _get_r_version(to_py: Callable[..., object]) -> str | None:
    """
    Return ryp's active R version string, if it can be queried.
    """
    try:
        version = to_py("as.character(getRversion())", squeeze=True)
    except Exception:
        return None

    version = str(version).strip()
    if not version:
        return None

    return version.split()[0]


def _is_r_46(version: str) -> bool:
    """
    Return whether an R version string identifies an R 4.6 release.
    """
    return re.match(r"^4\.6(?:\.|$)", version) is not None


def _warn_if_r_46(to_py: Callable[..., object]) -> None:
    """
    Warn about a known R 4.6 / ryp / pandas string-column issue.
    """
    version = _get_r_version(to_py)
    if version is None or not _is_r_46(version):
        return

    warnings.warn(
        "Liesel-GAM detected R "
        f"{version}. R 4.6 may trigger a known ryp/Arrow/pandas issue that can "
        "corrupt pandas string column labels after R-to-Python conversion, "
        "including in workflows that call pandas.read_csv(). See "
        f"{_R_46_PANDAS_ISSUE_URL} for details and temporary workarounds.",
        RuntimeWarning,
        stacklevel=2,
    )


on_rtd = os.environ.get("READTHEDOCS", "False") == "True"
# safeguard because R is not installed in the readthedocs build environment
if not on_rtd:
    import pandas as pd
    from ryp import r, to_py, to_r

    try:
        to_r(pd.DataFrame({"a": [1.0, 2.0]}), "___test___")
        r("rm('___test___')")
        _warn_if_r_46(to_py)
    except ImportError as e:
        raise ImportError(
            "Testing communication between R and Python failed. "
            "Probably, you need to install the R package 'arrow' using "
            "install.packages('arrow')."
            "Also, please consider the original traceback from ryp above."
        ) from e
