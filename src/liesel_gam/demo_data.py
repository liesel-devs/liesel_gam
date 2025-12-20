import numpy as np
import pandas as pd
from scipy import stats


def demo_data(n: int, seed: int = 1) -> pd.DataFrame:
    """
    Generate demo data for structured additive distributional regression.

    Parameters
    ----------
    n
        Number of samples (if grid=False).
    seed
        Random seed.

    Returns
    -------
        DataFrame with columns:

        - y (response)
        - x_nonlin (continuous covariate with nonlinear effect on both location and
          scale)
        - x_lin (continuous covariate with linear effect on both location and scale)
        - x_cat (categorical covariate with effect on location)
        - x (continuous covariate with no effect)

    Examples
    --------

    >>> import liesel_gam as gam
    >>> gam.demo_data(n=100).columns
    Index(['y', 'x_nonlin', 'x_lin', 'x_cat'], dtype='object')
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-2, 2, n)
    x3 = rng.choice(["a", "b", "c"], size=n, replace=True)
    x4 = rng.uniform(-2, 2, n)

    log_sigma = (
        -1.0
        + 0.3
        * (0.5 * x1 + 15 * stats.norm.pdf(2 * (x1 - 0.2)) - stats.norm.pdf(x1 + 0.4))
        - 0.1 * x2
    )
    mu = -x1 + np.pi * np.sin(np.pi * x1) + x2 + 0.3 * (x3 == "c") - 0.2 * (x3 == "b")

    y = mu + np.exp(log_sigma) * rng.normal(0.0, 1.0, n)

    df = pd.DataFrame({"y": y, "x_nonlin": x1, "x_lin": x2, "x_cat": x3, "x": x4})
    return df


def demo_data_ta(
    n: int, noise_sd: float = 0.3, grid: bool = False, seed: int = 1
) -> pd.DataFrame:
    """
    Generate demo data for anisotropic tensor products.

    Parameters
    ----------
    n
        Number of samples (if grid=False).
    noise_sd
        Standard deviation of Gaussian noise.
    grid
        If True, generate approximately n points on a regular grid.
        If False, generate random design points.
    seed
        Random seed.

    Returns
    -------
        DataFrame with columns: x, y, eta (signal), z (noisy response)
    """
    rng = np.random.default_rng(seed)

    # --- Design ----
    if grid:
        m = int(np.ceil(np.sqrt(n)))
        xs = np.linspace(0, 1, m)
        ys = np.linspace(0, 1, m)
        x, y = np.meshgrid(xs, ys, indexing="xy")
        x = x.ravel()
        y = y.ravel()
    else:
        x = rng.uniform(0, 1, n)
        y = rng.uniform(0, 1, n)

    # --- True anisotropic smooth ----
    def f_true(x, y):
        # 1) Fast variation in x, slow in y
        term1 = 1.5 * np.sin(6 * np.pi * x) * np.cos(1 * np.pi * y)

        # 2) Anisotropic Gaussian bump: elongated + rotated
        x0, y0 = 0.65, 0.35
        a_x, a_y = 0.10, 0.35  # different length scales: much tighter in x
        rho = 0.7  # correlation -> rotated ellipse

        X = x - x0
        Y = y - y0
        quad = (X**2) / (a_x**2) + (Y**2) / (a_y**2) + 2 * rho * X * Y / (a_x * a_y)
        term2 = 2.0 * np.exp(-quad)

        # 3) Mild linear trend (mainly in y for additional anisotropy)
        term3 = 0.5 * y

        return term1 + term2 + term3

    eta = f_true(x, y)
    z = eta + rng.normal(scale=noise_sd, size=len(eta))

    return pd.DataFrame({"x": x, "y": y, "eta": eta, "z": z})
