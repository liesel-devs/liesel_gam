import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def read_bnd(
    filename: str | Path,
    delimiter: str = ",",
) -> dict[str, np.typing.NDArray]:
    """
    Reads a .bnd file with geographical information, returns a polys dictionary.

    Parameters
    ----------
    filename
        Filename.
    delimiter
        Delimiter

    Notes
    -----
    The expected file format looks like this::

        "A", 66
        25.6422, -17.8037
        25.6477, -17.7607
        ...
        "B", 192
        27.276, -17.0096
        27.6848, -17.1684
        ...

    Where the lines ``"A",66`` and ``"B",192`` are header lines that indicate the region
    label, and the number of following lines that define the polygon for the respective
    region. In this case, the region labelled ``"A"`` is defined by the following 66
    lines, and the region labelled ``"B"`` is defined by the following 192 lines.
    """
    polygons = {}

    with open(filename) as f:
        while True:
            header = f.readline()
            if not header:
                break  # EOF

            header = header.strip()
            if not header:
                continue  # skip empty lines

            parts = header.split(delimiter)
            if len(parts) != 2:
                raise ValueError(f"Invalid header line: {header}")

            label, n_points = parts
            label = label.strip().strip("\"'")

            try:
                n_points = int(n_points)
            except ValueError:
                raise ValueError(f"Invalid number of points in header: {header}")

            coords = []
            for _ in range(n_points):
                line = f.readline()
                if not line:
                    raise ValueError("Unexpected end of file while reading coordinates")

                line = line.strip()
                if not line:
                    raise ValueError("Unexpected empty line while reading coordinates")

                xy = line.split(delimiter)
                if len(xy) != 2:
                    raise ValueError(f"Invalid coordinate line: {line}")

                x, y = map(float, xy)
                coords.append([x, y])

            polygons[label] = np.array(coords, dtype=float)

    for label, poly in polygons.items():
        if not polygon_is_closed(poly):
            msg = (
                f"Polygon of region with label '{label}' does not appear to be closed "
                "(first and last point are not equal)."
            )
            logger.warning(msg)

    return polygons


def polygon_is_closed(
    poly: np.typing.ArrayLike,
    *,
    atol: float = 1e-12,
    rtol: float = 0.0,
    require_min_points: bool = True,
) -> bool:
    """
    Validate that a polygon is closed: first vertex equals last vertex (within
    tolerance).

    Parameters
    ----------
    poly : array-like, shape (n, 2)
        Polygon vertices.
    atol, rtol : float
        Absolute / relative tolerances used by np.allclose.
    require_min_points : bool
        If True, require at least 4 points for a closed polygon (first==last + >=3
        distinct vertices).

    Returns
    -------
        True if closed, else False.
    """
    arr = np.asarray(poly, dtype=float)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected shape (n, 2), got {arr.shape}")

    n = arr.shape[0]
    if n == 0:
        return False

    if require_min_points and n < 4:
        return False

    return np.allclose(arr[0], arr[-1], atol=atol, rtol=rtol)
