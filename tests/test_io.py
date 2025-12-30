import logging
from pathlib import Path

import numpy as np
import pytest

from liesel_gam.io import polygon_is_closed, read_bnd


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


class TestIsPolygonClose:
    def test_is_polygon_closed_true_exact(self):
        poly = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ]
        )
        assert polygon_is_closed(poly) is True

    def test_is_polygon_closed_true_with_tolerance(self):
        poly = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0 + 1e-13, 0.0 - 1e-13],  # within default atol=1e-12
            ]
        )
        assert polygon_is_closed(poly) is True

    def test_is_polygon_closed_false_not_closed(self):
        poly = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        assert polygon_is_closed(poly) is False

    def test_is_polygon_closed_false_too_few_points_when_required(self):
        poly = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
        )  # closed but only 3 points
        assert polygon_is_closed(poly, require_min_points=True) is False

    def test_is_polygon_closed_true_too_few_points_when_not_required(self):
        poly = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])  # closed
        assert polygon_is_closed(poly, require_min_points=False) is True

    def test_is_polygon_closed_raises_on_bad_shape(self):
        poly = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match=r"Expected shape \(n, 2\)"):
            polygon_is_closed(poly)

        poly2 = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match=r"Expected shape \(n, 2\)"):
            polygon_is_closed(poly2)


class TestReadBND:
    def test_read_bnd_parses_multiple_regions_and_quotes(self, tmp_path: Path):
        # mixed quotes: "A" and 'B'
        content = "\"A\",4\n0,0\n1,0\n1,1\n0,0\n'B',4\n10,10\n11,10\n11,11\n10,10\n"
        p = tmp_path / "regions.bnd"
        write_text(p, content)

        polys = read_bnd(p, delimiter=",")

        assert set(polys.keys()) == {"A", "B"}
        assert polys["A"].shape == (4, 2)
        assert polys["B"].shape == (4, 2)
        assert polys["A"].dtype == float
        assert np.allclose(polys["A"][0], [0.0, 0.0])
        assert np.allclose(polys["B"][1], [11.0, 10.0])

    def test_read_bnd_supports_custom_delimiter(self, tmp_path: Path):
        content = '"A";4\n0;0\n1;0\n1;1\n0;0\n'
        p = tmp_path / "regions_semicolon.bnd"
        write_text(p, content)

        polys = read_bnd(p, delimiter=";")
        assert list(polys.keys()) == ["A"]
        assert polys["A"].shape == (4, 2)
        assert np.allclose(polys["A"][-1], polys["A"][0])

    def test_read_bnd_skips_empty_lines(self, tmp_path: Path):
        content = '\n\n"A",4\n0,0\n1,0\n1,1\n0,0\n\n'
        p = tmp_path / "empty_lines.bnd"
        write_text(p, content)

        polys = read_bnd(p, delimiter=",")
        assert list(polys.keys()) == ["A"]

    def test_read_bnd_warns_if_polygon_not_closed(self, tmp_path: Path, caplog):
        content = (
            '"A",4\n'
            "0,0\n"
            "1,0\n"
            "1,1\n"
            "0,1\n"  # not closed
        )
        p = tmp_path / "not_closed.bnd"
        write_text(p, content)

        caplog.set_level(logging.WARNING)
        polys = read_bnd(p, delimiter=",")

        assert "A" in polys
        assert any(
            "does not appear to be closed" in rec.message for rec in caplog.records
        )

    def test_read_bnd_raises_on_invalid_header(self, tmp_path: Path):
        content = '"A",4,extra\n0,0\n1,0\n1,1\n0,0\n'
        p = tmp_path / "bad_header.bnd"
        write_text(p, content)

        with pytest.raises(ValueError, match=r"Invalid header line"):
            read_bnd(p)

    def test_read_bnd_raises_on_invalid_n_points(self, tmp_path: Path):
        content = '"A",not_an_int\n0,0\n'
        p = tmp_path / "bad_npoints.bnd"
        write_text(p, content)

        with pytest.raises(ValueError, match=r"Invalid number of points"):
            read_bnd(p)

    def test_read_bnd_raises_on_invalid_coordinate_line(self, tmp_path: Path):
        content = (
            '"A",2\n'
            "0,0,0\n"  # too many fields
            "1,1\n"
        )
        p = tmp_path / "bad_coord.bnd"
        write_text(p, content)

        with pytest.raises(ValueError, match=r"Invalid coordinate line"):
            read_bnd(p)

    def test_read_bnd_raises_on_unexpected_eof(self, tmp_path: Path):
        content = (
            '"A",3\n0,0\n1,1\n'
            # missing third coordinate line
        )
        p = tmp_path / "eof.bnd"
        write_text(p, content)

        with pytest.raises(ValueError, match=r"Unexpected end of file"):
            read_bnd(p)
