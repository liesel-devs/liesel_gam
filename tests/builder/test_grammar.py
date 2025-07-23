import pytest

from liesel_gam.builder.grammer import make_lark_formula_parser


@pytest.fixture
def parser():
    return make_lark_formula_parser()


class TestValidFormulas:
    def test_intercept_formulas(self, parser):
        # Explicit intercept
        result = parser.parse("1")
        assert result["type"] == "formula"
        assert len(result["terms"]) == 1
        assert result["terms"][0]["type"] == "intercept"
        assert result["terms"][0]["value"] == 1

        # No intercept
        result = parser.parse("0")
        assert result["type"] == "formula"
        assert len(result["terms"]) == 1
        assert result["terms"][0]["type"] == "intercept"
        assert result["terms"][0]["value"] == 0

    def test_simple_variables(self, parser):
        # Single variable
        result = parser.parse("x1")
        assert result["type"] == "formula"
        assert len(result["terms"]) == 1
        assert result["terms"][0]["type"] == "var"
        assert result["terms"][0]["name"] == "x1"

        # Variable with underscore
        result = parser.parse("x1_2")
        assert result["type"] == "formula"
        assert len(result["terms"]) == 1
        assert result["terms"][0]["type"] == "var"
        assert result["terms"][0]["name"] == "x1_2"

    def test_mixed_formulas(self, parser):
        result = parser.parse("1 + x1 + x2 + s(x3)")
        assert result["type"] == "formula"
        assert len(result["terms"]) == 4

        # Check intercept
        assert result["terms"][0]["type"] == "intercept"
        assert result["terms"][0]["value"] == 1

        # Check variables
        assert result["terms"][1]["type"] == "var"
        assert result["terms"][1]["name"] == "x1"
        assert result["terms"][2]["type"] == "var"
        assert result["terms"][2]["name"] == "x2"

        # Check function call
        assert result["terms"][3]["type"] == "func_call"
        assert result["terms"][3]["name"] == "s"
        assert result["terms"][3]["positional"] == ["x3"]
        assert result["terms"][3]["keyword"] == {}

    def test_no_intercept_formula(self, parser):
        result = parser.parse("0 + x1")
        assert result["type"] == "formula"
        assert len(result["terms"]) == 2

        assert result["terms"][0]["type"] == "intercept"
        assert result["terms"][0]["value"] == 0
        assert result["terms"][1]["type"] == "var"
        assert result["terms"][1]["name"] == "x1"

    def test_simple_function_calls(self, parser):
        result = parser.parse("s(x1)")
        assert result["type"] == "formula"
        assert len(result["terms"]) == 1

        func_call = result["terms"][0]
        assert func_call["type"] == "func_call"
        assert func_call["name"] == "s"
        assert func_call["positional"] == ["x1"]
        assert func_call["keyword"] == {}

    def test_function_calls_with_kwargs(self, parser):
        result = parser.parse("s(x1, foo=1, bar='baz', bool=False)")
        assert result["type"] == "formula"
        assert len(result["terms"]) == 1

        func_call = result["terms"][0]
        assert func_call["type"] == "func_call"
        assert func_call["name"] == "s"
        assert func_call["positional"] == ["x1"]

        # Check keyword arguments
        kwargs = func_call["keyword"]
        assert kwargs["foo"] == 1
        assert kwargs["bar"] == "baz"
        assert kwargs["bool"] is False

    def test_function_calls_multiple_positional(self, parser):
        result = parser.parse("s(x1, x2)")
        assert result["type"] == "formula"
        assert len(result["terms"]) == 1

        func_call = result["terms"][0]
        assert func_call["type"] == "func_call"
        assert func_call["name"] == "s"
        assert func_call["positional"] == ["x1", "x2"]
        assert func_call["keyword"] == {}

    def test_empty_function_calls(self, parser):
        result = parser.parse("s()")
        assert result["type"] == "formula"
        assert len(result["terms"]) == 1

        func_call = result["terms"][0]
        assert func_call["type"] == "func_call"
        assert func_call["name"] == "s"
        assert func_call["positional"] == []
        assert func_call["keyword"] == {}

    def test_value_type_conversion(self, parser):
        result = parser.parse(
            "s(x1, int_val=42, float_val=3.14, str_val='test', bool_true=True, "
            "bool_false=False)"
        )

        func_call = result["terms"][0]
        kwargs = func_call["keyword"]

        assert kwargs["int_val"] == 42
        assert kwargs["float_val"] == 3.14
        assert kwargs["str_val"] == "test"
        assert kwargs["bool_true"] is True
        assert kwargs["bool_false"] is False


class TestInvalidFormulas:
    def test_negative_intercept(self, parser):
        with pytest.raises(Exception):
            parser.parse("-1")

    def test_subtraction_operator(self, parser):
        with pytest.raises(Exception):
            parser.parse("x1 - 1")

        with pytest.raises(Exception):
            parser.parse("x1 - x2")

    def test_multiplication_operator(self, parser):
        with pytest.raises(Exception):
            parser.parse("x1 * x2")

    def test_interaction_operator(self, parser):
        with pytest.raises(Exception):
            parser.parse("x1 : x2")

    def test_function_with_invalid_args(self, parser):
        with pytest.raises(Exception):
            parser.parse("s(x1 * x2)")

    def test_invalid_variable_names(self, parser):
        # Variable names starting with numbers
        with pytest.raises(Exception):
            parser.parse("2x")

        # Variable names with hyphens (should be underscores)
        with pytest.raises(Exception):
            parser.parse("x1-2")

        # Variable names with dots
        with pytest.raises(Exception):
            parser.parse("x1.2")

    def test_numeric_literals_as_terms(self, parser):
        with pytest.raises(Exception):
            parser.parse("2")

        with pytest.raises(Exception):
            parser.parse("2.3")

        with pytest.raises(Exception):
            parser.parse("x1 + 1")

    def test_parentheses_around_variables(self, parser):
        with pytest.raises(Exception):
            parser.parse("(x1)")

    def test_empty_formula(self, parser):
        with pytest.raises(Exception):
            parser.parse("")

        with pytest.raises(Exception):
            parser.parse("   ")

    def test_formula_ending_with_plus(self, parser):
        with pytest.raises(Exception):
            parser.parse("x1 + ")

    def test_formula_starting_with_plus(self, parser):
        with pytest.raises(Exception):
            parser.parse(" + x1")

    def test_double_plus(self, parser):
        with pytest.raises(Exception):
            parser.parse("x1 ++ x2")

    def test_malformed_function_calls(self, parser):
        # Missing closing parenthesis
        with pytest.raises(Exception):
            parser.parse("s(x1")

        # Missing opening parenthesis
        with pytest.raises(Exception):
            parser.parse("s x1)")

        # Malformed keyword argument (missing value)
        with pytest.raises(Exception):
            parser.parse("s(x1, k=)")

        # Malformed function call
        with pytest.raises(Exception):
            parser.parse("s(x1 x2)")


class TestEdgeCases:
    def test_whitespace_handling(self, parser):
        result1 = parser.parse("  1  +  x1  +  s(x2)  ")
        result2 = parser.parse("1+x1+s(x2)")

        assert result1 == result2

    def test_multiline_formula(self, parser):
        result1 = parser.parse("1 + \n x1")
        result2 = parser.parse("1 + x1")
        assert result1 == result2

    def test_complex_function_arguments(self, parser):
        result = parser.parse(
            "s(x1, x2, x3, k=20, bs='cr', fx=True, name='complex_smooth')"
        )

        func_call = result["terms"][0]
        assert func_call["positional"] == ["x1", "x2", "x3"]
        assert func_call["keyword"]["k"] == 20
        assert func_call["keyword"]["bs"] == "cr"
        assert func_call["keyword"]["fx"] is True
        assert func_call["keyword"]["name"] == "complex_smooth"

    def test_mixed_quote_types(self, parser):
        result = parser.parse("s(x1, single='test', double=\"test2\")")

        func_call = result["terms"][0]
        assert func_call["keyword"]["single"] == "test"
        assert func_call["keyword"]["double"] == "test2"

    def test_long_variable_names(self, parser):
        result = parser.parse("very_long_variable_name_with_lots_of_underscores")

        assert result["terms"][0]["type"] == "var"
        assert (
            result["terms"][0]["name"]
            == "very_long_variable_name_with_lots_of_underscores"
        )
