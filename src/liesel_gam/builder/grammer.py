"""
Formula parsing for GAM models.

Design Note: ParsedFormula class consideration
--------------------------------------------
Currently returns list[FormulaComponent] from parse(), but could return a
ParsedFormula class for better encapsulation:

Benefits of ParsedFormula:
- Immutable components list (defensive copy)
- Built-in validation (single intercept guarantee)
- Convenience methods: has_intercept(), get_linear_terms()
- Stores original formula string for debugging
- Natural workflow: formula.to_terms() vs parser.convert_to_terms(components)

Current approach (list) is simpler and more Pythonic, but ParsedFormula
would provide stronger guarantees and better UX for complex operations.
"""

from __future__ import annotations

from typing import Any

import lark

FORMULA_GRAMMAR = r"""
?start: formula

?formula: intercept ("+" term)*        -> formula
        | term ("+" term)*             -> formula

?term: func_call | var

?func_call: CNAME "(" args? ")"        -> func_call
?args: arg ("," arg)*
?arg: CNAME                            -> positional_arg
    | CNAME "=" VALUE                  -> keyword_arg

?var: CNAME                            -> var
?intercept: INTERCEPT                  -> intercept

VALUE: STRING | NUMBER | INTEGER | "True" | "False" | CNAME
STRING: /"[^"]*"|'[^']*'/
NUMBER: /-?\d+\.\d+/
INTEGER: /-?\d+/
INTERCEPT: /[01]/

%import common.CNAME
%import common.WS
%ignore WS
"""


@lark.v_args(inline=True)
class FormulaTransformer(lark.Transformer):
    """Transform Lark AST to dictionary representation."""

    def formula(self, *terms):
        return {"type": "formula", "terms": list(terms)}

    def intercept(self, value):
        return {"type": "intercept", "value": int(str(value))}

    def var(self, name):
        return {"type": "var", "name": str(name)}

    def func_call(self, name, *args_or_dict):
        if len(args_or_dict) == 0:
            # No arguments
            positional, keyword = [], {}
        elif (
            len(args_or_dict) == 1
            and isinstance(args_or_dict[0], dict)
            and "positional" in args_or_dict[0]
        ):
            # Multiple args wrapped in args() result
            positional = args_or_dict[0]["positional"]
            keyword = args_or_dict[0]["keyword"]
        else:
            # Single argument passed directly
            positional = []
            keyword = {}
            for arg in args_or_dict:
                if isinstance(arg, dict):
                    if arg["type"] == "positional":
                        positional.append(arg["value"])
                    elif arg["type"] == "keyword":
                        keyword[arg["name"]] = arg["value"]

        return {
            "type": "func_call",
            "name": str(name),
            "positional": positional,
            "keyword": keyword,
        }

    def args(self, *args):
        positional = []
        keyword = {}
        for arg in args:
            match arg["type"]:
                case "positional":
                    positional.append(arg["value"])
                case "keyword":
                    keyword[arg["name"]] = arg["value"]
                case _:
                    raise ValueError(f"Unexpected argument type: {arg['type']}")
        return {"positional": positional, "keyword": keyword}

    def positional_arg(self, value):
        return {"type": "positional", "value": str(value)}

    def keyword_arg(self, name, value):
        return {
            "type": "keyword",
            "name": str(name),
            "value": self._convert_value(value),
        }

    def _convert_value(self, value) -> Any:
        """Convert token value to appropriate Python type."""
        if isinstance(value, lark.Token):
            val_str = str(value)
            # Handle boolean tokens
            if val_str in ("True", "False"):
                return val_str == "True"
            # Try numeric conversion
            try:
                return int(val_str)
            except ValueError:
                try:
                    return float(val_str)
                except ValueError:
                    return val_str.strip("\"'")
        else:
            raise ValueError(f"Unexpected value type: {type(value)}")


def make_lark_formula_parser() -> lark.Lark:
    """Get Lark parser for GAM formula."""
    return lark.Lark(
        FORMULA_GRAMMAR,
        parser="lalr",
        transformer=FormulaTransformer(),
    )
