"""Safe whitelist-AST compiler: a user V(r) string -> NumPy-vectorized closure.

No eval/exec. Only a small math whitelist is allowed; every other AST construct
is rejected with a message naming it. See docs/specs/
2026-07-23-phase8-freeform-potential-design.md.
"""

import ast
import math
from collections.abc import Callable

import numpy as np

MAX_LEN = 200
MAX_NODES = 80
MAX_POW = 12

PotentialFn = Callable[[np.ndarray], np.ndarray]


class ExpressionError(ValueError):
    """User expression is empty, too large, or uses a non-whitelisted construct."""


_CONSTANTS = {"pi": math.pi, "e": math.e}

_FUNCS: dict[str, Callable] = {
    "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "abs": np.abs, "sign": np.sign,
    "minimum": np.minimum, "maximum": np.maximum, "where": np.where,
}

_BINOPS = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
    ast.Div: np.divide, ast.Pow: np.power,
}
_CMPOPS = {
    ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater,
    ast.GtE: np.greater_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal,
}


def compile_potential(expr: str) -> PotentialFn:
    """Compile a user V(r) expression into a NumPy-vectorized closure.

    Raises ExpressionError for anything outside the math whitelist, or if the
    expression is empty, too long, or too complex.
    """
    if not expr or not expr.strip():
        raise ExpressionError("expression is empty")
    if len(expr) > MAX_LEN:
        raise ExpressionError(f"expression too long ({len(expr)} > {MAX_LEN} chars)")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"syntax error: {exc.msg}") from exc

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise ExpressionError(f"expression too complex ({len(nodes)} > {MAX_NODES} nodes)")

    _check(tree.body)  # raises ExpressionError on any disallowed construct

    def potential(r: np.ndarray) -> np.ndarray:
        out = _eval(tree.body, r)
        return np.asarray(out, dtype=float) * np.ones_like(r, dtype=float)

    return potential


def _check(node: ast.AST, *, in_where: bool = False) -> None:
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise ExpressionError(f"operator {type(node.op).__name__} is not allowed")
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
            if abs(node.right.value) > MAX_POW:
                raise ExpressionError(f"exponent magnitude > {MAX_POW} is not allowed")
        _check(node.left)
        _check(node.right)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ExpressionError(f"unary {type(node.op).__name__} is not allowed")
        _check(node.operand)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise ExpressionError("only numeric constants are allowed")
    elif isinstance(node, ast.Name):
        if node.id != "r" and node.id not in _CONSTANTS:
            raise ExpressionError(f"unknown name {node.id!r}; only r, pi, e are allowed")
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            name = getattr(node.func, "id", type(node.func).__name__)
            raise ExpressionError(f"function {name!r} is not allowed")
        if node.keywords:
            raise ExpressionError("keyword arguments are not allowed")
        for arg in node.args:
            _check(arg, in_where=(node.func.id == "where"))
    elif isinstance(node, ast.Compare):
        if not in_where:
            raise ExpressionError("comparisons are only allowed as arguments to where()")
        if len(node.ops) != 1 or type(node.ops[0]) not in _CMPOPS:
            raise ExpressionError("only a single simple comparison is allowed")
        _check(node.left)
        _check(node.comparators[0])
    else:
        raise ExpressionError(f"{type(node).__name__} is not allowed in a potential")


def _eval(node: ast.AST, r: np.ndarray):
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](_eval(node.left, r), _eval(node.right, r))
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, r)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return r if node.id == "r" else _CONSTANTS[node.id]
    if isinstance(node, ast.Call):
        return _FUNCS[node.func.id](*[_eval(a, r) for a in node.args])
    if isinstance(node, ast.Compare):
        return _CMPOPS[type(node.ops[0])](_eval(node.left, r), _eval(node.comparators[0], r))
    raise ExpressionError(f"{type(node).__name__} is not allowed")  # defense in depth
