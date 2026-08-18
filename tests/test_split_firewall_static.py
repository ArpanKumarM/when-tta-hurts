"""Static analysis test: allow_test=True must never appear as an actual
call-site argument anywhere in the codebase except inside a function that
also calls verify_authorization() -- the guarded final-evaluation path.
Uses Python's `ast` module so docstrings/comments (which legitimately
mention "allow_test=True" for documentation purposes) are never
misdetected as real call sites.
"""

from __future__ import annotations

import ast
from pathlib import Path

SCAN_ROOTS = [Path("src/when_tta_hurts"), Path("scripts")]


def _find_allow_test_true_calls(tree: ast.AST) -> list[ast.FunctionDef]:
    """Return the enclosing FunctionDef (if any) for every call site that
    passes allow_test=True as a keyword argument."""
    hits = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.func_stack: list[ast.FunctionDef] = []

        def visit_FunctionDef(self, node):
            self.func_stack.append(node)
            self.generic_visit(node)
            self.func_stack.pop()

        def visit_Call(self, node):
            for kw in node.keywords:
                if kw.arg == "allow_test" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    hits.append(self.func_stack[-1] if self.func_stack else None)
            self.generic_visit(node)

    Visitor().visit(tree)
    return hits


def _function_calls_verify_authorization(func_node: ast.FunctionDef | None) -> bool:
    if func_node is None:
        return False
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "verify_authorization":
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr == "verify_authorization":
                return True
    return False


def test_no_unguarded_allow_test_true_call_sites():
    violations = []
    for root in SCAN_ROOTS:
        for py_file in root.rglob("*.py"):
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
            for enclosing_func in _find_allow_test_true_calls(tree):
                if not _function_calls_verify_authorization(enclosing_func):
                    violations.append(
                        (str(py_file), enclosing_func.name if enclosing_func else "<module level>")
                    )

    assert violations == [], (
        f"Found allow_test=True call site(s) NOT guarded by a verify_authorization() "
        f"call in the same function: {violations}"
    )


def test_currently_zero_allow_test_true_call_sites_at_all():
    """As of Phase 2B.2, no code anywhere actually calls allow_test=True --
    this is the strongest current state (stronger than merely 'guarded').
    If this test starts failing because a real guarded call site was added,
    that's expected to require deliberate review, not a silent pass."""
    total = 0
    for root in SCAN_ROOTS:
        for py_file in root.rglob("*.py"):
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
            total += len(_find_allow_test_true_calls(tree))
    assert total == 0


def test_load_pilot_split_structurally_cannot_reach_test():
    """Re-confirms (Phase 2B.2 level) the Phase 2A guarantee: the pilot
    loader has no allow_test parameter, so it is structurally impossible
    for any pilot code path to reach the test split."""
    import inspect

    from when_tta_hurts.data import load_pilot_split

    sig = inspect.signature(load_pilot_split)
    assert "allow_test" not in sig.parameters
