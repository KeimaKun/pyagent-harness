"""
harness/verifier.py
===================
Static AST safety checker and deterministic task completion verifier.

Key rules:
- The LLM CANNOT self-mark a task completed. Only this verifier can.
- AST safety blocks dangerous Python calls before execution.
- Completion gate requires: exit_code == 0 AND files exist AND files > 0 bytes.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dangerous AST patterns
# ---------------------------------------------------------------------------
# Each entry: (check_description, callable(node) -> bool)
_DANGEROUS_FUNCTIONS = {
    "os.system",
    "os.popen",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "eval",
    "exec",
    "__import__",
    "compile",
}

# Subprocess calls that are dangerous when shell=True
_SUBPROCESS_FUNCTIONS = {
    "subprocess.call",
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.check_call",
    "subprocess.check_output",
}

# File-system destructors (paths containing these are flagged)
_DESTRUCTIVE_PATH_FRAGMENTS = {
    "rm -rf",
    "rmdir /s",
    "format c:",
    "del /f",
    "shutil.rmtree",
}


class ASTViolation:
    """Represents a single AST safety violation."""
    def __init__(self, rule: str, line: int, detail: str) -> None:
        self.rule = rule
        self.line = line
        self.detail = detail

    def __str__(self) -> str:
        return f"Line {self.line}: [{self.rule}] {self.detail}"


class ASTSafetyChecker(ast.NodeVisitor):
    """Walks an AST tree and collects safety violations."""

    def __init__(self) -> None:
        self.violations: list[ASTViolation] = []

    def _qualified_name(self, node: ast.expr) -> str:
        """Attempt to reconstruct a dotted name like 'os.system'."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = self._qualified_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = self._qualified_name(node.func)

        # Check against dangerous function list
        if name in _DANGEROUS_FUNCTIONS:
            self.violations.append(ASTViolation(
                rule="DANGEROUS_CALL",
                line=node.lineno,
                detail=f"Blocked call to '{name}'",
            ))

        # Check subprocess with shell=True
        if name in _SUBPROCESS_FUNCTIONS:
            for kw in node.keywords:
                if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    self.violations.append(ASTViolation(
                        rule="SHELL_INJECTION",
                        line=node.lineno,
                        detail=f"'{name}' called with shell=True — potential shell injection",
                    ))

        # Check for string literals containing destructive commands
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                lower = arg.value.lower()
                for frag in _DESTRUCTIVE_PATH_FRAGMENTS:
                    if frag in lower:
                        self.violations.append(ASTViolation(
                            rule="DESTRUCTIVE_COMMAND",
                            line=node.lineno,
                            detail=f"String argument contains destructive fragment: '{frag}'",
                        ))

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        # Flag direct os import in suspicious context (informational, not blocking)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        # shutil.rmtree as attribute access without call context
        qualified = self._qualified_name(node)
        if qualified == "shutil.rmtree":
            self.violations.append(ASTViolation(
                rule="DESTRUCTIVE_CALL",
                line=node.lineno,
                detail="Reference to 'shutil.rmtree' detected — potential file destruction",
            ))
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Public AST check function
# ---------------------------------------------------------------------------
def check_ast(code: str) -> tuple[bool, str]:
    """
    Parse and safety-check Python source code.

    Parameters
    ----------
    code : str
        Python source code string to check.

    Returns
    -------
    (is_safe: bool, message: str)
        is_safe=True means no violations found.
        message contains violation details or 'AST check passed'.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        msg = f"Syntax error in code: {exc}"
        logger.warning(msg)
        return False, msg

    checker = ASTSafetyChecker()
    checker.visit(tree)

    if checker.violations:
        details = "\n".join(str(v) for v in checker.violations)
        msg = f"AST safety violations detected:\n{details}"
        logger.warning(msg)
        return False, msg

    return True, "AST check passed."


# ---------------------------------------------------------------------------
# Deterministic Task Completion Gate
# ---------------------------------------------------------------------------
def verify_task_completion(
    exit_code: int,
    output_files: list[str | Path],
    cwd: Path | str | None = None,
) -> tuple[bool, str]:
    """
    Deterministic completion gate. The LLM CANNOT mark a task completed —
    this function is the sole authority.

    Parameters
    ----------
    exit_code : int
        Subprocess exit code from executing the task's generated code.
    output_files : list of str or Path
        Files declared by the task as expected outputs.
    cwd : Path or str, optional
        Working directory to resolve relative file paths.

    Returns
    -------
    (passed: bool, reason: str)
    """
    base = Path(cwd) if cwd else Path.cwd()
    reasons: list[str] = []
    passed = True

    # Gate 1: Exit code
    if exit_code != 0:
        passed = False
        reasons.append(f"Exit code {exit_code} != 0")
    else:
        reasons.append("Exit code: 0 ✓")

    # Gates 2 & 3: File existence and non-zero size
    for raw_path in output_files:
        p = Path(raw_path)
        if not p.is_absolute():
            p = base / p

        if not p.exists():
            passed = False
            reasons.append(f"Output file missing: {p}")
        elif p.stat().st_size == 0:
            passed = False
            reasons.append(f"Output file is empty (0 bytes): {p}")
        else:
            reasons.append(f"Output file OK ({p.stat().st_size} bytes): {p.name} ✓")

    summary = "\n".join(reasons)
    if passed:
        logger.info("Completion gate PASSED:\n%s", summary)
    else:
        logger.warning("Completion gate FAILED:\n%s", summary)

    return passed, summary


# ---------------------------------------------------------------------------
# Convenience: verify generated code and execution together
# ---------------------------------------------------------------------------
def full_verification(
    code: str,
    exit_code: int,
    output_files: list[str | Path],
    cwd: Path | str | None = None,
    ast_enabled: bool = True,
) -> tuple[bool, str]:
    """
    Run AST safety check then deterministic completion gate.
    Returns (passed, combined_report).
    """
    report_parts: list[str] = []

    if ast_enabled:
        ast_ok, ast_msg = check_ast(code)
        report_parts.append(f"[AST] {ast_msg}")
        if not ast_ok:
            return False, "\n".join(report_parts)

    comp_ok, comp_msg = verify_task_completion(exit_code, output_files, cwd)
    report_parts.append(f"[COMPLETION GATE]\n{comp_msg}")

    return comp_ok, "\n".join(report_parts)
