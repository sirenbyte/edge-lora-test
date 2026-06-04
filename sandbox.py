"""Sandboxed Python execution for the run_python tool (PAL pattern).

Gives the 4B EXACT answers on multi-step / date / list-string problems it would
otherwise get wrong by mental math. Two gates:
  1. static AST allow-list — only safe imports, no exec/eval/open/__import__/dunder;
  2. isolated subprocess (python -I -S, stripped env) with an RLIMIT_CPU cap + a
     wall-clock timeout; stdout only. The model's code MUST print() its answer.

Never raises — always returns a string (the printed result or an 'ошибка: …').
"""
from __future__ import annotations

import ast
import subprocess
import sys
import textwrap

ALLOWED_IMPORTS = {
    "math", "statistics", "datetime", "itertools", "re", "fractions", "decimal",
    "collections", "random", "calendar", "string", "json", "functools", "bisect",
}
_FORBIDDEN = {
    "exec", "eval", "open", "__import__", "compile", "input", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "memoryview", "breakpoint", "help",
}


def _validate(code: str) -> None:
    tree = ast.parse(code)                       # raises SyntaxError on bad code
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ValueError(f"импорт '{a.name}' запрещён")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise ValueError(f"импорт из '{node.module}' запрещён")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("доступ к dunder-атрибутам запрещён")
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN:
            raise ValueError(f"'{node.id}' запрещён")


_PREAMBLE = textwrap.dedent("""
    import resource
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))   # 2s CPU hard cap
    except Exception:
        pass
""")


def run_python(code: str, timeout: int = 5) -> str:
    code = (code or "").strip()
    if not code:
        return "ошибка: пустой код"
    try:
        _validate(code)
    except SyntaxError as e:
        return f"ошибка синтаксиса: {e}"
    except ValueError as e:
        return f"ошибка безопасности: {e}"
    script = _PREAMBLE + "\n" + code
    try:
        p = subprocess.run([sys.executable, "-I", "-S", "-c", script],
                           capture_output=True, text=True, timeout=timeout,
                           env={"PATH": "/usr/bin"})
    except subprocess.TimeoutExpired:
        return "ошибка: превышено время выполнения"
    except Exception as e:
        return f"ошибка: {e}"
    out = (p.stdout or "").strip()
    if out:
        return out[:800]
    return (p.stderr or "(нет вывода — код должен print() результат)").strip()[:400]


if __name__ == "__main__":
    import datetime
    tests = [
        "print(47*83+12)",
        "from datetime import date; print((date(2027,1,1)-date.today()).days)",
        "import statistics; print(statistics.mean([12,18,30]))",
        "print('секунд:', 3*3600+25*60)",
        "import os; print(os.listdir('/'))",          # must be blocked
        "open('/etc/passwd').read()",                  # must be blocked
    ]
    for t in tests:
        print(f"{t[:45]:45} -> {run_python(t)[:60]}")
