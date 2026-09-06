"""Reward scoring for LiveCodeBench (code_generation_lite).

The ground_truth stored in the parquet is encoded as:
    base64 -> zlib -> pickle -> JSON string
The decoded JSON has the shape:
    {"inputs": [str, ...], "outputs": [str, ...], "fn_name": str | None}

LiveCodeBench mixes TWO test-case formats and the scorer must dispatch between
them based on `fn_name`:

  * `fn_name is None`  → stdio problems (Codeforces / AtCoder style). The input
                         string is piped to the model's program via stdin;
                         the program's stdout is compared to the expected output.

  * `fn_name is not None` → function-call problems (LeetCode-style). The model
                            is expected to define `class Solution` with the
                            named method. `inputs[i]` is a newline-delimited
                            list of Python-literal arguments; `outputs[i]` is a
                            Python literal of the expected return value.

We execute every test case in a sandboxed subprocess (one subprocess per test).
This is slow but strongly isolates side effects and lets us kill runaway loops
without leaking state into the scorer.
"""

import base64
import json
import os
import pickle
import signal
import subprocess
import sys
import time
import traceback
import zlib

# Per-sample global timeout guard. `max_tests` defaults to running EVERY test,
# so a pathological problem with 44 tests * 5s = 220s worst case; we cap at
# 300s (5 minutes) as a hard ceiling. Individual tests still time out at 5s.
GLOBAL_TIMEOUT = 300


def _decode_test_cases(ground_truth: str) -> dict:
    """Decode the compressed ground-truth blob into an in_outs dict."""
    raw = base64.b64decode(ground_truth)
    decompressed = zlib.decompress(raw, 15)
    s = pickle.loads(decompressed)
    return json.loads(s)


def _extract_code(solution_str: str) -> str:
    """Extract python code from markdown code block, or return as-is."""
    # Try ```python ... ``` first
    parts = solution_str.split("```python")
    if len(parts) > 1:
        code = parts[-1].split("```")[0]
        return code.strip()
    # Try generic ``` ... ```
    parts = solution_str.split("```")
    if len(parts) >= 3:
        return parts[-2].strip()
    return solution_str.strip()


def _run_subprocess(code: str, stdin: str, timeout: int) -> tuple[bool, str]:
    """Run `code` as a python subprocess, return (ok, stdout).

    Never raises. On timeout / crash returns (False, ""). Uses a fresh process
    group so we can SIGKILL the entire tree on hang.
    """
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, _ = proc.communicate(input=stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
            return False, ""
        return proc.returncode == 0, stdout
    except Exception:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        return False, ""


def _run_stdio_test(code: str, test_input: str, expected_output: str, timeout: int) -> bool:
    """stdio-style: pipe input to program, compare stdout to expected."""
    ok, stdout = _run_subprocess(code, test_input, timeout=timeout)
    if not ok:
        return False
    return stdout.strip() == expected_output.strip()


# --- Function-call (LeetCode-style) harness ---
#
# We wrap the model's `class Solution` in a small runner that:
#   1. Reads all argument literals from stdin (one per line, in order).
#   2. Parses each via ast.literal_eval (safe: no code execution).
#   3. Invokes Solution().<fn_name>(*args).
#   4. Prints repr(result) so the parent can compare to expected repr.
_FN_HARNESS_TEMPLATE = r"""
import ast, sys
from typing import List, Optional, Any, Dict, Tuple, Set

# --- BEGIN MODEL CODE ---
{model_code}
# --- END MODEL CODE ---

def _parse_arg(line):
    try:
        return ast.literal_eval(line)
    except Exception:
        # Some LCB inputs are bare tokens; try wrapping as a string literal.
        try:
            return ast.literal_eval('"' + line + '"')
        except Exception:
            return line

def _main():
    raw = sys.stdin.read()
    lines = raw.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    args = [_parse_arg(l) for l in lines]
    sol = Solution()
    result = getattr(sol, {fn_name_repr})(*args)
    sys.stdout.write(repr(result))

try:
    _main()
except Exception as e:
    sys.stderr.write("{{}}: {{}}\n".format(type(e).__name__, e))
    sys.exit(1)
"""


def _equal_pyliteral(actual_str: str, expected_str: str) -> bool:
    """Compare two Python-literal strings for semantic equality.

    Both sides come from repr() of the return value / a literal in the test
    file. Falls back to string equality for values that don't literal-eval.
    """
    import ast

    def _to_val(s: str):
        try:
            return ast.literal_eval(s.strip())
        except Exception:
            return s.strip()

    a = _to_val(actual_str)
    e = _to_val(expected_str)
    return _deep_eq(a, e)


def _deep_eq(a, b) -> bool:
    if type(a) is float or type(b) is float:
        try:
            fa, fb = float(a), float(b)
            return abs(fa - fb) <= 1e-6 + 1e-6 * max(abs(fa), abs(fb))
        except (TypeError, ValueError):
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_deep_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_deep_eq(a[k], b[k]) for k in a)
    return a == b


def _run_fn_test(code: str, fn_name: str, test_input: str, expected_output: str, timeout: int) -> bool:
    """function-call-style: build harness, execute, compare repr."""
    harness = _FN_HARNESS_TEMPLATE.format(
        model_code=code,
        fn_name_repr=repr(fn_name),
    )
    ok, stdout = _run_subprocess(harness, test_input, timeout=timeout)
    if not ok:
        return False
    return _equal_pyliteral(stdout.strip(), expected_output.strip())


def compute_score(
    solution_str: str,
    ground_truth: str,
    max_tests: int | None = None,
    timeout: int = 5,
) -> dict:
    """Compute reward for a LiveCodeBench code-generation problem.

    Args:
        solution_str: The model's full response (should contain ```python ... ```).
        ground_truth: The encoded test-case blob from the parquet.
        max_tests: Cap on number of test cases. `None` (default) runs every
            test; that's what the paper reports. Set to a small int only for
            training-time efficiency where speed matters more than fidelity.
        timeout: Per-test timeout in seconds.

    Returns:
        Dict with:
            - score: 2 * (passed / total) - 1, in [-1, +1].
            - acc:   passed / total, in [0, 1] (float, not bool).
            - pred:  human-readable pass-rate string with any error tags.
    """
    try:
        test_cases = _decode_test_cases(ground_truth)
    except Exception:
        traceback.print_exc(5)
        return {"score": -1.0, "acc": 0.0, "pred": "[DECODE_ERROR]"}

    code = _extract_code(solution_str)
    if not code:
        return {"score": -1.0, "acc": 0.0, "pred": "[NO_CODE]"}

    inputs = test_cases["inputs"]
    outputs = test_cases["outputs"]
    fn_name = test_cases.get("fn_name")

    total_available = len(inputs)
    total = total_available if max_tests is None else min(total_available, max_tests)

    if total == 0:
        return {"score": -1.0, "acc": 0.0, "pred": "[NO_TESTS]"}

    passed = 0
    start_time = time.time()

    for i in range(total):
        if time.time() - start_time > GLOBAL_TIMEOUT:
            return {
                "score": 2.0 * (passed / total) - 1.0,
                "acc": passed / total,
                "pred": f"pass_rate={passed}/{total}[GLOBAL_TIMEOUT@{i}]",
            }

        if fn_name:
            ok = _run_fn_test(code, fn_name, inputs[i], outputs[i], timeout=timeout)
        else:
            ok = _run_stdio_test(code, inputs[i], outputs[i], timeout=timeout)
        if ok:
            passed += 1

    acc = passed / total
    score = 2.0 * acc - 1.0
    return {
        "score": score,
        "acc": acc,
        "pred": f"pass_rate={passed}/{total}" + (f" [fn={fn_name}]" if fn_name else ""),
    }
