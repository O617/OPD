"""Reward scoring for GPQA (Graduate-Level Google-Proof Q&A) multiple-choice questions.

The ground truth is a single letter (A/B/C/D).
The instructed format is `Answer: $LETTER`, but reasoning-tuned models routinely
emit variants: `\\boxed{A}`, `The answer is A.`, `Final answer: (B)`, `(D)` alone,
`Answer: **C**`, `Answer: A.`, etc. The extractor below tries them in priority order
and picks the last match, which is what a human grader would do.

We deliberately try the high-confidence patterns first (\\boxed, explicit
"answer:" phrases) and fall back to weaker heuristics (bare "(A)" / bare letter
at end) only when nothing more explicit matches. The bare-letter fallback is
guarded by a negative lookbehind so we don't accidentally grab a variable name
like `x = A` in reasoning text.
"""

import re

_LETTER = r"[A-D]"

# Patterns are tried in order; the LAST regex match wins within a pattern.
# Any pattern that matches wins over lower-priority patterns.
_PATTERNS = [
    # \boxed{A} / \boxed{ (A) } — highest confidence, matches how math-tuned
    # models terminate an answer.
    rf"\\boxed\{{\s*[\(\[]?\s*({_LETTER})\s*[\)\]]?\s*\}}",

    # Explicit answer-declaration phrases. Covers:
    #   "Answer: A", "answer is A", "Answer: (A)", "Answer: **B**", "Answer: A."
    #   "Final answer: A", "Correct answer is B", "The answer is: C"
    #   "Final: A", "Result: A", "Conclusion: A"
    # Optional leading qualifier, optional "is", optional colon,
    # optional bold/italic wrappers, optional parens/brackets.
    rf"(?i)(?:the\s+|final\s+|correct\s+|my\s+)*"
    rf"(?:answer|final|result|conclusion|solution)\s*"
    rf"(?:is)?\s*[:：]?\s*[\*_`]*\s*[\(\[]?\s*({_LETTER})\s*[\)\]]?\s*[\*_`.]*",

    # "option A", "choice A", "letter A", "select A", "pick A"
    rf"(?i)(?:option|choice|letter|select(?:ed)?|pick(?:ed)?)\s+[\(\[]?({_LETTER})[\)\]]?\b",

    # Standalone "(A)" or "[A]" at end of string, optional trailing period.
    rf"[\(\[]({_LETTER})[\)\]]\s*\.?\s*$",

    # Bare letter at the very end of the response, optional trailing period.
    # Guarded by a negative lookbehind: reject cases where the letter is
    # preceded by "= " (a variable assignment like "x = A") — those aren't
    # answer commits. Also reject if preceded by a lowercase letter with no
    # space (i.e. mid-word capital like "MoleculeA").
    rf"(?<![=a-z])\b({_LETTER})\b\s*\.?\s*$",
]


def extract_answer(solution_str: str) -> str:
    """Return the extracted letter A/B/C/D, or empty string if none found."""
    if not solution_str:
        return ""

    # If the response has a </think> boundary, prefer text AFTER it: the model
    # may have considered wrong answers during reasoning, and the final commit
    # is what we should grade.
    tail = solution_str
    if "</think>" in solution_str:
        after_think = solution_str.rsplit("</think>", 1)[-1]
        if after_think.strip():
            tail = after_think

    for pat in _PATTERNS:
        matches = re.findall(pat, tail)
        if matches:
            return matches[-1].upper()

    # If nothing matched after </think>, fall back to the whole string. This
    # is a defense against models that emit their answer inside <think>.
    if tail is not solution_str:
        for pat in _PATTERNS:
            matches = re.findall(pat, solution_str)
            if matches:
                return matches[-1].upper()
    return ""


def compute_score(solution_str: str, ground_truth: str) -> dict:
    """Compute reward for GPQA multiple-choice answers.

    Args:
        solution_str: The model's full response string.
        ground_truth: The correct answer letter (e.g. "A", "B", "C", "D").

    Returns:
        Dict with score, acc, pred, and abstained (True iff no letter was
        extractable, which is different from an extracted-but-wrong answer).
    """
    gt = ground_truth.strip().upper()
    pred = extract_answer(solution_str)
    abstained = pred == ""
    correct = (pred == gt) and not abstained

    return {
        "score": 1.0 if correct else -1.0,
        "acc": correct,
        "pred": pred if pred else "[NO_ANSWER]",
        "abstained": abstained,
    }
