"""Compute automatic reward scores for generated C code.

This script provides a suite of automatic checks that produce a scalar reward for
each (prompt, completion) pair. The reward is a weighted sum of sub-scores:

  - compile_pass (0/1): does `gcc -fsyntax-only` succeed?
  - warnings_penalty: normalized negative score proportional to number of warnings
    from `gcc -Wall` and `cppcheck`/`clang-tidy` if available.
  - sanitizer_flag (0/1): if compilation with -fsanitize=address/undefined passes (optional)
  - static_security_penalty: penalize uses of banned/unsafe functions or patterns (gets, strcpy, strcat, system(...))
  - formatting_bonus: small positive if clang-format produces no changes (style)

Usage:
  python training/compute_rewards.py --in generated.jsonl --out rewards.jsonl --timeout 5

Input format (JSONL): each line is an object with at least 'prompt' and 'completion' fields.
Output format (JSONL): each line is the input object augmented with a 'reward' float and breakdown.

Security note: generated code may be malicious. The script only performs static checks and
compilation in a sandboxed Docker/VM is recommended for any dynamic execution or sanitizer runs.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, Tuple


UNSAFE_PATTERNS = [
    r"\bgets\s*\(",
    r"\bstrcpy\s*\(",
    r"\bstrcat\s*\(",
    r"\bsystem\s*\(",
    r"\bpopen\s*\(",
    r"\bexecl\s*\(",
    r"#\s*include\s*<unistd.h>",
]


def run_cmd(cmd, timeout=5):
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return -127, '', f'command not found: {cmd[0]}'
    except subprocess.TimeoutExpired as e:
        return -124, '', f'timeout after {timeout}s'


def gcc_syntax_check(code: str, timeout=5) -> Tuple[bool, str]:
    # Use -fsyntax-only to check syntax only
    with tempfile.NamedTemporaryFile('w', suffix='.c', delete=False, encoding='utf-8') as f:
        f.write(code)
        fname = f.name
    cmd = ['gcc', '-fsyntax-only', fname]
    rc, out, err = run_cmd(cmd, timeout=timeout)
    try:
        os.remove(fname)
    except Exception:
        pass
    return rc == 0, (out + '\n' + err).strip()


def gcc_full_warnings(code: str, timeout=5) -> Tuple[bool, str, int]:
    # compile with -Wall to capture warnings (no link)
    with tempfile.NamedTemporaryFile('w', suffix='.c', delete=False, encoding='utf-8') as f:
        f.write(code)
        fname = f.name
    cmd = ['gcc', '-Wall', '-fsyntax-only', fname]
    rc, out, err = run_cmd(cmd, timeout=timeout)
    try:
        os.remove(fname)
    except Exception:
        pass
    warnings = 0
    combined = out + '\n' + err
    # crude heuristic: count 'warning:' occurrences
    warnings = combined.count('warning:')
    return rc == 0, combined.strip(), warnings


def run_cppcheck(code: str, timeout=5) -> Tuple[bool, str, int]:
    if shutil.which('cppcheck') is None:
        return False, 'cppcheck not found', 0
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'tmp.c')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(code)
        cmd = ['cppcheck', '--enable=all', '--quiet', path]
        rc, out, err = run_cmd(cmd, timeout=timeout)
        combined = out + '\n' + err
        issues = 0
        # cppcheck prints 'error:' or 'warning:' lines
        issues = combined.count('error:') + combined.count('warning:')
        return rc == 0, combined.strip(), issues


def run_clang_tidy(code: str, timeout=5) -> Tuple[bool, str, int]:
    if shutil.which('clang-tidy') is None:
        return False, 'clang-tidy not found', 0
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'tmp.c')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(code)
        # clang-tidy needs compile_commands or -checks; we run basic check
        cmd = ['clang-tidy', path, '--', '-std=c11']
        rc, out, err = run_cmd(cmd, timeout=timeout)
        combined = out + '\n' + err
        issues = combined.count('warning:') + combined.count('error:')
        return rc == 0, combined.strip(), issues


def unsafe_pattern_score(code: str) -> Tuple[float, Dict[str, int]]:
    matches = {}
    penalty = 0
    for p in UNSAFE_PATTERNS:
        found = len(re.findall(p, code))
        if found:
            matches[p] = found
            penalty += found
    # normalize: more than 0 matches -> penalty between 0 and 1
    score = 1.0
    if penalty > 0:
        score = max(0.0, 1.0 - 0.25 * penalty)
    return score, matches


def clang_format_unchanged(code: str, timeout=3) -> Tuple[bool, str]:
    if shutil.which('clang-format') is None:
        return False, 'clang-format not found'
    with tempfile.NamedTemporaryFile('w', suffix='.c', delete=False, encoding='utf-8') as f:
        f.write(code)
        fname = f.name
    cmd = ['clang-format', fname]
    rc, out, err = run_cmd(cmd, timeout=timeout)
    try:
        os.remove(fname)
    except Exception:
        pass
    if rc != 0:
        return False, err.strip()
    formatted = out
    unchanged = (formatted.strip() == code.strip())
    return unchanged, ''


def compute_reward_for_code(code: str, timeout=5, weights=None) -> Dict:
    if weights is None:
        weights = {
            'compile': 3.0,
            'warnings': -0.5,
            'cppcheck': -0.5,
            'clang_tidy': -0.5,
            'unsafe': -1.5,
            'format': 0.1,
        }

    out = {}
    # syntax-only check
    compile_ok, compile_out = gcc_syntax_check(code, timeout=timeout)
    out['compile_ok'] = compile_ok
    out['compile_out'] = compile_out

    # warnings
    warn_ok, warn_out, warnings = gcc_full_warnings(code, timeout=timeout)
    out['gcc_warnings'] = warnings
    out['gcc_warn_out'] = warn_out

    # cppcheck
    cpp_ok, cpp_out, cpp_issues = run_cppcheck(code, timeout=timeout)
    out['cppcheck_issues'] = cpp_issues
    out['cppcheck_out'] = cpp_out

    # clang-tidy
    tidy_ok, tidy_out, tidy_issues = run_clang_tidy(code, timeout=timeout)
    out['clang_tidy_issues'] = tidy_issues
    out['clang_tidy_out'] = tidy_out

    # unsafe patterns
    unsafe_score, matches = unsafe_pattern_score(code)
    out['unsafe_matches'] = matches

    # formatting bonus
    fmt_ok, fmt_out = clang_format_unchanged(code)
    out['format_ok'] = fmt_ok

    # Compose reward
    reward = 0.0
    reward += weights['compile'] * (1.0 if compile_ok else 0.0)
    reward += weights['warnings'] * warnings
    reward += weights['cppcheck'] * cpp_issues
    reward += weights['clang_tidy'] * tidy_issues
    # unsafe patterns reduce reward multiplicatively
    reward += weights['unsafe'] * (0.0 if len(matches) == 0 else 1.0)
    reward += weights['format'] * (1.0 if fmt_ok else 0.0)

    # clamp reward
    out['reward_raw'] = reward
    out['reward'] = float(max(-10.0, min(10.0, reward)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--timeout', type=int, default=5)
    parser.add_argument('--max', type=int, default=None)
    args = parser.parse_args()

    with open(args.infile, 'r', encoding='utf-8') as inf, open(args.out, 'w', encoding='utf-8') as outf:
        for i, line in enumerate(inf):
            if args.max and i >= args.max:
                break
            obj = json.loads(line)
            code = obj.get('completion') or obj.get('code') or ''
            score = compute_reward_for_code(code, timeout=args.timeout)
            obj['reward'] = score['reward']
            obj['reward_details'] = score
            outf.write(json.dumps(obj) + '\n')


if __name__ == '__main__':
    main()
