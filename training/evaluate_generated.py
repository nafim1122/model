"""
Evaluate generated C programs for compilation, static analysis, style, and runtime.

Usage (PowerShell):
  python training/evaluate_generated.py --generated training/generated.jsonl --out training/eval_report.json

The script will attempt to run system tools (gcc, cppcheck, clang-tidy, clang-format).
If tools are missing it will note that in the report and continue gracefully.

Output: JSON report summarizing metrics and per-sample details.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def has_cmd(cmd):
    return shutil.which(cmd) is not None


def run_cmd(cmd, timeout=10, cwd=None):
    try:
        proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, cwd=cwd)
        return proc.returncode, proc.stdout.decode(errors='ignore'), proc.stderr.decode(errors='ignore')
    except subprocess.TimeoutExpired:
        return -1, '', 'TIMEOUT'
    except Exception as e:
        return -2, '', str(e)


def compile_c(source_path: Path, out_bin: Path):
    # Basic compilation (no sanitizers by default)
    cmd = f"gcc -std=c11 -O2 -Wall -Wextra -o {shlex.quote(str(out_bin))} {shlex.quote(str(source_path))}"
    return run_cmd(cmd, timeout=15)


def compile_with_sanitizers(source_path: Path, out_bin: Path):
    cmd = f"gcc -std=c11 -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -g -o {shlex.quote(str(out_bin))} {shlex.quote(str(source_path))}"
    return run_cmd(cmd, timeout=20)


def run_binary(bin_path: Path, timeout=5):
    start = time.time()
    try:
        proc = subprocess.run([str(bin_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        duration = time.time() - start
        return proc.returncode, proc.stdout.decode(errors='ignore'), proc.stderr.decode(errors='ignore'), duration
    except subprocess.TimeoutExpired:
        return -1, '', 'TIMEOUT', timeout
    except Exception as e:
        return -2, '', str(e), 0.0


def run_cppcheck(source_path: Path):
    cmd = f"cppcheck --enable=all --quiet {shlex.quote(str(source_path))}"
    return run_cmd(cmd, timeout=10)


def run_clang_tidy(source_path: Path):
    cmd = f"clang-tidy {shlex.quote(str(source_path))} --"
    return run_cmd(cmd, timeout=15)


def check_clang_format(source_path: Path):
    # Check whether running clang-format would change the file
    cmd = f"clang-format {shlex.quote(str(source_path))}"
    rc, out, err = run_cmd(cmd, timeout=10)
    if rc != 0:
        return False, out, err
    # compare formatted output to original
    orig = source_path.read_text(encoding='utf-8', errors='ignore')
    changed = out != orig
    return (not changed), out, err


FORBIDDEN_PATTERNS = [
    r"\bgets\s*\(",
    r"\bsystem\s*\(",
    r"\bexec\w*\s*\(",
    r"#include\s*<sys/socket.h>",
    r"__asm__",
    r"fork\s*\(",
]


def check_forbidden(code: str):
    for p in FORBIDDEN_PATTERNS:
        if re.search(p, code):
            return True, p
    return False, None


def evaluate_record(record, tools):
    prompt = record.get('prompt','')
    completion = record.get('completion','')
    with tempfile.TemporaryDirectory() as td:
        tdpath = Path(td)
        src = tdpath / "submission.c"
        src.write_text(completion, encoding='utf-8')
        details = {}
        # forbidden patterns
        forbidden, pattern = check_forbidden(completion)
        details['forbidden'] = bool(forbidden)
        details['forbidden_pattern'] = pattern

        # compile
        if tools['gcc']:
            rc, out, err = compile_c(src, tdpath / 'bin')
            details['gcc_returncode'] = rc
            details['gcc_stdout'] = out
            details['gcc_stderr'] = err
            details['compiled'] = (rc == 0)
        else:
            details['gcc_available'] = False
            details['compiled'] = False

        # compile with sanitizers
        if tools['gcc']:
            rc2, out2, err2 = compile_with_sanitizers(src, tdpath / 'bin.asan')
            details['asan_compile_returncode'] = rc2
            details['asan_compile_stderr'] = err2
            details['asan_compiled'] = (rc2 == 0)
            if details['asan_compiled']:
                rc_run, sout, serr, dur = run_binary(tdpath / 'bin.asan')
                details['asan_run_returncode'] = rc_run
                details['asan_run_stdout'] = sout
                details['asan_run_stderr'] = serr
                details['asan_run_duration'] = dur
                details['asan_ok'] = (rc_run == 0 and 'ERROR: AddressSanitizer' not in serr and 'undefined behavior' not in serr)
            else:
                details['asan_ok'] = False
        else:
            details['asan_available'] = False

        # run binary for runtime performance if compiled
        if details.get('compiled'):
            rc_run, sout, serr, dur = run_binary(tdpath / 'bin')
            details['run_returncode'] = rc_run
            details['run_stdout'] = sout
            details['run_stderr'] = serr
            details['run_duration'] = dur
        
        # static analyzers
        if tools['cppcheck']:
            rc, out, err = run_cppcheck(src)
            details['cppcheck_returncode'] = rc
            details['cppcheck_stderr'] = err
            details['cppcheck_ok'] = (rc == 0)
        else:
            details['cppcheck_available'] = False

        if tools['clang_tidy']:
            rc, out, err = run_clang_tidy(src)
            details['clang_tidy_returncode'] = rc
            details['clang_tidy_stdout'] = out
            details['clang_tidy_stderr'] = err
        else:
            details['clang_tidy_available'] = False

        # style
        if tools['clang_format']:
            ok, out, err = check_clang_format(src)
            details['clang_format_ok'] = ok
        else:
            details['clang_format_available'] = False

        return details


def summarize(all_details):
    n = len(all_details)
    compiled = sum(1 for d in all_details if d.get('compiled'))
    asan_ok = sum(1 for d in all_details if d.get('asan_ok'))
    cpp_ok = sum(1 for d in all_details if d.get('cppcheck_ok'))
    forbidden = sum(1 for d in all_details if d.get('forbidden'))
    run_ok = sum(1 for d in all_details if d.get('run_returncode') == 0)
    durations = [d.get('run_duration') for d in all_details if d.get('run_duration') is not None]
    avg_time = sum(durations)/len(durations) if durations else None
    return {
        'total': n,
        'compiled_count': compiled,
        'compiled_rate': compiled / n if n else 0.0,
        'asan_ok_count': asan_ok,
        'asan_ok_rate': asan_ok / n if n else 0.0,
        'cppcheck_ok_count': cpp_ok,
        'cppcheck_ok_rate': cpp_ok / n if n else 0.0,
        'forbidden_count': forbidden,
        'forbidden_rate': forbidden / n if n else 0.0,
        'run_ok_count': run_ok,
        'run_ok_rate': run_ok / n if n else 0.0,
        'avg_run_time': avg_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', required=True, help='JSONL of {prompt,completion} generated by model')
    parser.add_argument('--out', required=True, help='Output JSON report path')
    parser.add_argument('--max', type=int, default=None, help='Max records to evaluate')
    args = parser.parse_args()

    gen_path = Path(args.generated)
    if not gen_path.exists():
        raise SystemExit(f"Generated file not found: {gen_path}")

    tools = {
        'gcc': has_cmd('gcc'),
        'cppcheck': has_cmd('cppcheck'),
        'clang_tidy': has_cmd('clang-tidy') or has_cmd('clang-tidy.exe'),
        'clang_format': has_cmd('clang-format') or has_cmd('clang-format.exe')
    }
    print('Tool availability:', tools)

    all_details = []
    with open(gen_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if args.max is not None and i >= args.max:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            details = evaluate_record(rec, tools)
            details['index'] = i
            all_details.append(details)

    summary = summarize(all_details)
    report = {'summary': summary, 'details': all_details, 'tools': tools}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Wrote report to', out_path)


if __name__ == '__main__':
    main()
