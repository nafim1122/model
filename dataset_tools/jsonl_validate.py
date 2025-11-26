#!/usr/bin/env python3
"""
jsonl_validate.py
Simple checks for JSONL dataset: counts, empty fields, prompt/completion length stats, suspicious binary chars.
"""
import sys, json, argparse, statistics, re
from pathlib import Path

def is_binary_string(s):
    # heuristics: presence of NUL or many non-printable chars
    if '\x00' in s:
        return True
    nonprint = sum(1 for c in s if ord(c) < 32 and c not in '\n\t\r')
    return nonprint > 0

p = argparse.ArgumentParser()
p.add_argument('--jsonl', required=True)
p.add_argument('--sample', type=int, default=5)
args = p.parse_args()
path = Path(args.jsonl)
if not path.exists():
    print('File not found:', path)
    sys.exit(2)

lengths_prompt = []
lengths_comp = []
count = 0
empty_prompt = 0
empty_comp = 0
binary_count = 0
samples = []
max_prompt = (None,0)
max_comp = (None,0)

with path.open('r', encoding='utf-8') as fh:
    for i,line in enumerate(fh,1):
        line=line.strip()
        if not line:
            continue
        try:
            rec=json.loads(line)
        except Exception as e:
            print('JSON parse error on line', i, e)
            continue
        prompt = rec.get('prompt','')
        comp = rec.get('completion','')
        count += 1
        if not prompt.strip():
            empty_prompt += 1
        if not comp.strip():
            empty_comp += 1
        if is_binary_string(prompt) or is_binary_string(comp):
            binary_count += 1
        lp = len(prompt.split())
        lc = len(comp.split())
        lengths_prompt.append(lp)
        lengths_comp.append(lc)
        if lp > max_prompt[1]: max_prompt = (i, lp)
        if lc > max_comp[1]: max_comp = (i, lc)
        if len(samples) < args.sample:
            samples.append((i, prompt[:200].replace('\n','\\n'), comp[:200].replace('\n','\\n')))

print('Total records:', count)
print('Empty prompts:', empty_prompt)
print('Empty completions:', empty_comp)
print('Binary-like records:', binary_count)
if lengths_prompt:
    print('Prompt tokens: mean %.1f median %d min %d max %d' % (statistics.mean(lengths_prompt), statistics.median(lengths_prompt), min(lengths_prompt), max(lengths_prompt)))
if lengths_comp:
    print('Completion tokens: mean %.1f median %d min %d max %d' % (statistics.mean(lengths_comp), statistics.median(lengths_comp), min(lengths_comp), max(lengths_comp)))
print('Longest prompt at line', max_prompt[0], 'length', max_prompt[1])
print('Longest completion at line', max_comp[0], 'length', max_comp[1])
print('\nSample records:')
for i,p,c in samples:
    print('---', i)
    print('PROMPT:', p)
    print('COMPLETION:', c)

# simple heuristic: percent of small completions (<3 tokens)
small_comp = sum(1 for x in lengths_comp if x < 3)
print('\nSmall completions (<3 tokens):', small_comp, '(', '%.2f'%(100*small_comp/len(lengths_comp) if lengths_comp else 0), '%)')
