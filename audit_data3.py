"""Deep check: find ALL non-arrow animated keyframes."""
import json, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

arrow = "\u2192"
total_lines = 0
pattern_counts = {}

with open("data/animtoon_train_10k.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line.strip())
        for ol in rec["output"].split("\n"):
            ol = ol.strip()
            for prefix in ["rot ", "scale ", "opacity ", "pos "]:
                if ol.startswith(prefix):
                    val = ol[len(prefix):]
                    total_lines += 1

                    if arrow in val and "{" not in val:
                        key = "GOOD: arrow format"
                    elif val.startswith("[") and "{" not in val:
                        key = "GOOD: static array"
                    elif val.replace(".", "").replace("-", "").strip().isdigit():
                        key = "GOOD: static number"
                    elif "{'s'" in val or '{"s"' in val:
                        key = "BAD: raw dict with s key"
                    elif "{" in val:
                        key = "BAD: has curly brace"
                    elif "'" in val and "ease" not in val:
                        key = "BAD: has quote"
                    elif arrow in val and "{" in val:
                        key = "MIXED: arrow + dict"
                    else:
                        key = f"UNKNOWN: {val[:80]}"

                    if key not in pattern_counts:
                        pattern_counts[key] = 0
                    pattern_counts[key] += 1

print(f"Total animated property lines: {total_lines}")
print()
for k, v in sorted(pattern_counts.items(), key=lambda x: -x[1]):
    pct = v / total_lines * 100
    print(f"  {v:6d} ({pct:5.1f}%) {k}")
