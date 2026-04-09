"""Deep audit of training data quality."""
import json
import re
import sys

files = ["data/animtoon_train_10k.jsonl", "data/animtoon_train_90k.jsonl", "data/animtoon_train_87k.jsonl"]
total = 0
has_raw_dict = 0
has_arrow_rot = 0
has_arrow_scale = 0
has_arrow_opacity = 0
has_no_animation = 0  # only static props
samples_with_rot = 0
samples_with_scale = 0
samples_with_opacity = 0

for fpath in files:
    try:
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                out = rec["output"]
                total += 1

                # Check each line
                for ol in out.split("\n"):
                    ol = ol.strip()
                    if ol.startswith("rot "):
                        samples_with_rot += 1
                        if "\u2192" in ol:
                            has_arrow_rot += 1
                        elif "{" in ol or "'" in ol:
                            has_raw_dict += 1
                    elif ol.startswith("scale "):
                        samples_with_scale += 1
                        if "\u2192" in ol:
                            has_arrow_scale += 1
                        elif "{" in ol or "'" in ol:
                            has_raw_dict += 1
                    elif ol.startswith("opacity "):
                        samples_with_opacity += 1
                        if "\u2192" in ol:
                            has_arrow_opacity += 1
                        elif "{" in ol or "'" in ol:
                            has_raw_dict += 1
    except FileNotFoundError:
        print(f"Skipping {fpath}")

print(f"Total samples: {total}")
print()
print(f"Samples with rot:     {samples_with_rot}")
print(f"  Arrow format (good):  {has_arrow_rot}")
print(f"Samples with scale:   {samples_with_scale}")
print(f"  Arrow format (good):  {has_arrow_scale}")
print(f"Samples with opacity: {samples_with_opacity}")
print(f"  Arrow format (good):  {has_arrow_opacity}")
print()
print(f"Total raw dict lines: {has_raw_dict}")
print(f"Contamination rate:   {has_raw_dict}/{samples_with_rot+samples_with_scale+samples_with_opacity} animated props")
print()

# Show examples of rot/scale/opacity lines
count = 0
with open("data/animtoon_train_10k.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line.strip())
        out = rec["output"]
        for ol in out.split("\n"):
            ol = ol.strip()
            if ol.startswith("rot ") and count < 5:
                print(f"ROT: {ol[:150]}")
                count += 1
            if ol.startswith("scale ") and count < 10 and count >= 5:
                print(f"SCALE: {ol[:150]}")
                count += 1
            if ol.startswith("opacity ") and count < 15 and count >= 10:
                print(f"OPACITY: {ol[:150]}")
                count += 1
