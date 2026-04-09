"""Check what format rot/scale/opacity actually use."""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

arrow = "\u2192"
rot_formats = {"arrow": 0, "static": 0, "dict": 0, "other": 0}
scale_formats = {"arrow": 0, "static": 0, "dict": 0, "other": 0}
opa_formats = {"arrow": 0, "static": 0, "dict": 0, "other": 0}

rot_examples = {"arrow": [], "static": [], "dict": [], "other": []}
scale_examples = {"arrow": [], "static": [], "dict": [], "other": []}
opa_examples = {"arrow": [], "static": [], "dict": [], "other": []}

for fpath in ["data/animtoon_train_10k.jsonl"]:
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            for ol in rec["output"].split("\n"):
                ol = ol.strip()
                for prefix, fmt_dict, ex_dict in [
                    ("rot ", rot_formats, rot_examples),
                    ("scale ", scale_formats, scale_examples),
                    ("opacity ", opa_formats, opa_examples),
                ]:
                    if ol.startswith(prefix):
                        val = ol[len(prefix):]
                        if arrow in val:
                            fmt_dict["arrow"] += 1
                            if len(ex_dict["arrow"]) < 2:
                                ex_dict["arrow"].append(ol)
                        elif "{" in val or "'" in val:
                            fmt_dict["dict"] += 1
                            if len(ex_dict["dict"]) < 2:
                                ex_dict["dict"].append(ol)
                        elif val.strip().startswith("[") or val.strip().replace(".", "").replace("-", "").isdigit():
                            fmt_dict["static"] += 1
                            if len(ex_dict["static"]) < 2:
                                ex_dict["static"].append(ol)
                        else:
                            fmt_dict["other"] += 1
                            if len(ex_dict["other"]) < 2:
                                ex_dict["other"].append(ol)

for name, fmt, ex in [("ROT", rot_formats, rot_examples),
                       ("SCALE", scale_formats, scale_examples),
                       ("OPACITY", opa_formats, opa_examples)]:
    print(f"\n{name}: {fmt}")
    for k, v in ex.items():
        for e in v:
            print(f"  [{k}] {e[:150]}")
