"""Convert v3 training data (float format) to v4 (integer quantization).

Transforms:
  - Time: 0.0-1.0 floats → 0-1000 integers
  - Position: [0.5,0.5] → [500,500]
  - Detects repeating patterns → converts to loop format
  - Adds parent= hints where layers share position
"""
import json
import re
import sys

ARROW = "\u2192"


def convert_line_v4(line):
    """Convert a single AnimTOON property line from v3 to v4 format."""
    line = line.strip()

    if not line or line.startswith('#') or line.startswith('anim ') or line.startswith('layer'):
        return line

    # Only convert lines with arrows (keyframe data)
    if ARROW not in line:
        return line

    prop = line.split()[0]  # rot, pos, scale, opacity

    # Extract ease
    ease = ""
    ease_match = re.search(r'ease=(\w+)', line)
    if ease_match:
        ease = f" ease={ease_match.group(1)}"
    clean = re.sub(r'\s*ease=\w+', '', line).strip()

    # Parse keyframes: time→value pairs
    pattern = r'([\d.]+)' + ARROW + r'(\[[\d.,\s\-]+\]|[\d.\-]+)'
    matches = list(re.finditer(pattern, clean))

    if not matches:
        return line

    kf_parts = []
    for m in matches:
        t_float = float(m.group(1))
        val_str = m.group(2)

        # Convert time to int 0-1000
        t_int = round(t_float * 1000) if t_float <= 1.0 else round(t_float)

        if val_str.startswith('['):
            # Position/scale array
            nums = re.findall(r'[\d.\-]+', val_str)
            int_vals = []
            for n in nums:
                v = float(n)
                # If value looks like 0-1 range (position), scale to 0-1000
                if prop == 'pos' and abs(v) <= 1.5:
                    int_vals.append(str(round(v * 1000)))
                else:
                    # Scale values stay as-is (already 0-100 range)
                    int_vals.append(str(round(v)))
            kf_parts.append(f"{t_int}{ARROW}[{','.join(int_vals)}]")
        else:
            # Scalar (rotation, opacity)
            v = float(val_str)
            kf_parts.append(f"{t_int}{ARROW}{round(v, 1)}")

    # Check for loop pattern: values repeat symmetrically
    if len(kf_parts) >= 3:
        vals_only = []
        for m in matches:
            vs = m.group(2)
            if not vs.startswith('['):
                vals_only.append(float(vs))

        if len(vals_only) >= 3 and vals_only[0] == vals_only[-1]:
            # Check if it's a simple oscillation
            if len(vals_only) == 3:
                delta = round(vals_only[1] - vals_only[0], 1)
                t_mid = round(float(matches[1].group(1)) * 1000)
                return f"  {prop} loop={t_mid} {'+' if delta >= 0 else ''}{delta} {'+' if -delta >= 0 else ''}{-delta}{ease}"

    return f"  {prop} {' '.join(kf_parts)}{ease}"


def convert_output_v4(output_text):
    """Convert full AnimTOON output from v3 to v4."""
    lines = output_text.split('\n')
    converted = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('anim '):
            converted.append(stripped)
        elif stripped.startswith('layer'):
            converted.append(stripped)
        elif stripped.startswith('fill ') or stripped.startswith('stroke ') or stripped.startswith('path '):
            converted.append(f"  {stripped}")
        elif ARROW in stripped:
            converted.append(convert_line_v4(stripped))
        elif stripped.startswith('pos [') or stripped.startswith('scale ['):
            # Static position/scale — convert values
            if stripped.startswith('pos ['):
                m = re.match(r'pos \[([\d.\-]+),([\d.\-]+)\]', stripped)
                if m:
                    x, y = float(m.group(1)), float(m.group(2))
                    if abs(x) <= 1.5 and abs(y) <= 1.5:
                        converted.append(f"  pos [{round(x*1000)},{round(y*1000)}]")
                    else:
                        converted.append(f"  {stripped}")
                else:
                    converted.append(f"  {stripped}")
            else:
                converted.append(f"  {stripped}")
        elif stripped:
            converted.append(f"  {stripped}")
        else:
            converted.append('')

    return '\n'.join(converted)


def convert_dataset(input_file, output_file, max_samples=None):
    """Convert entire training dataset from v3 to v4 format."""
    converted = 0
    loops_found = 0

    with open(input_file, encoding='utf-8-sig') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for i, line in enumerate(fin):
            if max_samples and i >= max_samples:
                break

            rec = json.loads(line.strip())
            output = rec.get('output', '')

            v4_output = convert_output_v4(output)
            loops = v4_output.count('loop=')
            loops_found += loops

            rec['output'] = v4_output
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            converted += 1

            if converted % 1000 == 0:
                print(f"  Converted {converted} samples ({loops_found} loops found)...")

    print(f"\nDone: {converted} samples converted to v4 format")
    print(f"Loop patterns found: {loops_found}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/character_training_mix.jsonl"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/character_training_v4.jsonl"

    convert_dataset(input_file, output_file)
