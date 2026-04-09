"""Convert DragonBones skeleton JSON to AnimTOON training data."""
import json
import os
import sys
import glob
import random

ARROW = "\u2192"


def parse_dragonbones(db_data):
    """Convert DragonBones skeleton to AnimTOON training pairs."""
    pairs = []

    armature_list = db_data.get('armature', [])

    for armature in armature_list:
        name = armature.get('name', 'character')
        bones = armature.get('bone', [])
        animations = armature.get('animation', [])

        bone_names = [b.get('name', '') for b in bones]

        for anim in animations:
            anim_name = anim.get('name', 'animation')
            duration = anim.get('duration', 30)
            bone_timelines = anim.get('bone', [])

            if not bone_timelines:
                continue

            # Build AnimTOON
            dur = max(30, duration)
            lines = [f"anim fr=24 dur={dur}", ""]

            animated_count = 0

            for bt in bone_timelines:
                bone_name = bt.get('name', '')
                rot_frames = bt.get('rotateFrame', [])
                trans_frames = bt.get('translateFrame', [])
                scale_frames = bt.get('scaleFrame', [])

                lines.append(f"layer {bone_name} shape")
                has_anim = False

                # Rotation
                if len(rot_frames) >= 2:
                    parts = []
                    time_acc = 0
                    for rf in rot_frames:
                        t = round(time_acc / dur, 2) if dur > 0 else 0
                        t = min(1.0, t)
                        val = round(rf.get('rotate', 0), 1)
                        parts.append(f"{t}{ARROW}{val}")
                        time_acc += rf.get('duration', 1)
                    if len(parts) >= 2:
                        lines.append(f"  rot {' '.join(parts)} ease=smooth")
                        has_anim = True

                # Translation
                if len(trans_frames) >= 2:
                    parts = []
                    time_acc = 0
                    for tf in trans_frames:
                        t = round(time_acc / dur, 2) if dur > 0 else 0
                        t = min(1.0, t)
                        x = round(0.5 + tf.get('x', 0) / 512, 3)
                        y = round(0.5 + tf.get('y', 0) / 512, 3)
                        parts.append(f"{t}{ARROW}[{x},{y}]")
                        time_acc += tf.get('duration', 1)
                    if len(parts) >= 2:
                        lines.append(f"  pos {' '.join(parts)} ease=smooth")
                        has_anim = True

                # Scale
                if len(scale_frames) >= 2:
                    parts = []
                    time_acc = 0
                    for sf in scale_frames:
                        t = round(time_acc / dur, 2) if dur > 0 else 0
                        t = min(1.0, t)
                        sx = round(sf.get('x', 1.0) * 100, 1)
                        sy = round(sf.get('y', 1.0) * 100, 1)
                        parts.append(f"{t}{ARROW}[{sx},{sy}]")
                        time_acc += sf.get('duration', 1)
                    if len(parts) >= 2:
                        lines.append(f"  scale {' '.join(parts)} ease=smooth")
                        has_anim = True

                if has_anim:
                    animated_count += 1
                lines.append("")

            if animated_count < 2:
                continue

            animtoon = '\n'.join(lines).strip()

            # Build description
            desc = f"The video shows a {name} character performing a {anim_name} animation with {animated_count} animated body parts moving in coordination."

            # Build layer description
            layer_parts = []
            for i, bn in enumerate(bone_names[:10]):
                btype = 'body part'
                bl = bn.lower()
                if 'arm' in bl: btype = 'arm'
                elif 'leg' in bl: btype = 'leg'
                elif 'head' in bl: btype = 'head'
                elif 'body' in bl or 'torso' in bl: btype = 'body'
                elif 'eye' in bl: btype = 'eye'
                elif 'hair' in bl: btype = 'hair'
                layer_parts.append(f"{i+1}. {bn} ({btype})")

            layer_desc = f"SVG has {len(bone_names)} layers:\n  " + "\n  ".join(layer_parts)
            layer_desc += f"\nAnimate with {anim_name} motion."

            pairs.append({'instruction': desc, 'output': animtoon})
            pairs.append({'input': layer_desc, 'output': animtoon})

    return pairs


def convert_all(db_dir, output_file):
    """Convert all DragonBones files to training data."""
    all_pairs = []

    ske_files = glob.glob(os.path.join(db_dir, '**', '*_ske.json'), recursive=True)
    print(f"Found {len(ske_files)} DragonBones skeleton files")

    for filepath in ske_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            pairs = parse_dragonbones(data)
            if pairs:
                all_pairs.extend(pairs)
        except Exception as e:
            pass

    random.seed(42)
    random.shuffle(all_pairs)

    with open(output_file, 'w', encoding='utf-8') as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"Generated {len(all_pairs)} training pairs")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    db_dir = sys.argv[1] if len(sys.argv) > 1 else "data/dragonbones_raw"
    output = sys.argv[2] if len(sys.argv) > 2 else "data/dragonbones_animtoon.jsonl"
    convert_all(db_dir, output)
