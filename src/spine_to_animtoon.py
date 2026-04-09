"""
Convert Spine 2D animation JSON to AnimTOON training data.

Spine format: bones with hierarchical rotations, translations, scales
AnimTOON format: layers with keyframed properties

Each bone becomes an AnimTOON layer.
Each timeline becomes keyframes (rot, pos, scale).
"""
import json
import os
import sys
import random

ARROW = "\u2192"


def parse_spine_animation(spine_data, anim_name):
    """Convert one Spine animation to AnimTOON format."""

    bones = {b['name']: b for b in spine_data.get('bones', [])}
    animations = spine_data.get('animations', {})

    if anim_name not in animations:
        return None

    anim = animations[anim_name]
    bone_timelines = anim.get('bones', {})

    # Find animation duration
    max_time = 0
    for bone_name, timelines in bone_timelines.items():
        for prop, keyframes in timelines.items():
            for kf in keyframes:
                if kf.get('time', 0) > max_time:
                    max_time = kf['time']

    if max_time == 0:
        max_time = 1.0

    dur = max(30, int(max_time * 30))  # 30fps

    # Build AnimTOON layers
    lines = [f"anim fr=30 dur={dur}", ""]

    animated_bones = 0

    for bone_name, timelines in bone_timelines.items():
        bone_info = bones.get(bone_name, {})
        parent = bone_info.get('parent', '')

        # Determine bone type for description
        bone_lower = bone_name.lower()
        if any(w in bone_lower for w in ['arm', 'hand', 'finger']):
            bone_type = 'limb'
        elif any(w in bone_lower for w in ['leg', 'foot', 'toe']):
            bone_type = 'limb'
        elif any(w in bone_lower for w in ['head', 'skull']):
            bone_type = 'head'
        elif any(w in bone_lower for w in ['eye', 'blink']):
            bone_type = 'eye'
        elif any(w in bone_lower for w in ['mouth', 'jaw', 'lip']):
            bone_type = 'mouth'
        elif any(w in bone_lower for w in ['hair', 'ponytail', 'bangs']):
            bone_type = 'hair'
        elif any(w in bone_lower for w in ['body', 'torso', 'hip', 'root']):
            bone_type = 'body'
        else:
            bone_type = 'part'

        lines.append(f"layer {bone_name} shape")

        has_anim = False

        # Rotation timeline
        if 'rotate' in timelines:
            kfs = timelines['rotate']
            if len(kfs) >= 2:
                rot_parts = []
                for kf in kfs:
                    t = round(kf.get('time', 0) / max_time, 2)
                    t = min(1.0, max(0.0, t))
                    val = round(kf.get('value', kf.get('angle', 0)), 1)
                    rot_parts.append(f"{t}{ARROW}{val}")

                # Determine easing
                curve = kfs[0].get('curve', 'smooth')
                if curve == 'linear':
                    ease = 'linear'
                elif curve == 'stepped':
                    ease = 'linear'
                else:
                    ease = 'smooth'

                lines.append(f"  rot {' '.join(rot_parts)} ease={ease}")
                has_anim = True

        # Translation timeline
        if 'translate' in timelines:
            kfs = timelines['translate']
            if len(kfs) >= 2:
                pos_parts = []
                for kf in kfs:
                    t = round(kf.get('time', 0) / max_time, 2)
                    t = min(1.0, max(0.0, t))
                    # Normalize position to 0-1 range (Spine uses pixels)
                    x = round(0.5 + kf.get('x', 0) / 512, 3)
                    y = round(0.5 + kf.get('y', 0) / 512, 3)
                    pos_parts.append(f"{t}{ARROW}[{x},{y}]")

                lines.append(f"  pos {' '.join(pos_parts)} ease=smooth")
                has_anim = True

        # Scale timeline
        if 'scale' in timelines:
            kfs = timelines['scale']
            if len(kfs) >= 2:
                scale_parts = []
                for kf in kfs:
                    t = round(kf.get('time', 0) / max_time, 2)
                    t = min(1.0, max(0.0, t))
                    sx = round(kf.get('x', 1.0) * 100, 1)
                    sy = round(kf.get('y', 1.0) * 100, 1)
                    scale_parts.append(f"{t}{ARROW}[{sx},{sy}]")

                lines.append(f"  scale {' '.join(scale_parts)} ease=smooth")
                has_anim = True

        if has_anim:
            animated_bones += 1

        lines.append("")

    if animated_bones == 0:
        return None

    return '\n'.join(lines).strip()


def build_description(character_name, anim_name, bone_names):
    """Generate a text description for the animation."""

    # Map animation names to descriptions
    anim_descriptions = {
        'idle': 'standing still with subtle breathing movement',
        'walk': 'walking forward with coordinated arm and leg movement',
        'run': 'running with fast arm and leg cycling',
        'jump': 'jumping up and landing back down',
        'attack': 'performing an attack motion with arm swing',
        'death': 'falling down in a death animation',
        'hit': 'reacting to being hit with a flinch',
        'blink': 'blinking eyes briefly',
        'aim': 'aiming with arm extended forward',
        'crouch': 'crouching down low',
        'fall': 'falling through the air',
        'roar': 'roaring with mouth open and body tensed',
        'grow': 'growing larger from small to full size',
        'bounce': 'bouncing up and down with elastic motion',
    }

    base_desc = anim_descriptions.get(anim_name, f'performing a {anim_name} animation')

    # Count body part types
    has_arms = any('arm' in b.lower() or 'hand' in b.lower() for b in bone_names)
    has_legs = any('leg' in b.lower() or 'foot' in b.lower() for b in bone_names)
    has_head = any('head' in b.lower() or 'skull' in b.lower() for b in bone_names)
    has_eyes = any('eye' in b.lower() for b in bone_names)

    parts = []
    if has_head: parts.append('head')
    if has_arms: parts.append('arms')
    if has_legs: parts.append('legs')
    if has_eyes: parts.append('eyes')

    body_desc = f"with {', '.join(parts)}" if parts else f"with {len(bone_names)} body parts"

    templates = [
        f"The video shows a {character_name} character {body_desc} {base_desc}. The animation is smooth with coordinated movement between all body parts.",
        f"A {character_name} character {base_desc}. The character has {body_desc} that move in a coordinated way throughout the animation.",
        f"The animation features a {character_name} {base_desc}. Each body part ({', '.join(parts) if parts else 'all parts'}) animates smoothly with proper timing.",
    ]

    return random.choice(templates)


def spine_to_training_data(spine_dir, output_file):
    """Convert all Spine examples to AnimTOON training pairs."""

    pairs = []

    for filename in os.listdir(spine_dir):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(spine_dir, filename)
        character_name = filename.replace('.json', '').replace('-pro', '').replace('-ess', '')

        with open(filepath, 'r') as f:
            spine_data = json.load(f)

        bone_names = [b['name'] for b in spine_data.get('bones', [])]
        animations = spine_data.get('animations', {})

        for anim_name in animations:
            animtoon = parse_spine_animation(spine_data, anim_name)
            if not animtoon:
                continue

            description = build_description(character_name, anim_name, bone_names)

            # Also create layer-aware version
            layer_desc_parts = []
            for i, bone in enumerate(bone_names[:10]):  # Limit to 10 bones
                bone_lower = bone.lower()
                if 'arm' in bone_lower:
                    btype = 'arm segment'
                elif 'leg' in bone_lower:
                    btype = 'leg segment'
                elif 'head' in bone_lower:
                    btype = 'head'
                elif 'eye' in bone_lower:
                    btype = 'eye'
                elif 'body' in bone_lower or 'torso' in bone_lower:
                    btype = 'body'
                else:
                    btype = 'body part'
                layer_desc_parts.append(f"{i+1}. {bone} ({btype})")

            layer_desc = f"SVG has {len(bone_names)} layers:\n  " + "\n  ".join(layer_desc_parts)
            layer_desc += f"\nAnimate with {anim_name} motion, coordinated movement."

            # Add both description types
            pairs.append({
                'instruction': description,
                'output': animtoon
            })
            pairs.append({
                'input': layer_desc,
                'output': animtoon
            })

    # Shuffle
    random.seed(42)
    random.shuffle(pairs)

    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"Generated {len(pairs)} training pairs from Spine animations")
    print(f"Saved to: {output_file}")

    # Show examples
    for i in range(min(3, len(pairs))):
        p = pairs[i]
        inp = p.get('instruction', p.get('input', ''))
        print(f"\n--- Example {i+1} ---")
        print(f"INPUT: {inp[:200]}")
        print(f"OUTPUT: {p['output'][:200]}...")


if __name__ == "__main__":
    spine_dir = sys.argv[1] if len(sys.argv) > 1 else "data/spine_examples"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/spine_animtoon.jsonl"

    spine_to_training_data(spine_dir, output_file)
