"""
Download MMSVG datasets from HuggingFace.

Usage:
    python src/download_mmsvg.py --subset Illustration
    python src/download_mmsvg.py --subset Icon
"""
import argparse
import os

os.environ["HF_HOME"] = "D:/hf-cache"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=["Illustration", "Icon"],
                        help="MMSVG subset to download")
    args = parser.parse_args()

    from datasets import load_dataset

    subset_name = f"MMSVG-{args.subset}"
    save_path = f"data/mmsvg-{args.subset.lower()}"

    print(f"Downloading OmniSVG/MMSVG-2M ({subset_name})...")
    print(f"Cache: D:/hf-cache")
    print(f"Save to: {save_path}")
    print()

    ds = load_dataset("OmniSVG/MMSVG-2M", subset_name, split="train",
                      cache_dir="D:/hf-cache")

    print(f"\nDownloaded: {len(ds)} samples")
    print(f"Columns: {ds.column_names}")

    os.makedirs(save_path, exist_ok=True)
    ds.save_to_disk(save_path)
    print(f"Saved to: {save_path}")
