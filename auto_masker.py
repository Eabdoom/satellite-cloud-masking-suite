"""
auto_masker.py - Automated Cloud Mask Generator
------------------------------------------------
Supports three masking algorithms:
  - hsv   (default): HSV color space filter — isolates bright, colorless (low saturation) pixels
  - otsu           : Otsu's adaptive thresholding — auto-calculates optimal cutoff per image
  - fixed          : Original fixed grayscale threshold of 95

Usage:
    python auto_masker.py                             # intern5, HSV mode, from progress.txt
    python auto_masker.py intern6                     # intern6, HSV mode
    python auto_masker.py intern6 --mode otsu         # Otsu mode
    python auto_masker.py intern6 --mode fixed        # Original fixed threshold
    python auto_masker.py intern6 --force             # Overwrite masks from progress.txt index
    python auto_masker.py intern6 --force-from 217    # Overwrite masks from image index 217
    python auto_masker.py intern6 --limit 10          # Only process 10 images (for testing)
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Automated Cloud Mask Generator")
parser.add_argument("dataset", nargs="?", default="intern5",
                    help="Name of dataset folder inside interns_dataset (e.g., intern6)")
parser.add_argument("-m", "--mode", choices=["hsv", "otsu", "fixed"], default="hsv",
                    help="Masking algorithm: hsv (default), otsu, or fixed")
parser.add_argument("-f", "--force", action="store_true",
                    help="Force overwrite existing masks starting from progress.txt index")
parser.add_argument("--force-from", type=int, default=None, dest="force_from",
                    help="Ignore progress.txt and force overwrite masks from this image index onwards")
parser.add_argument("-l", "--limit", type=int, default=None,
                    help="Limit the number of masks generated (for testing)")
args = parser.parse_args()

# --- Paths ---
DATASET_NAME = args.dataset
BASE_DIR     = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / DATASET_NAME
IMAGES_DIR   = DATASET_ROOT / "images"
MASKS_DIR    = DATASET_ROOT / "masks"
PROGRESS_FILE = DATASET_ROOT / "progress.txt"

# Post-processing kernel
MORPH_KERNEL_SIZE = 5

# Safety: refuse to run on completed folders
PROTECTED = {"intern3", "intern5"}

# --- Safety Check ---
if DATASET_NAME in PROTECTED:
    print(f"[BLOCKED] '{DATASET_NAME}' is a protected dataset. "
          f"This script will not touch intern3 or intern5.")
    sys.exit(1)

MASKS_DIR.mkdir(parents=True, exist_ok=True)

image_files = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

total = len(image_files)
if total == 0:
    print(f"[ERROR] No images found in {IMAGES_DIR}")
    sys.exit(1)

# --- Determine Start Index ---
if args.force_from is not None:
    start_index = max(0, min(args.force_from, total))
    force_overwrite = True
    print(f"[INFO] --force-from {args.force_from}: ignoring progress.txt, starting from index {start_index}")
else:
    force_overwrite = args.force
    start_index = 0
    if PROGRESS_FILE.exists():
        try:
            start_index = int(PROGRESS_FILE.read_text().strip())
            start_index = max(0, min(start_index, total))
        except Exception:
            pass
    print(f"[INFO] Progress index: {start_index} (read from progress.txt)")

print(f"Auto Masker")
print(f"Dataset  : {DATASET_NAME}")
print(f"Mode     : {args.mode.upper()}")
print(f"Images   : {total} total, processing from index {start_index}")
print(f"Overwrite: {'YES' if force_overwrite else 'NO (skip existing)'}")
if args.limit:
    print(f"Limit    : {args.limit}")
print("-" * 50)

generated = 0
skipped = 0


def generate_mask(img, mode):
    """Generate a binary cloud mask for the given BGR image using the specified mode."""
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)

    if mode == "hsv":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Low saturation (colorless) AND high value (bright) = cloud
        _, low_sat  = cv2.threshold(s, 60, 255, cv2.THRESH_BINARY_INV)
        _, high_val = cv2.threshold(v, 100, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(low_sat, high_val)

    elif mode == "otsu":
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    else:  # fixed
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


for i, img_name in enumerate(image_files[start_index:], start=start_index):
    if args.limit is not None and generated >= args.limit:
        print(f"Reached limit of {args.limit}. Stopping.")
        break

    img_path  = IMAGES_DIR / img_name
    stem      = Path(img_name).stem
    mask_path = MASKS_DIR / f"{stem}_mask.png"

    if mask_path.exists() and not force_overwrite:
        skipped += 1
        print(f"[{i+1}/{total}] SKIP: {img_name}")
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[{i+1}/{total}] ERROR reading: {img_name}")
        continue

    mask = generate_mask(img, args.mode)
    cv2.imwrite(str(mask_path), mask)
    generated += 1
    print(f"[{i+1}/{total}] DONE ({args.mode.upper()}): {img_name}")

print("-" * 50)
print(f"Complete! Generated: {generated} | Skipped: {skipped}")
print(f"Open 'python gui.py {DATASET_NAME}' to review the results.")
