"""
kmeans_masker.py - K-Means Clustering Cloud Mask Generator
-----------------------------------------------------------
Uses K-Means (K=3) to cluster each image into 3 pixel groups,
automatically identifying the brightest cluster as thick cloud.

This is a SEPARATE script from auto_masker.py and only targets
the specified dataset folder. It will NOT touch intern3 or intern5.

Usage:
    python kmeans_masker.py                 # Defaults to intern1
    python kmeans_masker.py intern2         # Specify a different folder
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="K-Means Clustering Cloud Mask Generator")
parser.add_argument("dataset", nargs="?", default="intern1", help="Name of dataset folder (e.g., intern1)")
parser.add_argument("-l", "--limit", type=int, default=None, help="Limit the number of masks generated for testing")
parser.add_argument("-f", "--force", action="store_true", help="Force overwrite existing masks (useful if existing masks are wrong)")
parser.add_argument("-n", "--night-threshold", type=float, default=20.0, help="Brightness threshold below which an image is treated as night (generates a blank mask)")
args = parser.parse_args()

DATASET_NAME = args.dataset
DATASET_ROOT = Path(__file__).resolve().parent / DATASET_NAME
IMAGES_DIR   = DATASET_ROOT / "images"
MASKS_DIR    = DATASET_ROOT / "masks"

# K-Means settings
K          = 3    # Number of clusters (dark ground / haze / bright cloud)
MAX_ITER   = 10   # Max iterations per image (keeps it fast)
ATTEMPTS   = 3    # How many times K-Means restarts with different seeds

# Post-processing: morphological cleanup
MORPH_KERNEL_SIZE = 5   # Size of kernel for noise removal (increase for smoother masks)

# Safety: refuse to run on these folders
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

print(f"K-Means Cloud Masker")
print(f"Dataset  : {DATASET_NAME}")
print(f"Images   : {total}")
print(f"Output   : {MASKS_DIR}")
print(f"Settings : K={K}, MaxIter={MAX_ITER}, Attempts={ATTEMPTS}")
if args.limit is not None:
    print(f"Limit    : Up to {args.limit} new masks will be generated")
if args.force:
    print(f"Mode     : FORCE OVERWRITE (existing masks will be overwritten)")
print("-" * 50)

skipped = 0
generated = 0

for i, img_name in enumerate(image_files):
    # If a limit is specified and we've generated that many, stop.
    if args.limit is not None and generated >= args.limit:
        print(f"Reached limit of {args.limit} generated masks. Stopping.")
        break

    img_path  = IMAGES_DIR / img_name
    stem      = Path(img_name).stem
    mask_path = MASKS_DIR / f"{stem}_mask.png"

    # Skip if mask already exists (unless force overwrite is enabled)
    if mask_path.exists() and not args.force:
        skipped += 1
        print(f"[{i+1}/{total}] SKIP (mask exists): {img_name}")
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[{i+1}/{total}] ERROR reading: {img_name}")
        continue

    # --- Night Image Detection ---
    # Convert to grayscale to check mean brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()

    if mean_brightness < args.night_threshold:
        # Save a completely black mask for night images
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.imwrite(str(mask_path), mask)
        generated += 1
        print(f"[{i+1}/{total}] DONE (NIGHT - saved blank mask): {img_name} | Mean brightness: {mean_brightness:.1f}")
        continue

    # --- K-Means Segmentation ---
    pixel_data = img.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, MAX_ITER, 1.0)
    _, labels, centers = cv2.kmeans(
        pixel_data, K, None, criteria, ATTEMPTS, cv2.KMEANS_PP_CENTERS
    )

    # --- Identify the "Cloud" cluster ---
    center_brightness = centers.mean(axis=1)
    cloud_cluster_idx = int(np.argmax(center_brightness))

    # Build binary mask
    labels_flat = labels.flatten()
    mask = np.where(labels_flat == cloud_cluster_idx, 255, 0).astype(np.uint8)
    mask = mask.reshape(img.shape[:2])

    # --- Post-processing: clean up noise ---
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove tiny specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes

    cv2.imwrite(str(mask_path), mask)
    generated += 1
    print(f"[{i+1}/{total}] DONE: {img_name} | Cloud cluster brightness: {center_brightness[cloud_cluster_idx]:.1f}")

print("-" * 50)
print(f"Finished! Generated: {generated} | Skipped (already existed): {skipped}")
print(f"Open gui.py and switch to '{DATASET_NAME}' to review the results.")
