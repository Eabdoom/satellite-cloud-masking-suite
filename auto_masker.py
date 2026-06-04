import os
import cv2
import numpy as np
from pathlib import Path

# Your exact directories
DATASET_ROOT = r"C:\Users\arnav\Personal\Internships\XDLINX Space Labs\interns_dataset\intern3"
IMAGES_DIR = Path(DATASET_ROOT) / "images"
MASKS_DIR = Path(DATASET_ROOT) / "masks"

# Make sure the masks directory exists
MASKS_DIR.mkdir(parents=True, exist_ok=True)

# Find all images
image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
print(f"Found {len(image_files)} images. Starting automation...")

for img_name in image_files:
    img_path = str(IMAGES_DIR / img_name)
    stem = Path(img_name).stem
    mask_path = str(MASKS_DIR / f"{stem}_mask.png")
    
    # 1. Read the image
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    # 2. Convert to grayscale to isolate brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. The Magic Threshold
    # Lower this value (e.g., 120-150) to make it more sensitive to dim cloud edges.
    _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # 4. Optional Polish: Remove tiny specks of noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 5. Save exactly where the GUI expects it
    cv2.imwrite(mask_path, mask)

print("Automation complete. 600 masks generated. Open gui.py to verify.")
