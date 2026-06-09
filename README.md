# Satellite Cloud Masking Annotation & Automation Suite

A professional-grade computer vision suite designed to accelerate dataset creation for satellite cloud segmentation. This toolkit combines automated mask generation algorithms with a custom interactive desktop GUI, enabling rapid zero-shot masking and lightning-fast manual annotation refinement.

---

<img width="1904" height="935" alt="image" src="https://github.com/user-attachments/assets/439dcc64-a835-4d78-adb9-859b8b9707b5" />


## Key Features

* **Multi-Algorithm Automation**: Choose between several image processing pipelines:
  * **HSV Color Space Filter**: Isolates bright, colorless pixels (clouds) by saturation and value thresholds, ignoring green forests, soil, and blue water.
  * **Otsu's Adaptive Thresholding**: Mathematically optimizes binary mask segmentation per image.
  * **K-Means Clustering**: Groups image pixels into distinct clusters and labels the brightest segment as cloud.
  * **Fixed Grayscale**: Fast baseline thresholding.
* **Interactive PyQt5 GUI**: Designed for high-speed manual mask touch-ups and inspection.
  * Dynamic brush drawing (left-click) and erasing (right-click) with adjustable brush sizes.
  * Live-swapping of masking algorithms (Re-Mask) and on-the-fly threshold adjustments.
  * Bulk remasking capabilities for all remaining images directly from the UI.
  * Undo support (`Ctrl + Z`) and opacity control slider for the mask overlay.
* **Night Image Detection**: Auto-detects night/dark images below a threshold and skips them or generates blank masks.
* **Progress-Aware Workflows**: CLI and GUI read from a local `progress.txt` to seamlessly resume labeling sessions.

---

## Setup

### Step 1: Clone and Set Up Environment
We provide a one-click setup script for Windows users, or you can set it up manually.

#### Option A: One-Click Setup (Windows)
Double-click `setup.bat`. This script will:
1. Create a Python virtual environment (`venv`).
2. Upgrade `pip` and install all required libraries.
3. Configure your VS Code workspace settings to use this environment automatically.

#### Option B: Manual Setup
Open your terminal in the repository folder and run:
```bash
# 1. Create the environment
python -m venv venv

# 2. Activate the environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run Tools

### 1. Automated Baselines (`auto_masker.py` / `kmeans_masker.py`)
Run the automated mask generator first to build initial masks for your dataset.

#### **Standard Auto Masker (`auto_masker.py`)**
Supports HSV (default), Otsu, and Fixed Grayscale threshold modes. Respects `progress.txt` and skips already processed images by default.
```bash
# Generate masks for 'dataset_folder' using HSV mode
python auto_masker.py dataset_folder

# Generate masks using Otsu adaptive thresholding
python auto_masker.py dataset_folder --mode otsu

# Overwrite existing masks from your progress point onwards
python auto_masker.py dataset_folder --force

# Force overwrite masks starting from image index 217 (ignores progress.txt)
python auto_masker.py dataset_folder --force-from 217

# Run a quick test on the first 5 images only
python auto_masker.py dataset_folder --limit 5
```

#### **K-Means Masker (`kmeans_masker.py`)**
A separate tool leveraging K-Means (K=3) color clustering. Includes automatic night detection (outputs blank masks if mean brightness is low).
```bash
# Run K-Means masking on 'dataset_folder'
python kmeans_masker.py dataset_folder

# Run with custom night brightness threshold
python kmeans_masker.py dataset_folder --night-threshold 15.0

# Force overwrite all existing masks
python kmeans_masker.py dataset_folder --force
```

### 2. Manual Annotation GUI (`gui.py`)
Launch the desktop application to review baseline masks, manually correct errors, or bulk-remask remaining images.
```bash
# Start annotating the default dataset
python gui.py

# Open a specific dataset folder
python gui.py dataset_folder
```

---

## GUI Controls Legend

| Key/Mouse Action | Control Description |
|---|---|
| **Left-Click & Drag** | Draw/add cloud mask (Red overlay) |
| **Right-Click & Drag** | Erase/remove mask |
| **`[` and `]`** | Decrease / Increase brush size |
| **Up / Down Arrow Keys** | Adjust Grayscale Threshold value dynamically (updates slider) |
| **Left / Right Arrow Keys**| Navigate to Previous / Next image (automatically saves current mask) |
| **`Ctrl + Z`** | Undo last manual brush stroke |
| **`R`** | Clear mask completely |
| **`D`** | Delete image and mask pair (requires popup confirmation) |
| **`M`** (or **Re-Mask** button) | Live-swaps/regenerates the current image mask using the selected radio algorithm |
| **Remask All Remaining** | Overwrites all remaining images from current progress onwards with the selected algorithm |

---

## Expected Repository Structure
To use these tools, format your dataset directory in the root directory as follows:
```text
your_dataset_name/
├── images/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── masks/             # Created automatically if not present
│   ├── image_001_mask.png
│   └── ...
└── progress.txt       # Created automatically to track progress
```
