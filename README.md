# Satellite Cloud Masking: PyQt5 Annotator & OpenCV Auto-Masker

A high-productivity cloud masking solution developed during my internship at **XDLINX Space Labs** to streamline the annotation of thick clouds in remote sensing/satellite imagery. This project converts a tedious manual drawing pipeline into an efficient automated-generation and verification workflow for training machine learning models.

---

## 🚀 The Core Problem
To train AI models on satellite imagery, cloud cover must be accurately masked. Manually segmenting clouds pixel-by-pixel or with standard drawing utilities is incredibly time-consuming. 

This repository introduces a **hybrid pipeline** that:
1. **Automatically pre-masks** candidate cloud regions using Computer Vision.
2. **Allows rapid human-in-the-loop review and correction** via an interactive PyQt5 GUI desktop application.

---

## 🛠️ Architecture & Features

### 1. Automated Segmentation Pipeline (`auto_masker.py`)
Instead of starting annotations from scratch, a script uses computer vision to generate candidate masks for the entire dataset:
- **Grayscale Conversion & Thresholding**: Leverages brightness characteristics of thick clouds to segment candidate areas.
- **Morphological Operations**: Applies opening (`MORPH_OPEN`) filters to eliminate salt-and-pepper noise and clean up boundaries.
- **Incremental Workflow**: Automatically respects progress logs to avoid overwriting existing manual annotations.

### 2. Interactive Annotation GUI (`gui.py`)
A custom PyQt5 desktop application built to review and correct masks:
- **Dual Visuals**: Displays the original satellite image, the generated binary mask, and a red-masked image overlay side-by-side for instant comparison.
- **Brush & Erase**: Allows direct drawing with **Left-Click** and erasing with **Right-Click** with adjustable brush sizes.
- **State Undo**: Built-in undo operations (`Ctrl + Z`) to revert mistakes.
- **Keyboard Shortcuts**: Arrow-key navigation (Left/Right) for seamless image swapping, auto-saving progress, and quick deletion commands (`D` or `R`).
- **State Preservation**: Saves progress using a simple state file (`progress.txt`) allowing users to close the application and resume at any point.

---

## 📦 Setup & Installation

### Prerequisites
Make sure you have Python installed. You can install all dependencies via pip:
```bash
pip install numpy opencv-python PyQt5 Pillow
```

### Dataset Structure
Organize your dataset folder (e.g., `intern3`) as follows:
```text
intern3/
├── images/        # Put your raw satellite images here (.png, .jpg, .jpeg)
├── masks/         # Output directory where masks will be saved
└── progress.txt   # Tracks your last annotated image index
```

---

## 📖 Usage Guide

### Step 1: Pre-generate Masks
Run the automated masking script to do the heavy lifting:
```bash
python auto_masker.py
```
This generates initial binary masks in the `masks/` folder.

### Step 2: Review and Correct
Launch the GUI tool to verify the results:
```bash
python gui.py
```
Use the mouse to paint or erase regions where the thresholding was too sensitive or missed detail, and tap the **Right Arrow** key to automatically save and proceed.
