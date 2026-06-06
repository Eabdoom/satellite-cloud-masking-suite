# XDLINX Space Labs - Cloud Masking Tools Setup Guide

Welcome to the Cloud Masking dataset project. This guide will help you set up your environment and get started with the tools to generate and annotate thick clouds in satellite imagery.

## ⚠️ Important Note Before Starting
Do **NOT** share or copy the virtual environment folder (`venv/`) or progress files (`progress.txt`) from another machine. Doing so will break package paths and overwrite your work. Follow the setup below to create a clean environment.

---

## 🛠️ Step-by-Step Setup

### Step 1: Create a Fresh Virtual Environment
Open your terminal (PowerShell, Command Prompt, or terminal of choice) in this project folder (`interns_dataset/`) and run:

```bash
# 1. Create the environment
python -m venv venv

# 2. Activate the environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.\venv\Scripts\activate.bat

# On macOS/Linux:
source venv/bin/activate
```

---

### Step 2: Install Required Packages
With the environment activated, run the following command to install the required dependencies:

```bash
pip install opencv-python Pillow PyQt5 numpy
```

---

### Step 3: Run the Tools

#### 1. Automatic Masking (`auto_masker.py`)
Run this first to automatically pre-generate masks for any images that do not have them yet (it uses a binary threshold logic to detect bright clouds):
```bash
python auto_masker.py
```

#### 2. K-Means Clustering Masker (`kmeans_masker.py`)
An advanced alternative to `auto_masker.py`. Instead of a fixed brightness threshold, it uses **K-Means Clustering (K=3)** to automatically group pixels into dark ground, haze, and bright cloud regions — then extracts the brightest cluster as the cloud mask. This works better on datasets with varying lighting conditions.

```bash
# Run on intern1 (default)
python kmeans_masker.py

# Run on a different folder (e.g., intern2)
python kmeans_masker.py intern2
```

> ⚠️ This script is **hardcoded to refuse** running on `intern3` or `intern5` to protect completed work.
> It also **skips images that already have a mask**, so it will never overwrite your manual touch-ups.
> It takes roughly **1-2 seconds per image**, so expect ~10-20 minutes for 600 images.

#### 3. Manual Annotation GUI (`gui.py`)
Run this tool to inspect, manually touch up the masks (draw/erase), adjust thresholds, and save your progress:
```bash
python gui.py
```

---

## 🎮 GUI Controls Legend

* **Left-Click & Drag**: Draw/add cloud mask (Red overlay)
* **Right-Click & Drag**: Erase/remove mask
* **`[` and `]`**: Shrink and grow brush size
* **Up / Down Arrow Keys**: Increase / decrease threshold on the fly (updates the slider)
* **Left / Right Arrow Keys**: Go to Previous / Next image (automatically saves your current mask!)
* **`Ctrl + Z`**: Undo last stroke
* **`R` Key**: Clear mask completely
* **`D` Key**: Delete image & mask pair (confirmation popup will appear)
