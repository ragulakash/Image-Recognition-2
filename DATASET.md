# 🖼️ Dataset Information Report

This document provides a technical deep-dive into the dataset used for the Image Classification Comparison project.

## 📊 Summary Statistics
| Property | Value |
| :--- | :--- |
| **Data Type** | Synthetic Colored Noise |
| **Total Samples** | 200 |
| **Classes** | 3 (Red, Green, Blue Tints) |
| **Resolution** | 64x64 pixels |
| **Color Space** | RGB (3 Channels) |
| **Train/Test Split** | 160 / 40 (80% / 20%) |

---

## 🔬 Generation Methodology
The dataset is generated algorithmically when the `data/` directory is empty. This ensures the benchmarking pipeline can be verified even without an external dataset.

### **Algorithm Logic**
For each image in the 200-sample set:
1.  **Base Layer**: A 64x64x3 matrix is filled with random integers (0-255).
2.  **Labeling**: Images are assigned a class using `index modulo 3`.
3.  **Feature Injection**:
    - **Class 0 (Red)**: The Red channel intensity is increased by a constant offset (+50).
    - **Class 1 (Green)**: The Green channel intensity is increased by a constant offset (+50).
    - **Class 2 (Blue)**: The Blue channel intensity is increased by a constant offset (+50).

### **Why is accuracy low?**
Because the background is random noise, the models are trying to detect a subtle +50 shift in a specific channel amidst significant 0-255 variance. This represents a "High Noise" environment, which is intentionally difficult for small datasets.

---

## 🔄 Preprocessing Pipeline
Depending on the model architecture, the data undergoes different transformations:

### **1. Deep Learning (CNN, ResNet)**
- **Shape**: (N, 64, 64, 3)
- **Scaling**: Pixel values are divided by 255.0 to scale them between **[0, 1]**.
- **Input**: 4D Tensors.

### **2. Machine Learning (SVM, KNN, RF)**
- **Flattening**: Each 64x64x3 image is flattened into a **12,288-dimensional vector** (64*64*3).
- **Shape**: (N, 12288)
- **Normalization**: Min-Max scaling is applied to ensure feature parity.

---

## 🚀 Transitioning to Real Data
To replace this synthetic set with real images (e.g., Cats vs Dogs):

1.  Create subdirectories in `data/`:
    ```text
    data/
    ├── cats/
    └── dogs/
    ```
2.  Add your `.jpg` or `.png` images into these folders.
3.  Re-run the Jupyter notebooks. The logic will automatically detect the folders and load the real images instead of generating noise.
