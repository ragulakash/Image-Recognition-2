# 📊 Image Classification Model Comparison Study

A comprehensive benchmark and analysis project evaluating various Machine Learning (ML) and Deep Learning (DL) architectures for image classification tasks.

## 🚀 Overview
This project provides a standardized framework to train, test, and compare different image classification models. It automatically records performance metrics such as accuracy, training time, and inference latency, presenting them in an interactive dashboard for professional analysis.

---

## 🛠️ Project Structure
```text
├── CNN_Model.ipynb          # Custom CNN implementation
├── KNN_Model.ipynb          # K-Nearest Neighbors implementation
├── ResNet_Model.ipynb       # ResNet50 Transfer Learning implementation
├── SVM_Model.ipynb          # Support Vector Machine implementation
├── dashboard.py             # Interactive Streamlit Dashboard
├── data/                    # Dataset directory (organized by class folders)
├── results/
│   └── model_comparison.csv # Centralized performance results
└── requirements.txt         # Project dependencies
```

---

## 📊 Performance Report

The following results were obtained using the provided benchmarking pipeline. 

| Model | Type | Accuracy | Training Time (s) | Inference (ms/sample) | Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CNN** | Deep Learning | 37.50% | 4.93 | 9.55 | 822,467 |
| **CustomCNN** | Deep Learning | 25.00% | 1.04 | 6.49 | 822,467 |
| **KNN** | Machine Learning | 22.50% | 0.01 | 51.61 | 0 |
| **SVM** | Machine Learning | 22.50% | 3.44 | 14.29 | 0 |
| **RandomForest** | Machine Learning | 20.00% | 0.23 | 0.10 | 0 |
| **ResNet50** | Deep Learning | 20.00% | 17.81 | 86.58 | 23,593,859 |
| **MobileNetV2** | Deep Learning | 20.00% | 2.34 | 39.46 | 2,261,827 |

> [!NOTE]
> **Dataset Insight**: The current results are based on **Synthetic Colored Noise** data. For a technical deep-dive into how this data is generated and processed, see the [**Detailed Dataset Report (DATASET.md)**](DATASET.md).

---

## 🖥️ Interactive Dashboard
A high-performance Streamlit dashboard is included to visualize these results dynamically.

### **Features:**
- **KPI Cards**: Instant view of the best-performing models.
- **Accurancy & Latency Charts**: Interactive bar and scatter plots.
- **Dataset Navigator**: Detailed information about the training data characteristics.
- **Filtered Comparison**: Toggle between ML and DL models in real-time.

### **How to Run:**
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

---

## 📋 Requirements
- Python 3.8+
- TensorFlow/Keras
- scikit-learn
- OpenCV
- Pandas / NumPy
- Streamlit / Plotly

---

## 📈 Future Improvements
1. **Real Data Integration**: Benchmarking on standard datasets like CIFAR-10 or custom specialized data.
2. **Hyperparameter Tuning**: Implementing automated grid search for ML models.
3. **Model Export**: Adding functionality to save the best-performing models for deployment.

---
*Developed as a Comparative Study in Image Classification.*
