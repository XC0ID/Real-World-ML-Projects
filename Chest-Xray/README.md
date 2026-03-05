# 🩺 Chest X-Ray Pneumonia Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Deep Learning](https://img.shields.io/badge/DeepLearning-CNN-orange)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

# 📌 Project Overview

Pneumonia is a serious lung infection that can be life-threatening if not detected early.  
Chest X-rays are commonly used by doctors to diagnose pneumonia.

This project uses **Deep Learning and Computer Vision** to automatically detect pneumonia from chest X-ray images.

The system trains multiple Convolutional Neural Network (CNN) models and compares their performance.

---

# 🎯 Objectives

✔ Detect pneumonia from chest X-ray images  
✔ Train multiple deep learning models  
✔ Compare model performance  
✔ Visualize training results  
✔ Build a real-world medical AI project

---

# 🧠 Models Used

| Model | Description |
|------|-------------|
| Custom CNN | Baseline convolutional neural network |
| ResNet | Deep residual learning architecture |
| MobileNet | Lightweight CNN optimized for speed |
| EfficientNet | Advanced architecture with high accuracy |

---

# 📂 Dataset

Dataset: **Chest X-Ray Pneumonia Dataset**

The dataset contains thousands of labeled chest X-ray images.

### Classes

- NORMAL
- PNEUMONIA

---

# 📁 Dataset Structure

chest_xray
│

├── train
│ ├── NORMAL
│ └── PNEUMONIA

│
├── test
│ ├── NORMAL
│ └── PNEUMONIA

│
└── val
├── NORMAL
└── PNEUMONIA


---

# 🖼 Sample Images

## Normal X-ray

![Normal](images/normal.jpeg)

## Pneumonia X-ray

![Pneumonia](images/pneumonia.jpeg)

---

# ⚙️ Project Pipeline

Dataset

↓

Data Preprocessing

↓

Data Augmentation

↓

Model Training

↓

Model Evaluation

↓

Model Comparison

↓

Visualization


---

# 🔬 Data Preprocessing

The following preprocessing steps were applied:

- Image resizing
- Normalization
- Data augmentation
- Train / validation split

These steps help improve model generalization.

---

# 📊 Training Results

| Model | Validation Accuracy | Test Accuracy | 
|------|----------|----------|
|  MobileNetV3 | 0.93 | 0.86 |
| ResNet18 | 1 | o.93 |
| DenseNet121  | 0.87 | 0.91 |
| EfficientnetB0 | 1 | 0.93 |

---

# 🧪 Evaluation

Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- Confusion Matrix

---

# 💻 Technologies Used

| Tool | Purpose |
|-----|--------|
| Python | Programming |
| PyTorch / TensorFlow | Deep Learning |
| NumPy | Numerical computing |
| Pandas | Data processing |
| Matplotlib | Visualization |
| Scikit-learn | Evaluation |

---

# 📦 Installation

Clone the repository
git clone https://github.com/XC0ID/Real-World-ML-Projects.git

---

# 🚀 Future Improvements

- Deploy as web application
- Hyperparameter tuning
- Larger dataset training
- Model explainability (GradCAM)
- Integration with hospital systems

---


