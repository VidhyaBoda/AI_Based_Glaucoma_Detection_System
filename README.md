# 👁️ AI-Based Glaucoma Detection System

<p align="center">
  <img
    src="Images/AI- Based Glaucoma Detection System.png"
    alt="AI-Based Glaucoma Detection System project cover"
    width="100%"
  />
</p>

<p align="center">
  <strong>Deep Learning • Computer Vision • Transfer Learning</strong><br/>
  Retinal fundus image classification using EfficientNetB0
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow%2FKeras-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow/Keras"/></a>
  <img src="https://img.shields.io/badge/Model-EfficientNetB0-0F766E" alt="EfficientNetB0"/>
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-2563EB" alt="Binary classification"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"/></a>
</p>

> **Research/educational prototype only. This repository is not a clinically validated diagnostic system and must not be used to make medical decisions.**

---

## 📌 Project Overview

This project explores binary classification of retinal fundus images into **Glaucoma** and **Normal** classes using deep learning. It uses **EfficientNetB0** with ImageNet-pretrained weights, a custom classification head, transfer learning, and partial fine-tuning.

The notebook documents a research workflow from dataset preparation and image preprocessing through model training, evaluation, model saving, and single-image prediction.

## 🎯 Objectives

- Organize retinal fundus images into two classes: Glaucoma and Normal.
- Prepare reproducible training, validation, and test splits.
- Apply image resizing and training-time augmentation.
- Train an EfficientNetB0-based binary classifier.
- Review predictions with a confusion matrix and classification report.

## ✨ Key Features

- Retinal fundus image preparation and preprocessing
- Binary classification: Glaucoma vs Normal
- Approximate 70% / 15% / 15% train-validation-test split
- Image augmentation for training
- EfficientNetB0 transfer learning and partial fine-tuning
- Custom classification head
- Confusion matrix and scikit-learn classification report
- Saved model and single-image prediction workflow

## 🧭 Workflow

```text
Retinal Fundus Images
        ↓
Dataset Download / Organization
        ↓
Train / Validation / Test Split
        ↓
Image Preprocessing (224 × 224)
        ↓
Training Data Augmentation
        ↓
EfficientNetB0 (ImageNet Weights)
        ↓
Custom Classification Head
        ↓
Training and Partial Fine-Tuning
        ↓
Test Predictions
        ↓
Confusion Matrix + Classification Report
        ↓
Saved Model / Single-Image Prediction
```

## 🧰 Technology Stack

| Area | Technology | Use |
|---|---|---|
| Programming | Python | Notebook implementation |
| Deep learning | TensorFlow / Keras | Model construction and training |
| Backbone | EfficientNetB0 | Image feature learning via transfer learning |
| Image processing | Pillow | Image loading and resizing |
| Data processing | NumPy, Pandas | Array and data handling |
| Evaluation | Scikit-learn | Confusion matrix and classification report |
| Visualization | Matplotlib, Seaborn | Training/evaluation visualizations |
| Environment | Jupyter Notebook / Google Colab | Development and experimentation |
| Version control | Git / GitHub | Source management and documentation |

## 📂 Dataset

The notebook downloads a glaucoma-detection dataset through the Kaggle API using the identifier:

```text
dasa7753912/glaucoma-detection
```

The notebook organizes images into two labels:

- **Glaucoma**
- **Normal**

The README and notebook should identify the dataset precisely and consistently. Verify that this Kaggle source is the intended ACRIMA dataset before describing it as ACRIMA in a report or resume.

### Train / Validation / Test split

The notebook describes a two-stage split yielding approximately:

| Split | Target share |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

The split uses `random_state = 42` for reproducibility. Actual image counts depend on the dataset files successfully downloaded and included in the split; report the counts from the executed notebook rather than estimating them.

## 🧠 Model Approach

### EfficientNetB0 transfer learning

- Uses ImageNet-pretrained EfficientNetB0.
- Removes the original top classification layer with `include_top=False`.
- Uses an input size of `224 × 224 × 3`.
- Adds a custom binary-classification head.
- Freezes earlier backbone layers while fine-tuning the last 30 layers, as configured in the notebook.

### Training configuration documented in the notebook

| Parameter | Configuration |
|---|---|
| Optimizer | Adam |
| Learning rate | `5e-5` |
| Loss | Binary cross-entropy |
| Training epochs | 10 |
| Batch size | 32 |
| Image size | 224 × 224 |
| Decision threshold | 0.5 |

These are the documented notebook settings. Actual performance should be reported only after running the notebook and reviewing the outputs.

## 📊 Evaluation

The notebook evaluates test predictions using:

- Confusion matrix
- Classification report, including precision, recall, F1-score, and support

**Current evaluation limitation:** ROC-AUC, sensitivity, specificity, calibration, and ROC curves are identified as future improvements in this project documentation. Do not interpret a model prediction as a clinical diagnosis.

## 🖼️ Workflow overview

<p align="center">
  <img
    src="Images/AI-Based Glaucoma Detection System Workflow.png"
    alt="AI-Based Glaucoma Detection System project cover"
    width="100%"
  />
</p>

> These images illustrate the project and its workflow; they are not a substitute for actual evaluation outputs.

## 📁 Repository Structure

```text
AI_Based_Glaucoma_Detection_System/
├── Images/
│   ├── AI Glaucoma Detection System.png
│   └── AI-Based Glaucoma Detection System Workflow.png
├── AI_Based_Glaucoma_Detection_System.ipynb
├── README.md
├── LICENSE
└── SECURITY.md
```

This reflects the current repository layout. The main implementation is in the Jupyter Notebook; `src/`, `models/`, and `results/` folders are not represented as existing files unless they are added later.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/VidhyaBoda/AI_Based_Glaucoma_Detection_System.git
cd AI_Based_Glaucoma_Detection_System
```

### 2. Install dependencies

The current repository is notebook-based. In your Python environment, install the packages used by the notebook:

```bash
python -m pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow kaggle
```

Package compatibility depends on your Python version and environment. For repeatable setup, add and maintain a tested `requirements.txt` file.

### 3. Configure Kaggle access

Configure the Kaggle API credentials locally if required by the notebook. **Never commit `kaggle.json`, API keys, passwords, or access tokens to GitHub.**

### 4. Open and run the notebook

Open `AI_Based_Glaucoma_Detection_System.ipynb` in Jupyter Notebook or Google Colab and execute the cells in order. Confirm the dataset is downloaded correctly and review the generated evaluation outputs.

## ✅ Implementation Status

The notebook and accompanying documentation describe the following implemented workflow:

- Dataset download and organization
- Train/validation/test splitting
- Image preprocessing and augmentation
- EfficientNetB0 transfer learning and partial fine-tuning
- Model training
- Confusion matrix and classification report
- Model saving and single-image prediction

The following remain potential future improvements unless separately implemented and verified:

- ROC curve and ROC-AUC
- Sensitivity and specificity analysis
- Explainable AI (e.g., Grad-CAM)
- REST API and web interface
- Docker and cloud deployment
- Comparison with alternative model architectures

## 🔬 Future Improvements

- Report actual test-set metrics and image counts.
- Add reproducible output screenshots and error analysis.
- Examine class balance and potential data leakage.
- Add ROC-AUC, sensitivity, and specificity when implemented.
- Explore Grad-CAM or other explainability techniques.
- Refactor notebook code into reusable modules if the project grows.

## 👩‍💻 Author

**Vidhya Boda**

- GitHub: [@VidhyaBoda](https://github.com/VidhyaBoda)
- Project repository: [AI-Based Glaucoma Detection System](https://github.com/VidhyaBoda/AI_Based_Glaucoma_Detection_System)

## 📄 License

This project is released under the [MIT License](LICENSE).
