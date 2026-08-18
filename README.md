# 👁️ AI-Based Glaucoma Detection System

### Deep Learning & Transfer Learning for Automated Glaucoma Detection from Retinal Fundus Images

---

## 📌 Project Overview

The **AI-Based Glaucoma Detection System** is a deep learning-based medical image classification project designed to automatically classify retinal fundus images into two categories:

* **Glaucoma**
* **Normal**

The project uses **EfficientNetB0**, a convolutional neural network pretrained on ImageNet, together with transfer learning and partial fine-tuning to learn visual patterns from retinal fundus images.

The complete workflow covers dataset preparation, train-validation-test splitting, image preprocessing, data augmentation, transfer learning, model training, evaluation, model saving, and individual image prediction.

The objective is to demonstrate how deep learning and computer vision can be applied to assist in the automated analysis of retinal fundus images for glaucoma detection.

> ⚠️ This project is intended for educational and research purposes. It is not a medical diagnostic system and should not be used as a substitute for professional clinical diagnosis.

---

# 🎯 Key Features

* 👁️ Automated Glaucoma vs Normal Classification
* 🖼️ Retinal Fundus Image Processing
* 📊 Train / Validation / Test Dataset Split
* 🔄 Data Augmentation
* 🧠 EfficientNetB0 Transfer Learning
* 🔧 Partial Fine-Tuning
* 🏗️ Custom Deep Learning Classification Head
* 📈 Model Training and Validation
* 📊 Confusion Matrix Evaluation
* 📋 Classification Report
* 💾 Trained Model Saving
* 🔮 Single Image Prediction
* 🐍 Python + TensorFlow/Keras Implementation

---

# 🏗️ System Architecture

```text
                 Retinal Fundus Images
                          │
                          ▼
                  Kaggle Dataset
                          │
                          ▼
                 Dataset Extraction
                          │
                          ▼
               Dataset Organization
                          │
                          ▼
              Train / Validation / Test
                    Split (70/15/15)
                          │
                          ▼
              Image Preprocessing
                  Resize → 224×224
                          │
                          ▼
                Data Augmentation
           ┌──────────────┼──────────────┐
           │              │              │
       Rotation         Zoom       Horizontal Flip
                          │
                    Brightness
                          │
                          ▼
                  EfficientNetB0
                  ImageNet Weights
                          │
                          ▼
              Transfer Learning Model
                          │
                          ▼
             Custom Classification Head
                          │
                          ▼
                 Binary Classification
                          │
                    ┌─────┴─────┐
                    │           │
                Glaucoma      Normal
                          │
                          ▼
                   Model Evaluation
                          │
              ┌───────────┴───────────┐
              │                       │
       Confusion Matrix       Classification Report
                          │
                          ▼
                    Model Saving
                          │
                          ▼
                  Single Image
                    Prediction
```

---

# 🧠 Model Architecture

The project uses:

### EfficientNetB0

```text
EfficientNetB0
│
├── ImageNet pretrained weights
├── include_top=False
└── Input Shape: 224 × 224 × 3
```

The pretrained convolutional base is combined with a custom classification head.

### Custom Classification Head

```text
EfficientNetB0
      │
      ▼
Global Average Pooling
      │
      ▼
Dense(512) + BatchNormalization + Dropout(0.4)
      │
      ▼
Dense(256) + BatchNormalization + Dropout(0.4)
      │
      ▼
Dense(128) + BatchNormalization + Dropout(0.3)
      │
      ▼
Dense(64) + BatchNormalization + Dropout(0.3)
      │
      ▼
Dense(32)
      │
      ▼
Dense(16)
      │
      ▼
Sigmoid Output
      │
      ▼
Glaucoma / Normal
```

---

# 📊 Dataset

## Dataset Used

The project downloads the glaucoma detection dataset through the Kaggle API.

Dataset identifier used in the project:

```text
dasa7753912/glaucoma-detection
```

The notebook uses the following dataset organization:

```text
Acrima/
│
├── Glaucoma/
│   └── retinal fundus images
│
└── Normal/
    └── retinal fundus images
```

The classification task contains two classes:

```text
Glaucoma
Normal
```

---

# 📂 Dataset Split

The dataset is divided into:

```text
Dataset
│
├── Train
│   ├── Glaucoma
│   └── Normal
│
├── Validation
│   ├── Glaucoma
│   └── Normal
│
└── Test
    ├── Glaucoma
    └── Normal
```

The notebook performs the split in two stages:

```text
70% → Training
30% → Temporary Set

Temporary Set
│
├── 15% → Validation
└── 15% → Test
```

Therefore, the final approximate distribution is:

| Dataset    | Percentage |
| ---------- | ---------: |
| Training   |        70% |
| Validation |        15% |
| Testing    |        15% |

The split uses:

```text
random_state = 42
```

to make the dataset split reproducible.

---

# 🔄 Image Preprocessing

All images are resized to:

```text
224 × 224
```

The project uses EfficientNet preprocessing:

```python
preprocess_input
```

The same preprocessing is applied to:

* Training images
* Validation images
* Test images
* Individual prediction images

---

# 🔄 Data Augmentation

Data augmentation is applied to the training dataset to improve model generalization.

The implemented augmentation techniques include:

```text
Rotation
Zoom
Horizontal Flip
Brightness Adjustment
```

Configuration used:

```python
rotation_range = 15
zoom_range = 0.2
horizontal_flip = True
brightness_range = [0.8, 1.2]
```

Validation and test datasets use preprocessing without augmentation.

---

# 🧠 Transfer Learning

The project uses:

```text
EfficientNetB0
```

with pretrained:

```text
ImageNet weights
```

The original classification head is removed using:

```python
include_top=False
```

Input shape:

```text
224 × 224 × 3
```

The EfficientNetB0 base is partially fine-tuned.

The notebook keeps the earlier layers frozen while allowing the final 30 layers to train:

```python
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False
```

This allows the model to retain general visual representations while adapting deeper layers to retinal fundus image patterns.

---

# ⚙️ Model Compilation

The model uses:

### Optimizer

```text
Adam
```

### Learning Rate

```text
5e-5
```

### Loss Function

```text
Binary Crossentropy
```

### Metric

```text
Accuracy
```

Configuration:

```python
model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

---

# 🚀 Model Training

The model is trained using:

```text
Training Dataset
       │
       ▼
EfficientNetB0
       │
       ▼
Custom Classification Head
       │
       ▼
Binary Crossentropy
       │
       ▼
Validation Dataset
```

Training configuration currently implemented in the notebook:

```text
Epochs: 10
Batch Size: 32
Image Size: 224 × 224
```

Training is performed using:

```python
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)
```

---

# 📊 Model Evaluation

After training, predictions are generated on the test dataset.

The prediction probability is converted into a binary class using a threshold of:

```text
0.5
```

The evaluation workflow is:

```text
Test Images
     │
     ▼
Trained Model
     │
     ▼
Prediction Probability
     │
     ▼
Threshold = 0.5
     │
     ▼
Predicted Class
     │
     ▼
Compare with True Labels
     │
     ├── Confusion Matrix
     │
     └── Classification Report
```

---

# 📈 Evaluation Metrics

The current implementation generates:

### Confusion Matrix

The confusion matrix is calculated using:

```python
confusion_matrix(
    y_true,
    y_pred
)
```

### Classification Report

The project also generates:

```python
classification_report(
    y_true,
    y_pred,
    target_names=["Glaucoma", "Normal"]
)
```

The classification report provides:

* Precision
* Recall
* F1-score
* Support

> Note: ROC-AUC, specificity, sensitivity, and ROC curve are not currently calculated in the uploaded notebook. These can be added as a future improvement for a stronger medical-image evaluation pipeline.

---

# 💾 Model Saving

After training, the model is saved as an HDF5 model:

```python
model.save("gloucoma_detection.h5")
```

Recommended future filename:

```text
glaucoma_detection_efficientnetb0.h5
```

The recommended filename improves clarity and removes the `gloucoma` spelling mistake from the current notebook.

---

# 🔮 Single Image Prediction

The project also supports prediction on an individual retinal fundus image.

The image processing pipeline is:

```text
Input Image
     │
     ▼
Convert to RGB
     │
     ▼
Resize to 224 × 224
     │
     ▼
Convert to NumPy Array
     │
     ▼
EfficientNet Preprocessing
     │
     ▼
Model Prediction
     │
     ▼
Probability
     │
     ▼
Threshold = 0.5
     │
     ├── Glaucoma
     │
     └── Normal
```

Example:

```python
pred = model.predict(img_array, verbose=0)[0][0]

if pred > 0.5:
    print("Normal")
else:
    print("Glaucoma")
```

---

# 🛠️ Technology Stack

| Category                   | Technologies                    |
| -------------------------- | ------------------------------- |
| Programming                | Python                          |
| Deep Learning              | TensorFlow, Keras               |
| Model                      | EfficientNetB0                  |
| Transfer Learning          | ImageNet                        |
| Computer Vision            | TensorFlow / Keras              |
| Data Processing            | NumPy                           |
| Data Analysis              | Pandas                          |
| Visualization              | Matplotlib, Seaborn             |
| Machine Learning Utilities | Scikit-learn                    |
| Image Processing           | Pillow                          |
| Dataset Source             | Kaggle                          |
| Development Environment    | Jupyter Notebook / Google Colab |
| Version Control            | Git, GitHub                     |

---

# 📂 Project Structure

```text
AI_Based_Glaucoma_Detection_System/
│
├── notebooks/
│   └── AI_Based_Glaucoma_Detection_System.ipynb
│
├── models/
│   └── README.md
│
├── results/
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── training_history.png
│
├── src/
│   ├── data_preprocessing.py
│   ├── data_split.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

> The current implementation is primarily contained in the Jupyter Notebook. The `src/`, `results/`, and `models/` structure can be introduced as the project is refactored into a production-style ML repository.

---

# 🚀 Installation

## 1️⃣ Clone Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>

cd AI_Based_Glaucoma_Detection_System
```

---

## 2️⃣ Create Virtual Environment

### Using Python

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

Activate on Linux / macOS:

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

The project uses the following major Python libraries:

```text
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
kaggle
```

---

# 🔐 Kaggle Dataset Setup

The notebook downloads the dataset using the Kaggle API.

The Kaggle credentials file should **never be committed to GitHub**.

Use:

```text
kaggle.json
```

locally and configure the Kaggle API according to your environment.

### ⚠️ Security

Never upload:

```text
kaggle.json
```

to GitHub.

Never commit:

```text
API keys
Passwords
Access tokens
Private credentials
```

---

# ▶️ Running the Project

The current implementation is provided as a Jupyter Notebook.

Open:

```text
notebooks/
└── AI_Based_Glaucoma_Detection_System.ipynb
```

Run the notebook sequentially.

The workflow is:

```text
1. Configure Kaggle
2. Download Dataset
3. Extract Dataset
4. Split Dataset
5. Inspect Dataset
6. Create Image Generators
7. Apply Data Augmentation
8. Load EfficientNetB0
9. Build Classification Head
10. Compile Model
11. Train Model
12. Evaluate Model
13. Save Model
14. Perform Single Image Prediction
```

---

# 📊 Project Workflow

```text
                 ┌─────────────────────┐
                 │   Kaggle Dataset    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Dataset Preparation │
                 └──────────┬──────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │ Train / Validation / Test│
              └─────────────┬─────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Image Preprocessing│
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Augmentation   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   EfficientNetB0    │
                 │  ImageNet Weights   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Custom Dense Head   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      Training       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Evaluation       │
                 ├─────────────────────┤
                 │ Confusion Matrix    │
                 │ Classification Report│
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Model Prediction  │
                 └─────────────────────┘
```

---

# 📌 Current Project Status

| Component                   | Status                |
| --------------------------- | --------------------- |
| Dataset Download            | ✅ Implemented         |
| Dataset Extraction          | ✅ Implemented         |
| Train/Validation/Test Split | ✅ Implemented         |
| Image Preprocessing         | ✅ Implemented         |
| Data Augmentation           | ✅ Implemented         |
| EfficientNetB0              | ✅ Implemented         |
| Transfer Learning           | ✅ Implemented         |
| Partial Fine-Tuning         | ✅ Implemented         |
| Custom Classification Head  | ✅ Implemented         |
| Model Training              | ✅ Implemented         |
| Confusion Matrix            | ✅ Implemented         |
| Classification Report       | ✅ Implemented         |
| Model Saving                | ✅ Implemented         |
| Single Image Prediction     | ✅ Implemented         |
| ROC Curve                   | 🔄 Future Improvement |
| ROC-AUC                     | 🔄 Future Improvement |
| Specificity                 | 🔄 Future Improvement |
| Sensitivity Analysis        | 🔄 Future Improvement |
| Explainable AI              | 🔄 Future Improvement |
| REST API                    | 🔄 Future Improvement |
| Web Application             | 🔄 Future Improvement |
| Docker Deployment           | 🔄 Future Improvement |
| Cloud Deployment            | 🔄 Future Improvement |

---

# 🔬 Medical AI Considerations

Glaucoma detection is a medical imaging problem where model evaluation should go beyond simple accuracy.

Future versions of this project should focus on:

```text
Sensitivity / Recall
Specificity
Precision
F1 Score
ROC-AUC
Confusion Matrix
Calibration
Explainability
```

This is especially important because false negatives in glaucoma screening can be clinically significant.

---

# 🔮 Future Improvements

## 🧠 Model Improvements

* Compare EfficientNetB0 with EfficientNetB3
* Compare ResNet architectures
* Compare DenseNet architectures
* Hyperparameter tuning
* Learning-rate scheduling
* Early stopping
* Cross-validation
* Class imbalance handling

## 📊 Evaluation Improvements

* ROC Curve
* ROC-AUC
* Sensitivity
* Specificity
* Precision-Recall Curve
* Calibration Curve
* Error Analysis

## 🔍 Explainable AI

Future versions can include:

* Grad-CAM
* Saliency Maps
* Model attention visualization
* Explainable predictions

This can help visualize which retinal regions influenced the model prediction.

## 🚀 Deployment

Future versions may include:

```text
TensorFlow Model
       │
       ▼
FastAPI
       │
       ▼
REST API
       │
       ▼
Web Application
```

Potential deployment technologies:

* FastAPI
* Streamlit
* Docker
* Cloud deployment

---

# 🧪 Research & Experimentation

Future experiments can compare:

```text
                    CNN Models
                       │
        ┌──────────────┼──────────────┐
        │              │              │
 EfficientNetB0      ResNet         DenseNet
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
                Model Comparison
                       │
                       ▼
        Accuracy / Recall / Specificity
             F1 / ROC-AUC
```

The goal is to identify a model that provides a better balance between performance, computational efficiency, and generalization.

---

# 📷 Project Screenshots

Recommended screenshots for the repository:

## Dataset Structure

*Add dataset organization screenshot here.*

---

## Training

*Add training output screenshot here.*

---

## Confusion Matrix

*Add confusion matrix screenshot here.*

---

## Classification Report

*Add classification report screenshot here.*

---

## Sample Prediction

*Add sample glaucoma/normal prediction screenshot here.*

---

# 💡 Project Highlights

✅ Deep Learning Based Glaucoma Detection

✅ Retinal Fundus Image Classification

✅ EfficientNetB0 Transfer Learning

✅ ImageNet Pretrained Model

✅ Custom Classification Head

✅ Data Augmentation

✅ Partial Fine-Tuning

✅ Train / Validation / Test Pipeline

✅ Confusion Matrix

✅ Classification Report

✅ Single Image Prediction

✅ Reproducible Dataset Split

✅ TensorFlow / Keras Implementation

---

# 🎓 Learning Outcomes

This project demonstrates practical experience with:

* Deep Learning
* Convolutional Neural Networks
* Transfer Learning
* EfficientNet
* Computer Vision
* Medical Image Classification
* TensorFlow
* Keras
* Data Augmentation
* Model Evaluation
* Binary Classification
* Git and GitHub Project Organization

---

# 👨‍💻 Author

**Vidhya Boda**

**AI / ML | Data Analytics | Data Engineering**

GitHub:

`VidhyaBoda`

---

# 🙏 Acknowledgements

This project uses technologies and libraries including:

* TensorFlow
* Keras
* EfficientNet
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Pillow
* Kaggle

---

# ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.

---

# 📄 License

This project is released under the **MIT License**.

---

<div align="center">

### 👁️ AI-Based Glaucoma Detection System

**Deep Learning • Computer Vision • Medical AI • Transfer Learning**

Made with ❤️ using Python & TensorFlow By **VIDHYA BODA**

</div>
