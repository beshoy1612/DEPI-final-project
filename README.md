# 🩺 Medical Multimodal Triage Assistant

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

![Keras](https://img.shields.io/badge/Keras-DeepLearning-red)

![Transformers](https://img.shields.io/badge/HuggingFace-BERT-yellow)

![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-ff4b4b)

![License](https://img.shields.io/badge/License-Educational-green)

</p>

---

## 📖 Overview

Medical Multimodal Triage Assistant is an AI-powered decision support system that combines **Computer Vision** and **Natural Language Processing** to classify the medical risk level of skin lesions.

Instead of relying only on medical images, the system also considers structured clinical information such as:

- Patient age
- Gender
- Lesion location
- Diagnosis method

Both modalities are combined to improve the final prediction.

---

## 🎯 Objectives

- Develop a Computer Vision model for skin lesion classification.
- Build an NLP model to understand patient metadata.
- Fuse image and text predictions into a single decision.
- Reduce data leakage during preprocessing.
- Provide an interactive medical assistant using Streamlit.

---

# ✨ Features

- Skin lesion image classification
- Clinical text analysis
- Multimodal prediction
- Risk level estimation
- Confidence score
- Probability visualization
- Interactive Streamlit interface

---

# 🏗️ System Architecture

```
                  HAM10000 Dataset
                         │
          ┌──────────────┴──────────────┐
          │                             │
     Skin Images                  Metadata
          │                             │
          │                       Text Builder
          │                             │
   EfficientNetB0                    BERT
          │                             │
          └──────────────┬──────────────┘
                         │
                Weighted Average Fusion
                         │
                Risk Classification
                         │
                 Streamlit Interface
```

---

# 📂 Project Structure

```
Multimodal-Medical-Triage-Assistant

│

├── app.py

├── README.md

│

├── cv_models/

│ └── effb0_finetuned_best.keras

│

├── nlp_model/

│ ├── config.json

│ ├── tokenizer.json

│ ├── tokenizer_config.json

│ ├── special_tokens_map.json

│ ├── tf_model.h5

│ └── label_encoder.pkl

│

├── data/

│ ├── HAM10000_metadata.csv

│ ├── cv_train.csv

│ ├── cv_test.csv

│ ├── nlp_train.csv

│ └── nlp_test.csv

│

└── Src/

├── eda.ipynb

├── preprocessing.ipynb

├── CV_model.ipynb

├── NLP.ipynb

├── main.py

└── nlpmodels.py
```

---

# 📊 Dataset

The project uses the **HAM10000** dataset, which contains dermoscopic images of pigmented skin lesions collected from different medical institutions.

The metadata includes:

- Image ID
- Lesion ID
- Age
- Gender
- Localization
- Diagnosis
- Diagnosis Type

Original classes:

| Code | Disease |
|------|-----------------------------|
| nv | Melanocytic Nevi |
| mel | Melanoma |
| bkl | Benign Keratosis |
| bcc | Basal Cell Carcinoma |
| akiec | Actinic Keratosis |
| vasc | Vascular Lesion |
| df | Dermatofibroma |

---

# ⚙️ Data Preprocessing

The preprocessing pipeline includes:

✔ Missing value handling

✔ Age outlier clipping using IQR

✔ Train/Test split based on lesion IDs

✔ Data leakage prevention

✔ Class weight computation

✔ Text generation for NLP

✔ Risk mapping

---

# 🧠 Computer Vision Module

Model:

- EfficientNetB0

Training strategy:

- Transfer Learning
- Frozen Layers
- Fine-Tuning

Image size:

```
224 × 224
```

Output classes:

```
Low
Medium
High
```

---

# 💬 NLP Module

Model:

```
bert-base-uncased
```

Tokenizer:

```
BertTokenizer
```

Example input:

```
Dermatology case:
45 year old male patient.
Lesion location: back.
Diagnosis method: histo.
```

Output:

```
Low
Medium
High
```

---

# 🔀 Multimodal Fusion

The final prediction is obtained using weighted probability averaging.

```
Final Probability

=

0.5 × CV Prediction

+

0.5 × NLP Prediction
```

The class with the highest probability is selected as the final prediction.

---

# 🖥️ User Interface

The Streamlit application allows users to:

- Upload a skin lesion image.
- Enter clinical information.
- Predict the medical risk level.
- Display prediction confidence.
- Visualize class probabilities.

---

# 📈 Technologies

| Category | Technology |
|-----------|------------|
| Language | Python |
| CV | EfficientNetB0 |
| NLP | BERT |
| Framework | TensorFlow |
| Deep Learning | Keras |
| Interface | Streamlit |
| Visualization | Matplotlib |
| Data Processing | Pandas |
| ML Utilities | Scikit-learn |

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Multimodal-Medical-Triage-Assistant.git
```

Enter the project directory

```bash
cd Multimodal-Medical-Triage-Assistant
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run

```bash
streamlit run app.py
```

The application runs locally at

```
http://localhost:8501
```

---

# 📷 Example Workflow

1. Upload a skin lesion image.
2. Enter the patient's clinical description.
3. Click **Analyze**.
4. The system predicts the risk level.
5. View confidence and probability chart.

---

# 📌 Future Improvements

- Cloud deployment
- Mobile application
- Explainable AI (Grad-CAM)
- Clinical report generation
- Larger dermatology datasets
- Improved multimodal fusion strategies


