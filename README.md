# ✨ Real-Time Sign Language Recognition System

## 📖 Project Overview

The Real-Time Sign Language Recognition System is an intelligent Digital Image Processing and Machine Learning application designed to convert hand gestures into text and speech in real time. The system supports both a **desktop application** (OpenCV window) and a **web-based interface** (FastAPI + browser UI).

A webcam feed is processed with **MediaPipe Hands** to detect 21 anatomical hand landmarks. These (x, y) coordinates are transformed into a 42‑dimensional feature vector, normalized relative to the wrist landmark for translation invariance, and classified by a trained **Support Vector Machine (SVM)** model (RBF kernel). Recognized characters (A–Z, 0–9, SPACE) are combined into words and sentences, which can optionally be spoken using text‑to‑speech.

## 🎯 Objectives

- Develop a real-time hand gesture recognition system.
- Extract anatomical hand landmarks accurately.
- Apply machine learning techniques for gesture classification.
- Enable sentence formation from sequential character detection.
- Integrate text-to-speech functionality.
- Provide both desktop and web-based assistive communication interfaces.

## 🧱 Project Structure

```text
REAL-TIME-SIGN-LANGUAGE-RECOGNITION-SYSTEM/
│
├── asl_dataset/                     # Raw image dataset (optional; ignored in Git)
├── dataset.csv                      # Final landmark dataset used for training
├── sign_model.pkl                   # Trained SVM model
│
├── collect_data.py                  # Option 1: collect dataset directly from webcam
├── create_dataset_from_images.py    # Option 2: create dataset from saved images
├── train_model.py                   # Train SVM model and save sign_model.pkl
├── sign_language_recognition.py     # Desktop real-time recognition (OpenCV window)
│
├── web_app/
│   ├── main.py                      # FastAPI backend for web interface
│   └── static/
│       └── index.html               # Browser UI for real-time recognition
│
├── directkeys.py                    # Keyboard control helper (desktop version)
├── README.md
└── faceenv/                         # Python virtual environment (ignored in Git)
```

### Data Collection

You can build the dataset in **two ways**:

1. `collect_data.py` – capture hand landmarks directly from the webcam and append them to `dataset.csv`.
2. `create_dataset_from_images.py` – generate `dataset.csv` from pre‑captured images stored in `asl_dataset/`.

### Model Training

- `train_model.py` loads `dataset.csv`, trains an SVM (RBF kernel) classifier, and saves the trained model as `sign_model.pkl`.

### Real-Time Recognition

There are **two modes** for real-time sign detection:

1. **Desktop Mode** – `sign_language_recognition.py`  
   - Uses OpenCV to show a live webcam window.  
   - Performs real-time prediction, sentence formation, and text-to-speech.  
   - Keyboard controls (e.g., add letter, space, backspace, clear, speak) manage the constructed sentence.

2. **Web Mode** – `web_app/main.py` + `web_app/static/index.html`  
   - `main.py` exposes FastAPI endpoints to receive frames, run MediaPipe + SVM, and perform sentence operations.  
   - `index.html` uses JavaScript to stream webcam frames from the browser, display current letter and confidence, and provide buttons for **Add Letter**, **Space**, **Backspace**, **Clear**, and **Speak (browser TTS)**.

## 🧠 System Architecture

For both desktop and web pipelines, the core flow is:

Webcam Frame  
→ MediaPipe Hand Detection  
→ 21 Landmark Extraction (42 Features)  
→ Wrist-Based Coordinate Normalization  
→ SVM Classification (RBF Kernel)  
→ Character Prediction  
→ Sentence Formation  
→ Optional Text-to-Speech Output

## 🛠 Technologies Utilized

- **Python** – Core implementation language  
- **OpenCV** – Real-time image acquisition and visualization  
- **MediaPipe** – Robust hand landmark detection  
- **NumPy / Pandas** – Numerical computation and dataset handling  
- **Scikit-learn (SVM)** – Machine learning classification  
- **Pyttsx3** – Offline text-to-speech engine (desktop mode)  
- **FastAPI + Uvicorn** – Backend for web-based recognition  
- **HTML / CSS / JavaScript** – Frontend for the web interface (camera access, UI, Web Speech API)

## 🔢 Feature Representation

- Each gesture is represented as:  
  \( F = [x_1, y_1, x_2, y_2, ..., x_{21}, y_{21}] \)  
- Total features: 42.
- To achieve translation invariance, landmarks are normalized relative to the wrist coordinate (landmark 0):  
  \( (x_i', y_i') = (x_i - x_0, y_i - y_0) \)  
  where \((x_0, y_0)\) is the wrist landmark.

## 🤖 Machine Learning Model

- Classifier: **Support Vector Machine (SVM)**  
- Kernel: **Radial Basis Function (RBF)**  
- Type: Multi-class classification  
- Probability Estimation: Enabled  

The RBF kernel captures non-linear decision boundaries in the high-dimensional feature space, improving classification accuracy for complex hand shapes.

## 📢 System Output

The system supports:

- Real-time gesture recognition (desktop and web).
- Per-frame character prediction with confidence.
- Sentence construction from recognized characters.
- Optional speech generation of the formed sentence (desktop: `pyttsx3`, web: browser Web Speech API).
- Example sentence: `ANIVESH GUPTA`.

## 🎓 Academic Significance

This project demonstrates the integration of:

- Real-time image acquisition and pre-processing.  
- Landmark-based feature extraction and spatial normalization.  
- Supervised machine learning classification with SVM.  
- Human-computer interaction via desktop and web interfaces.  
- Practical assistive technology for sign-to-speech communication.

It combines Digital Image Processing principles with applied Machine Learning to create a practical, extensible real-world solution.

## 👨‍💻 Author

- **Anivesh Gupta**  
  Real-Time Sign Language Recognition System