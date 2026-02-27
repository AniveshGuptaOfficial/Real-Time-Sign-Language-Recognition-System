# ✨ Real-Time Sign Language Recognition System

## 📖 Project Overview

The Real-Time Sign Language Recognition System is an intelligent Digital Image Processing and Machine Learning application designed to convert hand gestures into text and speech in real time.
The system captures live video input through a webcam and detects hand landmarks using the MediaPipe framework. A total of 21 anatomical hand landmarks are extracted and transformed into a 42-dimensional feature vector (x, y coordinates). These features are normalized relative to the wrist landmark to ensure position invariance and then classified using a trained Support Vector Machine (SVM) model with an RBF kernel.
The recognized characters — including A–Z, 0–9, and SPACE — are sequentially combined to form meaningful words and complete sentences. The generated text can also be converted into audible speech, making the system a practical assistive communication solution.

## 🎯 Objectives

* Develop a real-time hand gesture recognition system.
* Extract anatomical hand landmarks accurately.
* Apply machine learning techniques for gesture classification.
* Enable sentence formation from sequential character detection.
* Integrate text-to-speech functionality.
* Provide an intelligent assistive communication interface.

## 🧠 System Architecture

Webcam Frame
        ↓
MediaPipe Hand Detection
        ↓
21 Landmark Extraction (42 Features)
        ↓
Wrist-Based Coordinate Normalization
        ↓
SVM Classification (RBF Kernel)
        ↓
Character Prediction
        ↓
Sentence Formation
        ↓
Text-to-Speech Output

## 🛠 Technologies Utilized

* Python – Core implementation language
* OpenCV – Real-time image processing
* MediaPipe – Hand landmark detection
* NumPy – Numerical computations
* Pandas – Dataset handling
* Scikit-learn (SVM) – Machine learning classification
* Pyttsx3 – Offline text-to-speech engine

## 🔢 Feature Representation

* Each gesture is mathematically represented as:
* F = [x1,y1,x2,y2,...,x21,y21]
* Total features: 42
* To achieve translation invariance, the landmarks are normalized relative to the wrist coordinate:
* (xi',yii') = (xi-x0,yi-y0)
* where (x0,y0) represents the wrist landmark.

## 🤖 Machine Learning Model

* Classifier: Support Vector Machine (SVM)
* Kernel: Radial Basis Function (RBF)
* Classification Type: Multi-class
* Probability Estimation: Enabled
* The RBF kernel enables non-linear separation of gesture patterns in high-dimensional feature space, improving classification accuracy.

## 📢 System Output

* The system performs:
* Real-time gesture recognition
* Character prediction
* Sentence construction
* Optional speech generation
* Example Output: ANIVESH GUPTA

## 🎓 Academic Significance

* This project demonstrates the integration of:
* Real-time image acquisition
* Landmark-based feature extraction
* Spatial normalization techniques
* Supervised machine learning classification
* Human-computer interaction design
* Assistive technology implementation
* The system combines Digital Image Processing principles with applied Machine Learning to create a practical real-world solution.

## 👨‍💻 Author

* Anivesh Gupta
* Real-Time Sign Language Recognition System
