# 🛡️ Multimodal Deepfake Detection System

## 📌 Overview

The **Multimodal Deepfake Detection System** is a deep learning-based application designed to detect AI-generated (fake) videos by analyzing both **visual** and **audio** information. The system integrates computer vision, audio signal processing, and multimodal feature fusion to improve deepfake detection accuracy.

This project was developed as a collaborative team project, where each member was responsible for a dedicated module that was later integrated into a unified application.

---

# 🎯 Objectives

* Detect manipulated (deepfake) videos using facial analysis.
* Analyze audio characteristics for synthetic speech detection.
* Combine video and audio representations through multimodal fusion.
* Provide an interactive interface for uploading and analyzing videos.

---

# 🚀 Key Features

* 📹 Video-based deepfake detection
* 🎵 Audio deepfake detection
* 🤝 Feature-level multimodal fusion
* 🧠 Deep learning-based classification
* 📊 Confidence score visualization
* 🖥️ Interactive Streamlit dashboard

---

# 🏗️ System Architecture

```text
                    Input Video
                         │
          ┌──────────────┴──────────────┐
          │                             │
     Video Processing              Audio Processing
          │                             │
   Frame Extraction             Audio Extraction
          │                             │
     Face Detection             Audio Preprocessing
          │                             │
   Face Preprocessing          MFCC Feature Extraction
          │                             │
      ResNet18 Model            Audio Classification
          │                             │
     Video Features            Audio Features
          └──────────────┬──────────────┘
                         │
               Feature-Level Fusion
                         │
                 Final Prediction
                 (Real / AI Generated)
```

---

# 🧠 Technologies Used

### Programming Language

* Python

### Deep Learning

* PyTorch
* Torchvision

### Computer Vision

* OpenCV
* MTCNN
* Pillow

### Audio Processing

* Librosa
* NumPy

### User Interface

* Streamlit

### Development Tools

* Git
* GitHub

---

# 📂 Repository Structure

```text
multimodal-deepfake-detection/
│
├── audio_module/
├── fusion_module/
├── models/
├── ui/
├── video_module/
├── data/
├── deepfake_dataset/
├── tests/
├── requirements.txt
└── README.md
```

---

# 👥 Team Contributions

## 👤 Member 1 – Video Processing & Deepfake Detection

Responsibilities:

* Video frame extraction
* Face detection using MTCNN
* Face preprocessing
* ResNet18 model development
* Model training
* Model evaluation
* Image prediction
* Video prediction pipeline
* Performance testing
* GitHub integration

### Video Module Performance

* **Training Accuracy:** 97.67%
* **Validation Accuracy:** 84.25%
* **Test Accuracy:** 81.17%

---

## 👤 Member 2 – Audio Processing

Responsibilities:

* Audio extraction
* Audio preprocessing
* MFCC feature extraction
* Audio classification
* Audio inference pipeline

---

## 👤 Member 3 – Fusion & User Interface

Responsibilities:

* Feature-level multimodal fusion
* Fusion classifier implementation
* Streamlit dashboard
* System integration
* Testing and deployment

---

# 📊 Project Workflow

1. Upload a video.
2. Extract video frames.
3. Detect and preprocess faces.
4. Analyze facial features using ResNet18.
5. Extract and analyze audio features.
6. Generate video and audio embeddings.
7. Fuse both feature representations.
8. Produce the final prediction.

---

# 📈 Performance

### Video Module

| Metric              | Result     |
| ------------------- | ---------- |
| Training Accuracy   | **97.67%** |
| Validation Accuracy | **84.25%** |
| Test Accuracy       | **81.17%** |

The reported performance corresponds to the video classification module evaluated on the project dataset.

---

# ⚠️ Limitations

* Performance depends on the diversity and quality of the training dataset.
* Face-based analysis requires detectable human faces in the input video.
* Highly realistic AI-generated videos may remain challenging to classify.
* The project is intended as an academic prototype and research implementation.

---

# 🔮 Future Scope

* Transformer-based video architectures
* Improved multimodal fusion techniques
* Larger and more diverse datasets
* Real-time inference
* Performance optimization
* Cloud deployment

---

# 📚 Academic Purpose

This repository has been developed for educational, research, and academic demonstration purposes to explore multimodal approaches for deepfake detection using computer vision and audio analysis.

--

# 👨‍💻 Contributors

Developed collaboratively by a three-member team.

* **Member 1:** Video Processing & Deepfake Detection
* **Member 2:** Audio Processing & Classification
* **Member 3:** Fusion Module & User Interface

--

## ⭐ If you found this project interesting, consider giving the repository a star!

---
📄 License

This repository is shared for academic demonstration and portfolio purposes. The source code is not intended for unrestricted reuse or commercial deployment without permission from the project contributors.
