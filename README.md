# ⚕️ Snakebite Clinical Decision Support System (CDSS)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://parth-makote-snakebite-triage-ml.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An AI-powered multi-output machine learning solution designed for rapid rural emergency triage, automated venom classification, and precise Polyvalent Anti-Snake Venom (ASV) dosage recommendation aligned with WHO and National Health Mission (NHM) clinical guidelines.**

---

## 🌐 Live Web Application

🚀 **Experience the live interactive application here:**  
👉 **[Launch Snakebite CDSS Web Portal](https://parth-makote-snakebite-triage-ml.streamlit.app)**

---

## 📌 Project Overview

Snakebite envenomation is a critical health concern in tropical regions, particularly in rural India. Delayed triage or incorrect dosage administration often leads to severe complications like neuromuscular paralysis or acute kidney injury (AKI). 

This project delivers a **Clinical Decision Support System (CDSS)** that processes patient admission vitals in real time to generate instant, standardized clinical action directives for medical officers.

---

## ⚙️ Key Technical Features

- **Multi-Output Machine Learning Architecture:** Uses a **Multi-Output Random Forest Classifier** to simultaneously predict three distinct targets from a single set of patient vitals:
  1. **Triage Severity Tier** (Low / Moderate / Critical Shock Risk)
  2. **Identified Venom Profile** (Neurotoxic / Hemotoxic / Dry Bite)
  3. **Required Polyvalent ASV Dosage** (Exact vial count)
- **Deterministic Data Pipeline:** Trained on a 10,000-sample dataset simulated using physiological thresholds established in WHO/NHM treatment protocols.
- **Model Serialization & Deployment:** Model artifacts and categorical encoders serialized using `Joblib` and deployed as a micro-web application on **Streamlit Community Cloud**.

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python
* **Machine Learning:** `scikit-learn` (Random Forest, MultiOutputClassifier)
* **Data Processing & Serialization:** `pandas`, `numpy`, `joblib`
* **Web Framework & Hosting:** `streamlit`, `Streamlit Cloud`

---

## 📁 Repository Structure

```text
├── app.py                         # Streamlit web application interface
├── snakebite_triage_model.pkl     # Serialized model & encoder artifacts
├── snakebite_multitarget_data.csv # Clinical training dataset
├── requirements.txt               # Dependencies list for deployment
└── README.md                      # Project documentation


📜 Clinical Disclaimer
This system is developed as an academic machine learning prototype and decision support tool. Clinical decisions in hospital settings should always be supervised by a licensed medical professional.
