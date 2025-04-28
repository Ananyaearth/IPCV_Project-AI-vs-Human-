# AI vs Human-Generated Image Detection -- pipeline2 🧠📸
**Kaggle Competition:** [Detect AI vs Human Generated Images](https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images)

---

## 📋 Project Overview
This project tackles the challenge of distinguishing between **AI-generated** and **human-created** images using **smart feature engineering** and **ensemble modeling**.

Rather than deep CNNs, we extract meaningful handcrafted features and blend lightweight models to achieve strong performance.

---

## ⚙️ Approach

| Stage | Techniques Used |
|:---|:---|
| **Feature Extraction** | SNR, Laplacian Variance, Edge Density, Color Statistics (Skew, Kurtosis), Saturation Variance, Hue Variance, High Frequency Energy |
| **Feature Engineering** | Added `SatEdgeRatio`, `log_SNR` |
| **Data Cleaning** | Replaced inf/NaNs, filled with column means |
| **Scaling** | `RobustScaler` for stability against outliers |
| **Model 1** | XGBoost Classifier |
| **Model 2** | Neural Network (MLP) |
| **Model 3** | Logistic Regression |
| **Blending** | Weighted ensemble of XGBoost, NN, Logistic Regression |
| **Threshold Optimization** | Fine-tuned decision threshold for maximum F1 score |

---

## 🏆 Results

| Metric | Score |
|:---|:---|
| **Private Score** | 0.67900 |
| **Public Score** | 0.66774 |

Achieved using a blend of XGBoost, Neural Network, and Logistic Regression!

---



