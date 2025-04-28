
# AI vs Human-Generated Image Classification 🧠🖼️

This project solves the Kaggle challenge [Detect AI vs Human Generated Images](https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images) by building an optimized hybrid pipeline combining statistical features (SNR) and machine learning (XGBoost).

🚀 Final Leaderboard Scores:
- **Public Score**: 0.68753
- **Private Score**: 0.69187

---

## 🔥 Approach Overview

1. **Dataset Exploration**
   - Images categorized as either human-generated (label 0) or AI-generated (label 1).
   - CSVs provided for train and test splits.

2. **Feature Extraction**
   - Computed **Signal-to-Noise Ratio (SNR)** for all images efficiently using multithreading.
   - Extracted other statistical features (Edge Density, Saturation Variance, Hue Variance, High Frequency Energy).

3. **Feature Engineering**
   - Visualized feature distributions to select the most important ones.
   - Dropped noisy and less informative features (color skewness/kurtosis).

4. **Model Training (XGBoost)**
   - Trained a tuned XGBoost classifier on the selected features.
   - Early stopping based on validation AUC score.

5. **SNR Threshold Optimization**
   - Independently optimized the SNR threshold to separate Human vs AI using the F1-score as the metric.
   - Achieved a standalone SNR model with ~69% F1 score.

6. **Blending Strategy**
   - Combined XGBoost and SNR-based models through optimized weighted blending.
   - Blend weight automatically tuned to maximize performance.

7. **Final Submission**
   - Generated final prediction file with hard 0/1 labels.

---

## 📦 Folder Structure

```
.
├── final_pipeline.ipynb  # Full training and inference pipeline
├── train_features_updated.csv  # Updated train features with correct SNR
├── test_features_updated.csv   # Updated test features with correct SNR
├── final_submission_blended_optimized.csv  # Final Kaggle submission file
├── README.md
```

---

## ⚙️ Technologies Used

- Python 3.11
- OpenCV for image processing
- NumPy, Pandas for data manipulation
- XGBoost for machine learning
- scikit-learn for evaluation metrics
- Joblib for multithreading
- Matplotlib, Seaborn for visualizations

---

## 📈 Key Results

- SNR thresholding alone achieved ~69% F1 score.
- XGBoost classifier achieved ~94% Validation AUC.
- Final blended model achieved nearly **0.69 public/private leaderboard scores**.
- Robust performance across dataset shifts.

---

## ✨ Highlights

- **Feature Analysis**: Visualized SNR, Edge Density, Saturation Variance distributions.
- **Optimization**: Automated threshold and blending weight tuning.
- **Efficient Processing**: Parallelized SNR computation across all CPU cores.
- **Smart Blending**: Combined statistical and ML models for best generalization.

---

## 🙌 Acknowledgements

Thanks to Kaggle for hosting the competition and providing the dataset.  
This project was developed as part of a learning journey to build competition-level machine learning pipelines.

---
