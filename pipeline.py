# ==============================================
# AI vs Human-Generated Image Detection - Final Complete Pipeline
# Author: Ananya Verma
# ==============================================

# ================
# 1. Libraries
# ================
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import skew, kurtosis
from scipy.fft import fft2
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# ================
# 2. Feature Extraction Functions
# ================

def calculate_snr(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (128, 128))
    signal = np.mean(image)
    noise = np.std(image)
    return float('inf') if noise == 0 else 20 * np.log10(signal / noise)

def extract_features_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (512, 512))
    features = []
    features.append(np.mean(image) / (np.std(image) + 1e-5))
    features.append(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    features.append(np.sum(cv2.Canny(image, 100, 200) > 0) / (512 * 512))
    for i in range(3):
        channel = image[:, :, i].flatten()
        features.append(skew(channel))
        features.append(kurtosis(channel))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features.append(np.var(hsv[:, :, 1].flatten()))
    features.append(np.var(hsv[:, :, 0].flatten()))
    f = np.abs(fft2(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))
    features.append(np.sum(f[30:, 30:]) / np.sum(f))
    return features

def build_features_dataframe_threadpool(image_folder, max_workers=8):
    image_names = os.listdir(image_folder)
    features_list = []

    def extract_single(img_name):
        return extract_features_from_image(os.path.join(image_folder, img_name))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_single, img_name) for img_name in tqdm(image_names, position=0, leave=True)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting Results", position=0, leave=True):
            features_list.append(future.result())

    columns = ['SNR', 'LaplacianVar', 'EdgeDensity',
               'BlueSkew', 'BlueKurtosis', 'GreenSkew', 'GreenKurtosis',
               'RedSkew', 'RedKurtosis', 'SatVariance', 'HueVariance', 'HighFreqEnergy']
    df = pd.DataFrame(features_list, columns=columns)
    df['ImageName'] = image_names
    return df

# ================
# 3. Load Dataset and Extract Features
# ================

train_labels = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/train.csv')
test_labels = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/test.csv')
train_labels['file_name'] = train_labels['file_name'].apply(lambda x: os.path.basename(x))
test_labels['id'] = test_labels['id'].apply(lambda x: os.path.basename(x))

train_images_path = '/kaggle/input/ai-vs-human-generated-dataset/train_data'
test_images_path = '/kaggle/input/ai-vs-human-generated-dataset/test_data_v2'

print("⚡ Extracting Train Features...")
train_features = build_features_dataframe_threadpool(train_images_path)

print("⚡ Extracting Test Features...")
test_features = build_features_dataframe_threadpool(test_images_path)

train_df = pd.merge(train_labels, train_features, left_on='file_name', right_on='ImageName')
test_df = pd.merge(test_labels, test_features, left_on='id', right_on='ImageName')
train_df.drop(columns=['ImageName'], inplace=True)
test_df.drop(columns=['ImageName'], inplace=True)

train_df.to_csv('/kaggle/working/train_features_updated.csv', index=False)
test_df.to_csv('/kaggle/working/test_features_updated.csv', index=False)

print("✅ Train and Test feature CSV files saved!")

# ================
# 4. Exploratory Data Analysis (EDA)
# ================

plt.figure(figsize=(10,6))
sns.histplot(data=train_df, x='SNR', hue='label', kde=True, stat='density', common_norm=False, palette='Set2')
plt.title('SNR Distribution: Human (0) vs AI (1)')
plt.grid()
plt.show()

print("🔎 SNR Statistics by Class")
print(train_df.groupby('label')['SNR'].agg(['mean', 'median', 'std', 'min', 'max', 'count']))

# ================
# 5. Train XGBoost Model (using only strong features)
# ================

features_to_use = [
    'SNR', 'LaplacianVar', 'EdgeDensity',
    'SatVariance', 'HueVariance', 'HighFreqEnergy'
]

X = train_df[features_to_use].copy()
X_test = test_df[features_to_use].copy()
y = train_df['label'].astype(int)

X['SatEdgeRatio'] = X['SatVariance'] / (X['EdgeDensity'] + 1e-5)
X_test['SatEdgeRatio'] = X_test['SatVariance'] / (X_test['EdgeDensity'] + 1e-5)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=1500,
    learning_rate=0.005,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc',
    tree_method='hist'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=50
)

val_preds = model.predict_proba(X_val)[:,1]
val_auc = roc_auc_score(y_val, val_preds)
print(f"✅ Validation AUC after training: {val_auc:.5f}")

# Feature Importance
importances = model.feature_importances_
features = X_train.columns
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10,6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
plt.yticks(np.arange(len(sorted_idx)), [features[i] for i in sorted_idx])
plt.title('Feature Importance from XGBoost')
plt.grid(True)
plt.show()

# ================
# 6. Optimize SNR Threshold
# ================

def optimize_snr_threshold(df, metric='f1'):
    best_score = -1
    best_thresh = None
    thresholds = np.linspace(1.0, 5.0, 400)
    for thresh in thresholds:
        preds = (df['SNR'] < thresh).astype(int)
        if metric == 'f1':
            score = f1_score(df['label'], preds)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_thresh, best_score

best_snr_thresh, best_snr_score = optimize_snr_threshold(train_df, metric='f1')
print(f"🚀 Best SNR Threshold: {best_snr_thresh:.4f}")
print(f"✅ Best F1 Score achieved: {best_snr_score:.5f}")

snr_preds = (test_df['SNR'] < best_snr_thresh).astype(int)

# ================
# 7. Optimize Blend Weights + Final Threshold
# ================
val_snr_preds = (X_val['SNR'] < best_snr_thresh).astype(int)

best_weight = 0
best_blend_score = 0
blend_weights = np.linspace(0, 1, 101)

for w in blend_weights:
    blended_val_preds = (w * val_preds) + ((1 - w) * val_snr_preds)
    blended_val_labels = (blended_val_preds > 0.5).astype(int)
    score = f1_score(y_val, blended_val_labels)
    if score > best_blend_score:
        best_blend_score = score
        best_weight = w

print(f"✅ Best Blend: XGB {best_weight:.2f}, SNR {1-best_weight:.2f}")

# Fine-tune final threshold
def find_best_final_threshold(preds, y_true, metric='f1'):
    best_score = -1
    best_thresh = 0.5
    thresholds = np.linspace(0.1, 0.9, 400)
    for thresh in thresholds:
        preds_binary = (preds > thresh).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, preds_binary)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_thresh, best_score

final_blended_val_preds = (best_weight * val_preds) + ((1-best_weight) * val_snr_preds)

best_final_thresh, best_final_score = find_best_final_threshold(final_blended_val_preds, y_val)
print(f"🚀 Best Final Threshold: {best_final_thresh:.4f}")
print(f"✅ Best Fine-tuned F1: {best_final_score:.5f}")

# ================
# 8. Create Final Submission
# ================

print("⚡ Creating Final Submission...")

# Predict on Test Set
test_preds = model.predict_proba(X_test)[:,1]
blended_test_preds = (best_weight * test_preds) + ((1 - best_weight) * snr_preds)

final_labels = (blended_test_preds > best_final_thresh).astype(int)

submission = pd.DataFrame({
    'id': test_df['id'],
    'label': final_labels
})

submission['id'] = submission['id'].apply(lambda x: f'test_data_v2/{x}')
submission.to_csv('/kaggle/working/final_submission_finetuned.csv', index=False)

print("🎯 Final finetuned Submission File Created Successfully!")
