import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
import gc
import joblib

# 1. Setup & Load
file_path = r'D:\ids\testing\cic_clean_under.csv'
top_45_features = ['Init Fwd Win Byts', 'Fwd Seg Size Min', 'Flow IAT Mean', 'Fwd IAT Tot', 'Fwd Pkts/s', 'Fwd IAT Max', 'Flow Pkts/s', 'Flow Duration', 'Fwd IAT Min', 'Flow IAT Max', 'Flow IAT Min', 'Fwd Header Len', 'Bwd Pkts/s', 'Bwd Pkt Len Mean', 'Pkt Len Mean', 'Pkt Size Avg', 'Bwd Seg Size Avg', 'Flow Byts/s', 'Tot Fwd Pkts', 'Bwd Header Len', 'Fwd Seg Size Avg', 'ECE Flag Cnt', 'Fwd Pkt Len Mean', 'Pkt Len Var', 'TotLen Bwd Pkts', 'Init Bwd Win Byts', 'Pkt Len Std', 'RST Flag Cnt', 'Bwd IAT Max', 'Bwd Pkt Len Max', 'Subflow Bwd Byts', 'Flow IAT Std', 'Fwd Pkt Len Std', 'Fwd Pkt Len Max', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Min', 'Active Mean', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Max', 'Idle Min']
df = pd.read_csv(file_path)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[[c for c in top_45_features if c in df.columns] + ['Label']]

# 2. 50% Benign Strategy (To maintain balance while saving RAM)
df_benign = df[df['Label'] == 'Benign'].sample(frac=1.0, random_state=42)
df_attacks = df[df['Label'] != 'Benign']
df = pd.concat([df_benign, df_attacks])

# 3. FEATURE ENGINEERING (The Accuracy Booster)
df['Pkt_Time_Density'] = df['Flow Duration'] / (df['Tot Fwd Pkts'] + 1)
df['Payload_Efficiency'] = df['Pkt Len Mean'] / (df['Fwd Header Len'] + 1)

# 4. Encoding & SMOTE
label_names = ['Benign'] + sorted([c for c in df['Label'].unique() if c != 'Benign'])
label_map = {name: i for i, name in enumerate(label_names)}
df['Label'] = df['Label'].map(label_map)

X = df.drop(columns=['Label'])
y = df['Label']
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42, stratify=y_res)
del df, X, y, X_res, y_res
gc.collect()

# 5. Paper-Spec LightGBM (XGBoost Equivalent)
print("Training Paper-Spec Model (CPU)...")
model = lgb.LGBMClassifier(
    n_estimators=300,        # Increased to allow lower learning rate to work
    max_depth=10,            # Proven better for Infiltration/DoS classes
    num_leaves=128,          # Essential for depth 10 to be effective
    learning_rate=0.05,      # Slower learning for better precision
    subsample=0.8,
    bagging_freq=2,          # Updated: Faster training than freq=1
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.1,           # Small increase to help with overfit at depth 10
    min_split_gain=0.01,     # Faster: avoids useless splits
    min_child_samples=30,    # Better generalization
    class_weight='balanced',
    num_class = len(label_names),
    n_jobs=-1,
    random_state=42,
    verbosity=-1
)

model.fit(X_train, y_train)

# 6. FULL PERFORMANCE EVALUATION
y_prob = model.predict_proba(X_test)
threshold = 0.5
y_pred = []

for prob in y_prob:
    # If the max probability of any ATTACK class (indices 1+) >= 0.3, pick highest prob class
    if np.max(prob[1:]) >= threshold:
        y_pred.append(np.argmax(prob))
    else:
        y_pred.append(0) # Default to Benign

y_pred = np.array(y_pred)


# Calculate metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')

print("\n" + "="*30)
print(f"OVERALL PERFORMANCE METRICS:")
print("-" * 30)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC ROC:   {auc:.4f}")
print("="*30)

# 7. FPR & CLASSIFICATION REPORT
cm = confusion_matrix(y_test, y_pred)
print(f"\n{'Class Name':<25} | {'FPR':<10}")
print("-" * 40)
for i, name in enumerate(label_names):
    fp = cm[:, i].sum() - cm[i, i]
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"{name:<25} | {fpr:.4f}")

# 8. Plot Result
fig, ax = plt.subplots(figsize=(14, 10))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=label_names, ax=ax, cmap='plasma', xticks_rotation=90)
plt.title("Confusion Matrix: Paper-Spec Model (Depth=6, Balanced Weights)")
plt.show()

joblib.dump(model, r'D:\ids\hybrid_DAE-RF\lgbm_paper_spec.pkl')