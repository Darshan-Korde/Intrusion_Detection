import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

# ======================================================
# 1. LOAD CICIDS2017 FILES
# ======================================================
monday = pd.read_csv("archive/Monday-WorkingHours.pcap_ISCX.csv")
thursday = pd.read_csv("archive/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")

# Fix column spacing
monday.columns = monday.columns.str.strip()
thursday.columns = thursday.columns.str.strip()

# ======================================================
# 2. SAMPLE NORMAL TRAFFIC (50%)
# ======================================================
monday_50 = monday.sample(frac=0.5, random_state=42)

# ======================================================
# 3. FILTER INFILTRATION ONLY
# ======================================================
thursday = thursday[thursday['Label'].str.contains("Infiltration", case=False)]

# ======================================================
# 4. COMBINE DATASETS
# ======================================================
df = pd.concat([monday_50, thursday], ignore_index=True)

# ======================================================
# 5. FIX LABELS
# ======================================================
df['Label'] = df['Label'].astype(str).str.strip()

df['Label'] = df['Label'].replace({
    'BENIGN': 0,
    'Benign': 0,
    'Infiltration': 1
})

df['Label'] = df['Label'].astype(int)

print("Label distribution:")
print(df['Label'].value_counts())

# ======================================================
# 6. TOP-10 INFILTRATION FEATURES
# ======================================================
INFILTRATION_TOP_10 = [
    'Flow Duration',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Packet Length Mean',
    'Packet Length Std',
    'Flow Bytes/s',
    'Active Mean',
    'Idle Mean'
]

features = [f for f in INFILTRATION_TOP_10 if f in df.columns]

if len(features) < 6:
    raise ValueError("Too few infiltration features found")

print("Using features:", features)

X = df[features]
y = df['Label']

# ======================================================
# 7. DATA CLEANING
# ======================================================
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# ======================================================
# 8. TRAIN LIGHTGBM (CLASS IMBALANCE AWARE)
# ======================================================
model = lgb.LGBMClassifier(
    objective='binary',
    class_weight={0: 1, 1: 500},  # critical for infiltration
    boosting_type='gbdt',
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=12,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X, y)

print("✅ Infiltration model training completed")

# ======================================================
# 9. SAVE MODEL & FEATURES
# ======================================================
joblib.dump(model, "infiltration_lightgbm_model.pkl")
joblib.dump(features, "infiltration_features.pkl")

print("💾 Model saved: infiltration_lightgbm_model.pkl")
print("💾 Features saved: infiltration_features.pkl")
