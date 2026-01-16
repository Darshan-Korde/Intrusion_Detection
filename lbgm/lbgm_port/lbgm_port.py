import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

from sklearn.model_selection import train_test_split

# ======================================================
# 1. LOAD CICIDS2017 FILES
# ======================================================
monday = pd.read_csv("archive/Monday-WorkingHours.pcap_ISCX.csv")
friday = pd.read_csv("archive/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")

# Fix column name spacing
monday.columns = monday.columns.str.strip()
friday.columns = friday.columns.str.strip()

# ======================================================
# 2. SAMPLE NORMAL TRAFFIC (50%)
# ======================================================
monday_50 = monday.sample(frac=0.5, random_state=42)

# ======================================================
# 3. COMBINE DATASETS
# ======================================================
df = pd.concat([monday_50, friday], ignore_index=True)

# ======================================================
# 4. FIX LABEL COLUMN
# ======================================================
df['Label'] = df['Label'].astype(str).str.strip()

df['Label'] = df['Label'].replace({
    'BENIGN': 0,
    'Benign': 0,
    '0': 0,
    'PortScan': 1,
    '1': 1
})

df['Label'] = df['Label'].astype(int)

print("Labels found:", df['Label'].value_counts())

# ======================================================
# 5. SELECT TOP-10 PORTSCAN FEATURES
# ======================================================
PORTSCAN_TOP_10 = [
    'SYN Flag Count',
    'Flow Packets/s',
    'Total Fwd Packets',
    'Flow Duration',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Packet Length Mean',
    'RST Flag Count',
    'Init_Win_bytes_forward',
    'Fwd Header Length'
]

available_features = [f for f in PORTSCAN_TOP_10 if f in df.columns]

if len(available_features) < 5:
    raise ValueError("Too few PortScan features found.")

print("Using features:", available_features)

X = df[available_features]
y = df['Label']

# ======================================================
# 6. DATA CLEANING
# ======================================================
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# ======================================================
# 7. TRAIN-TEST SPLIT (ONLY FOR TRAINING STABILITY)
# ======================================================
X_train, _, y_train, _ = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ======================================================
# 8. TRAIN LIGHTGBM MODEL
# ======================================================
model = lgb.LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

print("✅ Model training completed")

# ======================================================
# 9. SAVE MODEL & FEATURES
# ======================================================
joblib.dump(model, "portscan_lightgbm_model.pkl")
joblib.dump(available_features, "portscan_features.pkl")

print("💾 Model saved as: portscan_lightgbm_model.pkl")
print("💾 Features saved as: portscan_features.pkl")
