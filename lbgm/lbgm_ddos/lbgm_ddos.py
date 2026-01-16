import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

from sklearn.model_selection import train_test_split

# ======================================================
# 1. LOAD CICIDS2017 FILES
# ======================================================
monday = pd.read_csv("archive/Monday-WorkingHours.pcap_ISCX.csv")
friday = pd.read_csv("archive/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

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
    'DDoS': 1,
    'DoS': 1,
    '1': 1
})

df['Label'] = df['Label'].astype(int)

print("Label distribution:")
print(df['Label'].value_counts())

# ======================================================
# 5. TOP-10 DoS / DDoS FEATURES
# ======================================================
DOS_DDOS_TOP_10 = [
    'Flow Packets/s',
    'Flow Bytes/s',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Flow Duration',
    'Packet Length Mean',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'SYN Flag Count',
    'Active Mean'
]

available_features = [f for f in DOS_DDOS_TOP_10 if f in df.columns]

if len(available_features) < 5:
    raise ValueError("Too few DoS/DDoS features found.")

print("Using features:", available_features)

X = df[available_features]
y = df['Label']

# ======================================================
# 6. DATA CLEANING
# ======================================================
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# ======================================================
# 7. TRAIN SPLIT (NO EVALUATION)
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

print("✅ DoS/DDoS model training completed")

# ======================================================
# 9. SAVE MODEL & FEATURES
# ======================================================
joblib.dump(model, "dos_ddos_lightgbm_model.pkl")
joblib.dump(available_features, "dos_ddos_features.pkl")

print("💾 Model saved: dos_ddos_lightgbm_model.pkl")
print("💾 Features saved: dos_ddos_features.pkl")
