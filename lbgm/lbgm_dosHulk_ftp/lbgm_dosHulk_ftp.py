import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

# ======================================================
# 1. LOAD TUESDAY & WEDNESDAY FILES
# ======================================================
tuesday = pd.read_csv("archive/Tuesday-WorkingHours.pcap_ISCX.csv")
wednesday = pd.read_csv("archive/Wednesday-workingHours.pcap_ISCX.csv")

# Fix column spacing
tuesday.columns = tuesday.columns.str.strip()
wednesday.columns = wednesday.columns.str.strip()

# ======================================================
# 2. FILTER NORMAL + DoS ATTACKS
# ======================================================
dos_labels = ['DoS Hulk', 'FTP-Patator']
normal_label = ['BENIGN']

tuesday = tuesday[tuesday['Label'].isin(normal_label + dos_labels)].copy()
wednesday = wednesday[wednesday['Label'].isin(normal_label + dos_labels)].copy()

# Combine
df = pd.concat([tuesday, wednesday], ignore_index=True)

# Map labels to 0/1
df['Label'] = df['Label'].replace({
    'BENIGN': 0,
    'DoS Hulk': 1,
    'FTP-Patator': 1
}).astype(int)

print("Label distribution:")
print(df['Label'].value_counts())

# ======================================================
# 3. SELECT TOP-10 DOS FEATURES
# ======================================================
DOS_TOP_10_FEATURES = [
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

features = [f for f in DOS_TOP_10_FEATURES if f in df.columns]
print("Using features:", features)

X = df[features].copy()
y = df['Label'].copy()

# ======================================================
# 4. DATA CLEANING
# ======================================================
X.replace([np.inf, -np.inf], 0, inplace=True)
X.fillna(0, inplace=True)

# ======================================================
# 5. CLASS WEIGHT
# ======================================================
class_weight = {0: 1, 1: 5}

# ======================================================
# 6. TRAIN LIGHTGBM
# ======================================================
model = lgb.LGBMClassifier(
    objective='binary',
    class_weight=class_weight,
    boosting_type='gbdt',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X, y)

print("✅ DoS Hulk / FTP-Patator model training completed")

# ======================================================
# 7. SAVE MODEL, FEATURES & THRESHOLD
# ======================================================
joblib.dump(model, "dos_hulk_ftp_patator_lightgbm_model.pkl")
joblib.dump(features, "dos_hulk_ftp_patator_features.pkl")
joblib.dump(0.45, "dos_hulk_ftp_patator_threshold.pkl")

print("💾 Model saved: dos_hulk_ftp_patator_lightgbm_model.pkl")
print("💾 Features saved: dos_hulk_ftp_patator_features.pkl")
print("💾 Threshold saved: dos_hulk_ftp_patator_threshold.pkl")
