import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ===============================
# 1. LOAD TRAIN DATA
# ===============================
train_df = pd.read_csv("cicids_2017_combined.csv", low_memory=False)

print("Train shape:", train_df.shape)

# ===============================
# 2. REMOVE LABEL COLUMN IF PRESENT
# ===============================
label_cols = [c for c in train_df.columns if c.strip().lower() == "label"]
train_df.drop(columns=label_cols, inplace=True, errors="ignore")

# ===============================
# 3. KEEP NUMERIC COLUMNS ONLY
# ===============================
train_df = train_df.select_dtypes(include=[np.number])

print("Numeric feature count:", train_df.shape[1])

# ===============================
# 4. CLEAN CICIDS DATA (CRITICAL)
# ===============================
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.fillna(0, inplace=True)

# Clip extreme values (prevents float overflow)
train_df = train_df.clip(lower=-1e6, upper=1e6)

print("Data cleaned successfully")

# ===============================
# 5. FEATURE SCALING
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df)

# ===============================
# 6. TRAIN ISOLATION FOREST
# ===============================
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

iso_forest.fit(X_train)

print("✅ Isolation Forest training completed")

# ===============================
# 7. SAVE MODEL, SCALER & FEATURES
# ===============================
joblib.dump(iso_forest, "isolation_forest_model.pkl")
joblib.dump(scaler, "isolation_forest_scaler.pkl")
joblib.dump(train_df.columns.tolist(), "isolation_forest_features.pkl")

print("💾 Model saved: isolation_forest_model.pkl")
print("💾 Scaler saved: isolation_forest_scaler.pkl")
print("💾 Features saved: isolation_forest_features.pkl")
