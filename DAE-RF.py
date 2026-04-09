import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import gc

# 1. Setup & Data Loading
df = pd.read_csv(r'D:\ids\testing\cic_clean_under.csv')
df = df.replace([np.inf, -np.inf], np.nan).dropna()

top_45_features = ['Init Fwd Win Byts', 'Fwd Seg Size Min', 'Flow IAT Mean', 'Fwd IAT Tot', 'Fwd Pkts/s', 'Fwd IAT Max', 'Flow Pkts/s', 'Flow Duration', 'Fwd IAT Min', 'Flow IAT Max', 'Flow IAT Min', 'Fwd Header Len', 'Bwd Pkts/s', 'Bwd Pkt Len Mean', 'Pkt Len Mean', 'Pkt Size Avg', 'Bwd Seg Size Avg', 'Flow Byts/s', 'Tot Fwd Pkts', 'Bwd Header Len', 'Fwd Seg Size Avg', 'ECE Flag Cnt', 'Fwd Pkt Len Mean', 'Pkt Len Var', 'TotLen Bwd Pkts', 'Init Bwd Win Byts', 'Pkt Len Std', 'RST Flag Cnt', 'Bwd IAT Max', 'Bwd Pkt Len Max', 'Subflow Bwd Byts', 'Flow IAT Std', 'Fwd Pkt Len Std', 'Fwd Pkt Len Max', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Min', 'Active Mean', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Max', 'Idle Min']
available_cols = [c for c in top_45_features if c in df.columns]

# Split for DAE training (100% Benign)
df_benign = df[df['Label'] == 'Benign'].copy()
scaler = MinMaxScaler(feature_range=(0, 1))
X_benign_scaled = scaler.fit_transform(df_benign[available_cols])

# 2. Define Keras DAE (Matching your Optimized Architecture)
input_dim = len(available_cols)
input_layer = layers.Input(shape=(input_dim,))

# Encoder: Input -> 64 -> 32 (Latent)
enc = layers.GaussianNoise(0.05)(input_layer) # Equivalent to your noise addition
enc = layers.Dense(64)(enc)
enc = layers.BatchNormalization()(enc)
enc = layers.Activation('relu')(enc)
latent_layer = layers.Dense(32, activation='relu', name='bottleneck')(enc)

# Decoder: 32 -> 64 -> Input_Dim
dec = layers.Dense(64)(latent_layer)
dec = layers.BatchNormalization()(dec)
dec = layers.Activation('relu')(dec)
output_layer = layers.Dense(input_dim, activation='sigmoid')(dec) # Sigmoid for MinMax match

dae = Model(inputs=input_layer, outputs=output_layer)
dae.compile(optimizer='adam', loss='mse')

# 3. Train DAE
print("Step 1: Training Keras DAE on Benign data...")
dae.fit(X_benign_scaled, X_benign_scaled,
        epochs=100,
        batch_size=256,
        shuffle=True,
        verbose=1)

# 4. Feature Extraction: Latent Space + Reconstruction Error
# Create encoder-only model to get latent features
encoder_model = Model(inputs=dae.input, outputs=dae.get_layer('bottleneck').output)

X_all_scaled = scaler.transform(df[available_cols])
recons = dae.predict(X_all_scaled)
latent_features = encoder_model.predict(X_all_scaled)

# Calculate Reconstruction Error (MSE per sample)
recon_error = np.mean(np.power(X_all_scaled - recons, 2), axis=1).reshape(-1, 1)

# Combine: Original (45) + Latent (32) + Recon Error (1)
X_hybrid = np.hstack([X_all_scaled, latent_features, recon_error])

# 5. Optimized Random Forest
y = (df['Label'] != 'Benign').astype(int)
X_train, X_test, y_train, y_test = train_test_split(X_hybrid, y, test_size=0.3, stratify=y, random_state=42)

print("Step 2: Training Random Forest on Hybrid Features...")
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    class_weight='balanced_subsample',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# 6. Evaluation
y_pred = rf.predict(X_test)
print("\n" + "="*40)
print("OPTIMIZED HYBRID DAE (KERAS) + RF RESULTS")
print("-" * 40)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print("="*40)

# --- 7. SAVE MODELS ---
save_path = 'D:/ids/hybrid_DAE_RF/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

print(f"\nSaving model components to {save_path}...")

# Save Keras DAE
dae.save(os.path.join(save_path, 'keras_dae_full.h5'))
# Save Encoder only (for prediction speed)
encoder_model.save(os.path.join(save_path, 'keras_encoder_only.h5'))

# Save Scaler and RF
joblib.dump(scaler, os.path.join(save_path, 'minmax_scaler.pkl'))
joblib.dump(rf, os.path.join(save_path, 'random_forest_model.pkl'))
joblib.dump(available_cols, os.path.join(save_path, 'feature_names.pkl'))

print("All components saved successfully.")

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlGnBu')
plt.show()