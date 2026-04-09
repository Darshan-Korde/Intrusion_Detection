import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras import layers, Model
import plotly.express as px
import time

# ==========================================
# 1. CONSTANTS & CONFIGURATION
# ==========================================
MODEL_DIR = r'D:\ids\hybrid_DAE-RF'
MASTER_LOG_PATH = r'D:\ids\threat_history_log.csv'

MODEL_TRAINED_FEATURES = [
    'Init Fwd Win Byts', 'Fwd Seg Size Min', 'Flow IAT Mean', 'Fwd IAT Tot', 
    'Fwd Pkts/s', 'Fwd IAT Max', 'Flow Pkts/s', 'Flow Duration', 
    'Fwd IAT Min', 'Flow IAT Max', 'Flow IAT Min', 'Fwd Header Len', 
    'Bwd Pkts/s', 'Bwd Pkt Len Mean', 'Pkt Len Mean', 'Pkt Size Avg', 
    'Bwd Seg Size Avg', 'Flow Byts/s', 'Tot Fwd Pkts', 'Bwd Header Len', 
    'Fwd Seg Size Avg', 'ECE Flag Cnt', 'Fwd Pkt Len Mean', 'Pkt Len Var', 
    'TotLen Bwd Pkts', 'Init Bwd Win Byts', 'Pkt Len Std', 'RST Flag Cnt', 
    'Bwd IAT Max', 'Bwd Pkt Len Max', 'Subflow Bwd Byts', 'Flow IAT Std', 
    'Fwd Pkt Len Std', 'Fwd Pkt Len Max', 'Bwd IAT Tot', 'Bwd IAT Mean', 
    'Bwd IAT Min', 'Active Mean', 'Active Max', 'Active Min', 
    'Idle Mean', 'Idle Max', 'Idle Min'
]

LABEL_MAP = {
    0: "Benign", 1: "Bot", 2: "Brute Force -Web", 3: "Brute Force -XSS",
    4: "DDOS attack-HOIC", 5: "DDOS attack-LOIC-UDP", 6: "DoS attacks-GoldenEye",
    7: "DoS attacks-Hulk", 8: "DoS attacks-SlowHTTPTest", 9: "DoS attacks-Slowloris",
    10: "FTP-BruteForce", 11: "Infilteration", 12: "SQL Injection", 13: "SSH-Bruteforce"
}

st.set_page_config(page_title="Shield-IDS Static Test", layout="wide")

# ==========================================
# 2. MODEL UTILITIES
# ==========================================
def build_dae_architecture(input_dim=43): 
    input_layer = layers.Input(shape=(input_dim,), name="input_layer")
    enc = layers.GaussianNoise(0.05)(input_layer)
    enc = layers.Dense(64)(enc)
    enc = layers.BatchNormalization()(enc)
    enc = layers.Activation('relu')(enc)
    latent_layer = layers.Dense(32, activation='relu', name='bottleneck')(enc)
    dec = layers.Dense(64)(latent_layer)
    dec = layers.BatchNormalization()(dec)
    dec = layers.Activation('relu')(dec)
    output_layer = layers.Dense(input_dim, activation='sigmoid')(dec)
    
    full_model = Model(inputs=input_layer, outputs=output_layer)
    encoder_only = Model(inputs=input_layer, outputs=full_model.get_layer('bottleneck').output)
    return full_model, encoder_only

@st.cache_resource
def load_assets():
    lgbm = joblib.load(os.path.join(MODEL_DIR, 'lgbm_v10_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'minmax_scaler.pkl'))
    rf_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
    
    dae_full, dae_encoder = build_dae_architecture(input_dim=43)
    dae_full.load_weights(os.path.join(MODEL_DIR, 'keras_dae_full.h5'))
    
    return lgbm, scaler, dae_full, dae_encoder, rf_model

def pre_process_data(df):
    processed = pd.DataFrame(index=df.index)
    for f in MODEL_TRAINED_FEATURES:
        if f in df.columns:
            processed[f] = pd.to_numeric(df[f], errors='coerce')
        else:
            processed[f] = 0.0
    return processed.replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')

# REAL-TIME LOGISTIC REGRESSION FUNCTION
def real_time_logic_verifier(lgbm_conf, rf_anomaly, dae_mse):
    """
    Simulates Logistic Regression verification logic.
    Weights are tuned to prioritize High Confidence LGBM + RF Anomaly detection.
    """
    # Weights for the decision features
    w1, w2, w3, bias = 4.5, 3.2, 2.1, -5.0
    
    # Linear combination
    z = (w1 * lgbm_conf) + (w2 * rf_anomaly) + (w3 * np.clip(dae_mse, 0, 1)) + bias
    
    # Sigmoid function
    prob = 1 / (1 + np.exp(-z))
    return (prob > 0.5).astype(int)

# ==========================================
# 3. UI & ANALYSIS ENGINE
# ==========================================
st.title("🛡️ Shield-IDS: Advanced Static Analysis")

lgbm, scaler, dae_full, dae_encoder, rf_model = load_assets()

# Sidebar: File Upload
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Network Traffic (CSV)", type=['csv'])

st.sidebar.header("Simulation Settings")
batch_size = st.sidebar.slider("Batch Size", 1000, 50000, 10000)
run_simulation = st.sidebar.button("Run Analysis")

if uploaded_file is not None:
    df_full = pd.read_csv(uploaded_file)
    st.sidebar.success(f"File: {uploaded_file.name}")
    
    if run_simulation:
        progress_bar = st.progress(0)
        
        for i in range(0, len(df_full), batch_size):
            df_raw = df_full.iloc[i : i + batch_size].copy()
            X_raw = pre_process_data(df_raw)
            
            # --- FEATURE EXTRACTION ---
            X_scaled = scaler.transform(X_raw)
            latent = dae_encoder.predict(X_scaled, verbose=0)
            recon = dae_full.predict(X_scaled, verbose=0)
            mse = np.mean(np.power(X_scaled - recon, 2), axis=1)
            
            # --- MODEL INFERENCE ---
            X_hybrid_rf = np.hstack([X_scaled, latent, mse.reshape(-1, 1)])
            rf_anomalies = rf_model.predict(X_hybrid_rf)
            
            lgbm_probs = lgbm.predict_proba(X_raw)
            lgbm_preds = np.argmax(lgbm_probs, axis=1)
            lgbm_max_conf = np.max(lgbm_probs, axis=1)
            
            # --- REAL-TIME LOGISTIC VERIFICATION ---
            # Verifies if the threat is real based on the combination of all models
            verified_attacks = real_time_logic_verifier(lgbm_max_conf, rf_anomalies, mse)
            
            # --- PREPARE RESULTS ---
            results_df = pd.DataFrame({
                'Timestamp': time.strftime('%H:%M:%S'),
                'Source_IP': df_raw.get('src_ip', 'Internal'),
                'Detected_Type': [LABEL_MAP[p] if v == 1 else "Benign" for v, p in zip(verified_attacks, lgbm_preds)],
                'Status': ["ATTACK" if v == 1 else "Benign" for v in verified_attacks]
            })

            # --- UI RENDERING ---
            st.subheader(f"Batch Analysis: {i} to {min(i + batch_size, len(df_full))}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Processing", uploaded_file.name)
            m2.metric("Rows Processed", f"{i + len(df_raw):,}")
            
            threat_count = int((results_df['Status'] == "ATTACK").sum())
            if threat_count > 0:
                m3.error(f"Verified Threats: {threat_count}")
            else:
                m3.success("Batch Clean")

            c1, c2 = st.columns([1, 1])
            with c1:
                fig = px.pie(results_df, names='Status', color='Status', 
                             color_discrete_map={'Benign':'#00CC96', 'ATTACK':'#EF553B'}, 
                             hole=0.4, title="Verification Results")
                st.plotly_chart(fig, use_container_width=True, key=f"pie_{i}")
            
            with c2:
                st.dataframe(results_df, use_container_width=True, height=300)

            results_df.to_csv(MASTER_LOG_PATH, mode='a', header=not os.path.exists(MASTER_LOG_PATH), index=False)
            progress_bar.progress((i + len(df_raw)) / len(df_full))
            st.divider() 
            
        st.balloons()
else:
    st.info("Please upload a CSV file to begin.")