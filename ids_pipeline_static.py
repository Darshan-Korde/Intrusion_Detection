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
TEST_FILE_PATH = r'D:\ids\datasets\cic_clean_under.csv' 
MODEL_DIR = r'D:\ids\hybrid_DAE-RF'
MASTER_LOG_PATH = r'D:\ids\threat_history_log.csv'

ALL_INCOMING_FEATURES = [
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
    'Idle Mean', 'Idle Max', 'Idle Min', 
    'Pkt_Time_Density', 'Payload_Efficiency' 
]

MODEL_TRAINED_FEATURES = ALL_INCOMING_FEATURES[:43]

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

# ==========================================
# 3. STATIC FILE ANALYSIS ENGINE
# ==========================================
st.title("🛡️ Shield-IDS: Static Analysis Mode")

lgbm, scaler, dae_full, dae_encoder, rf_model = load_assets()

# Sidebar controls for static simulation
st.sidebar.header("Simulation Settings")
batch_size = st.sidebar.slider("Batch Size (Rows)", 1, 100000, 1000)
run_simulation = st.sidebar.button("Run Analysis")

if os.path.exists(TEST_FILE_PATH):
    df_full = pd.read_csv(TEST_FILE_PATH)
    st.sidebar.success(f"Loaded: {os.path.basename(TEST_FILE_PATH)}")
    st.sidebar.info(f"Total Rows: {len(df_full):,}")
    
    if run_simulation:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # We iterate through the static file in batches
        for i in range(0, len(df_full), batch_size):
            df_raw = df_full.iloc[i : i + batch_size].copy()
            
            # Step 1: Pre-process
            X_raw = pre_process_data(df_raw)
            
            # Step 2: DAE & Scaler
            X_scaled = scaler.transform(X_raw)
            latent = dae_encoder.predict(X_scaled, verbose=0)
            recon = dae_full.predict(X_scaled, verbose=0)
            mse = np.mean(np.power(X_scaled - recon, 2), axis=1).reshape(-1, 1)
            
            # Step 3: Hybrid RF Prediction
            X_hybrid = np.hstack([X_scaled, latent, mse])
            rf_anomalies = rf_model.predict(X_hybrid)
            
            # Step 4: Multi-Class LGBM Prediction
            lgbm_probs = lgbm.predict_proba(X_raw)
            lgbm_preds = np.argmax(lgbm_probs, axis=1)
            lgbm_conf = np.max(lgbm_probs, axis=1)
            
            # Step 5: Verification Logic
            verified_attacks = ((lgbm_preds > 0) & (lgbm_conf > 0.90) & (rf_anomalies == 1)).astype(int)
            
            results_df = pd.DataFrame({
                'Timestamp': time.strftime('%H:%M:%S'),
                'Source_IP': df_raw.get('src_ip', 'Internal'),
                'Label': [LABEL_MAP[p] if v == 1 else "Benign" for v, p in zip(verified_attacks, lgbm_preds)],
                'Status': ["ATTACK" if v == 1 else "Benign" for v in verified_attacks]
            })

            # --- UPDATE UI FOR THIS BATCH ---
            st.subheader(f"Batch Analysis: Rows {i} to {min(i + batch_size, len(df_full))}")
            
            metric_cols = st.columns(3)
            metric_cols[0].metric("Source File", os.path.basename(TEST_FILE_PATH))
            metric_cols[1].metric("Rows Processed", f"{i + len(df_raw):,}")
            
            threat_count = int(verified_attacks.sum())
            if threat_count > 0:
                metric_cols[2].error(f"Threats Found: {threat_count}")
            else:
                metric_cols[2].success("Safe Batch")

            chart_cols = st.columns([1, 1])
            with chart_cols[0]:
                fig = px.pie(results_df, names='Status', color='Status', 
                             color_discrete_map={'Benign':'#00CC96', 'ATTACK':'#EF553B'}, 
                             hole=0.4, title="Batch Status Distribution")
                st.plotly_chart(fig, use_container_width=True, key=f"pie_{i}")
            
            with chart_cols[1]:
                st.markdown("**Batch Data Preview**")
                st.dataframe(results_df, use_container_width=True, height=300)

            # Log to master file
            results_df.to_csv(MASTER_LOG_PATH, mode='a', header=not os.path.exists(MASTER_LOG_PATH), index=False)
            
            # Update progress bar
            progress = (i + len(df_raw)) / len(df_full)
            progress_bar.progress(progress)
            status_text.text(f"Progress: {int(progress * 100)}%")

            # Horizontal separation after batch visualization
            st.divider() 
            
            time.sleep(0.2)
            
        st.balloons()
        st.success("Analysis Complete! All batches processed.")
    else:
        st.info("Click 'Run Analysis' in the sidebar to process the test file.")

else:
    st.error(f"File not found at {TEST_FILE_PATH}. Please check the path.")