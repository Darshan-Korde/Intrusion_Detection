import streamlit as st
import pandas as pd
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(page_title="CICIDS 2018 Flow Simulator", layout="wide")

st.title("📊 CICIDS 2018 Flow Traffic Simulator")
st.markdown("Generates synthetic flow statistics based on the CICIDS 2018 dataset distributions.")

# --- Sidebar ---
st.sidebar.header("Configuration")
target_ip = st.sidebar.text_input("Target IP Address", value="192.168.1.5")
intensity = st.sidebar.select_slider("Traffic Intensity", options=["Low", "Medium", "High", "Insane"])

# Mapping intensity to packet count ranges
intensity_map = {"Low": (10, 50), "Medium": (50, 500), "High": (500, 5000), "Insane": (5000, 50000)}

# --- Fast Simulation Logic ---
def simulate_flow(attack_name, target):
    low, high = intensity_map[intensity]
    flow_count = np.random.randint(low, high)
    
    # Generate random statistical data mimicking CICIDS features
    data = {
        "Flow Duration": np.random.uniform(100, 50000, flow_count),
        "Tot Fwd Pkts": np.random.randint(1, 100, flow_count),
        "Tot Bwd Pkts": np.random.randint(0, 100, flow_count),
        "Flow Byts/s": np.random.uniform(1000, 1000000, flow_count),
        "Flow Pkts/s": np.random.uniform(10, 5000, flow_count),
        "Timestamp": [time.strftime('%H:%M:%S') for _ in range(flow_count)]
    }
    
    df = pd.DataFrame(data)
    
    # Display Results Instantly
    st.subheader(f"🔥 Attack: {attack_name}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Flows Sent", f"{flow_count}")
    col2.metric("Target IP", target)
    col3.metric("Avg Packet Rate", f"{df['Flow Pkts/s'].mean():.2f} p/s")
    
    with st.expander("View Simulated Flow Data (First 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)

# --- UI Grid for Attack Buttons ---
attacks = [
    ["DDoS Hulk", "Slowloris", "Infiltration"],
    ["HOIC", "LOIC", "UDP Flood"],
    ["Botnet", "FTP Patator", "SSH Patator"],
    ["XSS", "SQLi", "DDoS SlowHTTP"],
    ["DoS GoldenEye", "Port Scan", "Brute Force"]
]

# Create the button layout
for row in attacks:
    cols = st.columns(3)
    for i, attack_label in enumerate(row):
        if cols[i].button(attack_label, key=attack_label, use_container_width=True):
            simulate_flow(attack_label, target_ip)

st.success("UI loaded)