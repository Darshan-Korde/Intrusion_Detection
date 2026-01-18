import pandas as pd
import numpy as np

# ======================================================
# 1️⃣ SIGNATURE RULE DEFINITIONS
# ======================================================
SIGNATURE_RULES = [

    # FLOOD ATTACKS
    {"name": "TCP_FLOOD", "severity": "High", "confidence": 0.9,
     "match": lambda r: r.get("Protocol") == "TCP"
     and r.get("Flow Packets/s", 0) > 1500
     and r.get("SYN Flag Count", 0) > 5},
    {"name": "UDP_FLOOD", "severity": "High", "confidence": 0.9,
     "match": lambda r: r.get("Protocol") == "UDP"
     and r.get("Flow Packets/s", 0) > 2000},
    {"name": "ICMP_FLOOD", "severity": "High", "confidence": 0.95,
     "match": lambda r: r.get("Protocol") == "ICMP"
     and r.get("Flow Packets/s", 0) > 1000},

    # DoS / DDoS
    {"name": "DOS_DDOS", "severity": "High", "confidence": 0.85,
     "match": lambda r: r.get("Flow Packets/s", 0) > 1200
     and r.get("Flow Bytes/s", 0) > 1e6},
    {"name": "DOS_HULK", "severity": "High", "confidence": 0.88,
     "match": lambda r: r.get("Total Fwd Packets", 0) > 50
     and r.get("Flow Duration", 0) < 2e6},

    # FTP / Brute Force
    {"name": "FTP_PATATOR", "severity": "Medium", "confidence": 0.8,
     "match": lambda r: r.get("Dst Port") == 21
     and r.get("Total Fwd Packets", 0) > 30},
    {"name": "BRUTE_FORCE", "severity": "Medium", "confidence": 0.75,
     "match": lambda r: r.get("Dst Port") in [22, 23, 3389]
     and r.get("SYN Flag Count", 0) > 4},

    # Web attacks
    {"name": "SQL_INJECTION", "severity": "High", "confidence": 0.85,
     "match": lambda r: r.get("Dst Port") in [80, 443]
     and r.get("Packet Length Mean", 0) > 800
     and r.get("Flow Packets/s", 0) < 50},
    {"name": "XSS", "severity": "Medium", "confidence": 0.75,
     "match": lambda r: r.get("Dst Port") in [80, 443]
     and r.get("Packet Length Mean", 0) > 600},

    # Infiltration
    {"name": "INFILTRATION", "severity": "High", "confidence": 0.9,
     "match": lambda r: r.get("Idle Mean", 0) > 1e6
     and r.get("Flow Packets/s", 0) < 10},

    # Heartbleed
    {"name": "HEARTBLEED", "severity": "Critical", "confidence": 0.95,
     "match": lambda r: r.get("Dst Port") == 443
     and r.get("Packet Length Mean", 0) > 1200
     and r.get("Flow Packets/s", 0) < 5},
]

# ======================================================
# 2️⃣ SIGNATURE MATCH FUNCTION
# ======================================================
def run_signature_ids(row):
    for rule in SIGNATURE_RULES:
        try:
            if rule["match"](row):
                return rule
        except Exception:
            continue
    return None

# ======================================================
# 3️⃣ SIGNATURE IDS PIPELINE
# ======================================================
def signature_pipeline(df):
    signature_alerts = []
    anomaly_input = []

    for _, row in df.iterrows():
        sig = run_signature_ids(row)
        if sig:
            signature_alerts.append({
                "Timestamp": row.get("Timestamp"),
                "Src IP": row.get("Src IP"),
                "Dst IP": row.get("Dst IP"),
                "Attack": sig["name"],
                "Confidence": sig["confidence"],
                "Severity": sig["severity"]
            })
        else:
            anomaly_input.append(row)

    return pd.DataFrame(signature_alerts), pd.DataFrame(anomaly_input)

# ======================================================
# 4️⃣ MAIN EXECUTION
# ======================================================
if __name__ == "__main__":

    # Load JSON / NDJSON flows
    df_flows = pd.read_json("D:/S_ids/scapy_flows.json", lines=True)

    # Normalize column names
    df_flows.rename(columns={
        "timestamp": "Timestamp",
        "src_ip": "Src IP",
        "dst_ip": "Dst IP"
    }, inplace=True)

    # Clean data
    df_flows.replace([np.inf, -np.inf], 0, inplace=True)
    df_flows.fillna(0, inplace=True)

    # ---------------- Run Signature IDS ----------------
    signature_alerts, anomaly_input = signature_pipeline(df_flows)

    print("\n=== SIGNATURE ALERTS ===")
    print(signature_alerts.head())

    signature_alerts.to_csv("signature_alerts.csv", index=False)

    # ---------------- Run Anomaly IDS for remaining flows ----------------
    if not anomaly_input.empty:
        # Import your anomaly model from 'gate' folder
        import sys
        sys.path.append("D:/S_ids/gate")  # path to anomaly.py folder

        from anomaly import predict_dataframe

        # anomaly_input is a DataFrame
        anomaly_results = predict_dataframe(anomaly_input)

        print("\n=== ANOMALY ALERTS ===")
        print(anomaly_results.head())

        anomaly_results.to_csv("anomaly_alerts.csv", index=False)
