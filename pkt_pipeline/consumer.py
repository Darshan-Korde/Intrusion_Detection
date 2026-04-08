import pandas as pd
import numpy as np
from scapy.all import PcapReader, IP, IPv6, TCP, UDP
import os
from confluent_kafka import Consumer
import time

# --- CONFIGURATION ---
KAFKA_BROKER = "127.0.0.1:9092" 
TOPIC_NAME = "pcap-notifications"
IDLE_THRESHOLD = 5000000.0  # 5 seconds in microseconds

def get_stats(data):
    if not data or len(data) == 0:
        return 0, 0, 0, 0, 0
    return np.mean(data), np.std(data), np.max(data), np.min(data), np.var(data)

def extract_flow_features(pcap_path):
    print(f"\n[STEP 2/3] Extracting features from: {os.path.basename(pcap_path)}...")
    flows = {}

    try:
        with PcapReader(pcap_path) as pcap_iterable:
            for pkt in pcap_iterable:
                if not (pkt.haslayer(IP) or pkt.haslayer(IPv6)):
                    continue

                proto = pkt[IP].proto if pkt.haslayer(IP) else pkt[IPv6].nh
                src_ip = pkt[IP].src if pkt.haslayer(IP) else pkt[IPv6].src
                dst_ip = pkt[IP].dst if pkt.haslayer(IP) else pkt[IPv6].dst

                sport, dport, h_len, win_size = 0, 0, 0, 0
                tcp_flags = None

                if pkt.haslayer(TCP):
                    sport, dport = pkt[TCP].sport, pkt[TCP].dport
                    tcp_flags = pkt[TCP].flags
                    h_len = len(pkt[TCP])
                    win_size = pkt[TCP].window
                elif pkt.haslayer(UDP):
                    sport, dport = pkt[UDP].sport, pkt[UDP].dport
                    h_len = 8
                else:
                    continue

                flow_id = tuple(sorted((src_ip, dst_ip))) + tuple(sorted((sport, dport))) + (proto,)
                direction = "fwd" if src_ip < dst_ip else "bwd"

                if flow_id not in flows:
                    flows[flow_id] = {
                        'src_ip': src_ip, 'dst_ip': dst_ip, 'dport': dport, 'proto': proto,
                        'fwd': [], 'bwd': [], 'all_times': [], 'all_lens': [],
                        'init_win_fwd': 0, 'init_win_bwd': 0, 'fwd_seg_min': 0
                    }
                
                if pkt.haslayer(TCP):
                    if direction == "fwd" and flows[flow_id]['init_win_fwd'] == 0:
                        flows[flow_id]['init_win_fwd'] = win_size
                        flows[flow_id]['fwd_seg_min'] = h_len
                    elif direction == "bwd" and flows[flow_id]['init_win_bwd'] == 0:
                        flows[flow_id]['init_win_bwd'] = win_size

                pkt_info = {
                    'len': len(pkt),
                    'time': float(pkt.time) * 1000000, 
                    'h_len': h_len,
                    'flags': tcp_flags,
                    'is_tcp': pkt.haslayer(TCP)
                }

                flows[flow_id][direction].append(pkt_info)
                flows[flow_id]['all_times'].append(pkt_info['time'])
                flows[flow_id]['all_lens'].append(pkt_info['len'])

    except Exception as e:
        print(f"  [ERROR] Problem reading packets: {e}")
        return None

    csv_data = []
    for fid, data in flows.items():
        f_pkts, b_pkts = data['fwd'], data['bwd']
        all_times = sorted(data['all_times'])
        all_lens = data['all_lens']
        
        # 1. Time Calculations
        duration = all_times[-1] - all_times[0] if len(all_times) > 1 else 0.0
        dur_sec = max(duration / 1000000.0, 0.000001)

        gaps = np.diff(all_times) if len(all_times) > 1 else [0]
        f_gaps = np.diff([p['time'] for p in f_pkts]) if len(f_pkts) > 1 else [0]
        b_gaps = np.diff([p['time'] for p in b_pkts]) if len(b_pkts) > 1 else [0]

        # 2. Stats
        f_lens = [p['len'] for p in f_pkts]
        b_lens = [p['len'] for p in b_pkts]
        f_mean, f_std, f_max, f_min, _ = get_stats(f_lens)
        b_mean, b_std, b_max, b_min, _ = get_stats(b_lens)
        a_mean, a_std, a_max, a_min, a_var = get_stats(all_lens)

        def count_flags(pkts, flag_char):
            return sum(1 for p in pkts if p['is_tcp'] and flag_char in str(p['flags']))

        # 3. Active/Idle Logic
        idles = [g for g in gaps if g > IDLE_THRESHOLD]
        active_times = [g for g in gaps if g <= IDLE_THRESHOLD]
        num_subflows = len(idles) + 1

        row = {
            'Init Fwd Win Byts': data['init_win_fwd'],
            'Fwd Seg Size Min': data['fwd_seg_min'],
            'Flow IAT Mean': np.mean(gaps),
            'Fwd IAT Tot': sum(f_gaps),
            'Fwd Pkts/s': len(f_pkts) / dur_sec,
            'Fwd IAT Max': np.max(f_gaps),
            'Flow Pkts/s': len(all_lens) / dur_sec,
            'Flow Duration': duration,
            'Fwd IAT Min': np.min(f_gaps),
            'Flow IAT Max': np.max(gaps),
            'Flow IAT Min': np.min(gaps),
            'Fwd Header Len': sum(p['h_len'] for p in f_pkts),
            'Bwd Pkts/s': len(b_pkts) / dur_sec,
            'Bwd Pkt Len Mean': b_mean,
            'Pkt Len Mean': a_mean,
            'Pkt Size Avg': np.mean(all_lens) if all_lens else 0,
            'Bwd Seg Size Avg': b_mean,
            'Flow Byts/s': sum(all_lens) / dur_sec,
            'Tot Fwd Pkts': len(f_pkts),
            'Bwd Header Len': sum(p['h_len'] for p in b_pkts),
            'Fwd Seg Size Avg': f_mean,
            'ECE Flag Cnt': count_flags(f_pkts + b_pkts, 'E'),
            'Fwd Pkt Len Mean': f_mean,
            'Pkt Len Var': a_var,
            'TotLen Bwd Pkts': sum(b_lens),
            'Init Bwd Win Byts': data['init_win_bwd'],
            'Pkt Len Std': a_std,
            'RST Flag Cnt': count_flags(f_pkts + b_pkts, 'R'),
            'Bwd IAT Max': np.max(b_gaps),
            'Bwd Pkt Len Max': b_max,
            'Subflow Bwd Byts': sum(b_lens) / num_subflows,
            'Flow IAT Std': np.std(gaps),
            'Fwd Pkt Len Std': f_std,
            'Fwd Pkt Len Max': f_max,
            'Bwd IAT Tot': sum(b_gaps),
            'Bwd IAT Mean': np.mean(b_gaps),
            'Bwd IAT Min': np.min(b_gaps),
            'Active Mean': np.mean(active_times) if active_times else 0,
            'Active Max': np.max(active_times) if active_times else 0,
            'Active Min': np.min(active_times) if active_times else 0,
            'Idle Mean': np.mean(idles) if idles else 0,
            'Idle Max': np.max(idles) if idles else 0,
            'Idle Min': np.min(idles) if idles else 0,
            # Engineered Features
            'Pkt_Time_Density': duration / (len(f_pkts) + 1),
            'Payload_Efficiency': a_mean / (sum(p['h_len'] for p in f_pkts) + 1)
        }
        csv_data.append(row)

    # STRICT SEQUENCE MATCHING
    ordered_columns = [
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
        'Idle Mean', 'Idle Max', 'Idle Min', 'Pkt_Time_Density', 'Payload_Efficiency'
    ]

    final_df = pd.DataFrame(csv_data)
    if not final_df.empty:
        return final_df[ordered_columns]
    return final_df

# --- CONSUMER LOOP ---
conf = {
    'bootstrap.servers': KAFKA_BROKER,
    'group.id': 'windows-ids-group',
    'auto.offset.reset': 'latest'
}

consumer = Consumer(conf)
consumer.subscribe([TOPIC_NAME])

print("-" * 50)
print(f"[*] Pipeline Active. Waiting for Redpanda notifications...")
print("-" * 50)

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None: continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue

        pcap_path = msg.value().decode('utf-8')
        print(f"\n[STEP 1/3] Notification received for: {pcap_path}")

        if not os.path.exists(pcap_path):
            print(f"  [!] Skipping: File not found yet.")
            continue

        df = extract_flow_features(pcap_path)
        
        if df is not None and not df.empty:
            csv_path = pcap_path.replace(".pcapng", ".csv").replace(".pcap", ".csv")
            df.to_csv(csv_path, index=False)
            print(f"[STEP 3/3] SUCCESS: Created CSV with {len(df.columns)} features.")
            print("-" * 30)
        else:
            print("  [!] No valid network flows found.")

except KeyboardInterrupt:
    print("\n[*] Stopping pipeline...")
finally:
    consumer.close()