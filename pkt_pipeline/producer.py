import subprocess
import os
import time
from confluent_kafka import Producer

# --- CONFIGURATION ---
# Use 127.0.0.1, but the 'hosts' file trick above is what solves the "No such host" error
KAFKA_BROKER = "127.0.0.1:9092" 
TOPIC_NAME = "pcap-notifications"
STORAGE_DIR = r"D:\ids\ids_storage" 
WIFI_INTERFACE_ID = "5" 

# Ensure directory exists
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# Producer configuration with added timeout and logging settings
producer_conf = {
    'bootstrap.servers': KAFKA_BROKER,
    'client.id': 'shield-ids-producer',
    'socket.timeout.ms': 10000,
    'reconnect.backoff.max.ms': 5000
}

try:
    producer = Producer(producer_conf)
except Exception as e:
    print(f"[!] Could not initialize Producer: {e}")

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result. """
    if err is not None:
        print(f"   [X] Kafka Delivery Failed: {err}")
    else:
        print(f"   [√] Kafka Notified: {msg.value().decode('utf-8')}")

def run_capture():
    print("-" * 50)
    print(f"[*] SHIELD-IDS PRODUCER: LIVE MODE")
    print(f"[*] Interface: {WIFI_INTERFACE_ID} | Target: {KAFKA_BROKER}")
    print("-" * 50)

    while True:
        timestamp = int(time.time())
        filename = f"capture_{timestamp}.pcap"
        win_full_path = os.path.normpath(os.path.join(STORAGE_DIR, filename))

        print(f"\n[!] CAPTURING: {filename}")
        
        try:
            # -a packets:150 (Capture 150 packets)
            # -a duration:10 (Or stop after 10 seconds)
            process = subprocess.Popen([
                "tshark", 
                "-i", WIFI_INTERFACE_ID, 
                "-n", 
                "-a", "packets:150",   
                "-a", "duration:10",  
                "-w", win_full_path,
                "-S", "-l"  
            ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) # Suppress tshark noise if preferred

            # Wait for capture to complete
            process.wait()

            # Verify file exists and has actual data (PCAP headers are ~24 bytes)
            if os.path.exists(win_full_path) and os.path.getsize(win_full_path) > 100:
                size_kb = round(os.path.getsize(win_full_path) / 1024, 2)
                print(f"\n[+] Capture Finished! ({size_kb} KB)")
                
                # Produce to Kafka
                producer.produce(
                    TOPIC_NAME, 
                    value=win_full_path.encode('utf-8'), 
                    callback=delivery_report
                )
                
                # Flush tells the producer to send all buffered messages immediately
                producer.flush() 
            else:
                print(f"\n[.] Cycle ended: No significant traffic captured.")
                if os.path.exists(win_full_path): 
                    os.remove(win_full_path)

        except Exception as e:
            print(f"   [CRITICAL ERROR] {e}")
            time.sleep(5)

if __name__ == "__main__":
    try:
        run_capture()
    except KeyboardInterrupt:
        print("\n[*] Shutting down gracefully...")
        producer.flush(3) # Wait up to 3s to clear buffers