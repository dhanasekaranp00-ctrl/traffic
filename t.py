import serial
import time

try:
    # உங்க போர்ட் நேம்-ஐ மாத்திக்கோங்க
    s = serial.Serial('COM5', 9600, timeout=1) 
    print("Port Opened! Waiting for data...")
    while True:
        if s.in_waiting > 0:
            data = s.readline().decode('utf-8').strip()
            print(f"✅ Received from HW: {data}")
except Exception as e:
    print(f"❌ Error: {e}")