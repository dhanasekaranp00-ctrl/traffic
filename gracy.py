import cv2
import re
import csv
import time
import smtplib
import threading
import logging
import serial
import serial.tools.list_ports
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
# CONFIGURATION (Updated for Raspberry Pi)
# ──────────────────────────────────────────────────────────────
TESSERACT_PATH     = "C:/Program Files/Tesseract-OCR/tesseract.exe"   
EMAIL_SENDER       = "sekaran2405@gmail.com"
EMAIL_PASSWORD     = "wrrf awsj dlkm smrn"
CSV_DB_PATH        = "vehicle_data_template.csv"  
LOG_CSV_PATH       = "violation_log.csv"
VEHICLE_MODEL_PATH = "yolov8n.pt"
CUSTOM_MODEL_PATH  = "models/best.pt"
EMAIL_COOLDOWN_SECONDS = 120

# ──────────────────────────────────────────────────────────────
# UART / SERIAL CONFIGURATION
# ──────────────────────────────────────────────────────────────
SERIAL_PORT     = "COM5"      
SERIAL_BAUDRATE = 9600
SERIAL_ENABLED  = True                

# ──────────────────────────────────────────────────────────────
# SECONDARY CAMERA CONFIG
# ──────────────────────────────────────────────────────────────
SECONDARY_CAM_SOURCE = None
SECONDARY_WINDOW     = "OCR CAM - Show Number Plate"
PRIMARY_WINDOW       = "SMART TRAFFIC PRO - ESC to quit"

# ──────────────────────────────────────────────────────────────
# TESSERACT SETUP
# ──────────────────────────────────────────────────────────────
if TESSERACT_PATH and Path(TESSERACT_PATH).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ──────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# GLOBAL STATE
# ──────────────────────────────────────────────────────────────
email_sent_times              = {}
processed_plates_this_session = set()
serial_conn                   = None

# ──────────────────────────────────────────────────────────────
# SERIAL / UART SETUP & FUNCTIONS
# ──────────────────────────────────────────────────────────────
def init_serial():
    global serial_conn
    if not SERIAL_ENABLED:
        log.info("[UART] Serial disabled (SERIAL_ENABLED=False)")
        return

    try:
        serial_conn = serial.Serial(
            port=SERIAL_PORT,
            baudrate=SERIAL_BAUDRATE,
            timeout=1
        )
        time.sleep(2)   
        log.info(f"[UART] ✅ Serial connected: {SERIAL_PORT} @ {SERIAL_BAUDRATE} baud")
    except serial.SerialException as e:
        log.error(f"[UART] ❌ Cannot open {SERIAL_PORT}: {e}")
        log.warning("[UART] Running without serial — traffic signals won't be sent.")
        serial_conn = None

def check_hw_signal():
    global serial_conn
    if serial_conn is not None and serial_conn.in_waiting > 0:
        try:
            data = serial_conn.readline().decode('utf-8', errors='ignore').strip()
            return data
        except Exception as e:
            log.error(f"[UART] Read error: {e}")
    return None

def send_uart(value: str, label: str = ""):
    global serial_conn
    if serial_conn is None or not serial_conn.is_open:
        return
    try:
        serial_conn.write(f"{value}\n".encode('utf-8'))
        serial_conn.flush()
        log.info(f"[UART] ✅ Sent: {value}\\n | Reason: {label}")
    except serial.SerialException as e:
        log.error(f"[UART] Send error: {e}")

def close_serial():
    global serial_conn
    if serial_conn and serial_conn.is_open:
        serial_conn.close()
        log.info("[UART] Serial connection closed.")

# ══════════════════════════════════════════════════════════════
# OCR & IMAGE FUNCTIONS
# ══════════════════════════════════════════════════════════════
def clean_plate_text(raw):
    return re.sub(r"[^A-Z0-9]", "", raw.upper())

def validate_plate(text):
    return 8 <= len(text) <= 12

def extract_number_plate(frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]

    pad = 20
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    try:
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        config = "--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = pytesseract.image_to_string(thresh, config=config)

        plate = clean_plate_text(text)

        if validate_plate(plate):
            return plate

        return None

    except Exception as e:
        log.error(f"[OCR] Error: {e}")
        return None

# ══════════════════════════════════════════════════════════════
# DATABASE & LOGGING 
# ══════════════════════════════════════════════════════════════
def create_sample_csv():
    pd.DataFrame({
        "Number Plate": ["TN47AX1433", "TN47BW9654", "MH12EF9012"],
        "Owner Name":   ["ANU",  "DHANA",  "Rahul Mehta"],
        "Phone Number": ["2323543210",  "9384215035",  "7654321098"],
        "Email ID":     ["anugracyp@gmail.com", "sekaran2405@gmail.com", "rahul@example.com"],
    }).to_csv(CSV_DB_PATH, index=False)

def read_csv_data():
    if not Path(CSV_DB_PATH).exists(): create_sample_csv()
    try:
        df = pd.read_csv(CSV_DB_PATH, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        df["Number Plate"] = df["Number Plate"].str.upper().str.replace(r"\s+", "", regex=True)
        return df
    except Exception as e:
        log.error(f"[CSV] Failed: {e}")
        return pd.DataFrame()

def lookup_owner(plate, df):
    if df.empty: return None
    
    match = df[df["Number Plate"] == plate]
    if not match.empty:
        row = match.iloc[0]
        return {
            "plate": plate,
            "name":  row.get("Owner Name",   "Unknown"),
            "phone": row.get("Phone Number", "N/A"),
            "email": row.get("Email ID",     ""),
        }
        
    for idx, row in df.iterrows():
        db_plate = str(row["Number Plate"])
        if len(plate) >= 8 and len(db_plate) >= 8:
            match_count = sum(1 for a, b in zip(plate, db_plate) if a == b)
            if match_count >= len(db_plate) - 2:  
                return {
                    "plate": db_plate,  
                    "name":  row.get("Owner Name",   "Unknown"),
                    "phone": row.get("Phone Number", "N/A"),
                    "email": row.get("Email ID",     ""),
                }
                
    return None

def init_log_csv():
    if not Path(LOG_CSV_PATH).exists():
        with open(LOG_CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["Number Plate", "Owner Name", "Phone", "Email", "Date", "Time", "Violation Type"])

def log_violation(plate, owner, violation="Line Cross"):
    now = datetime.now()
    try:
        with open(LOG_CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                plate,
                owner["name"]  if owner else "Unknown",
                owner["phone"] if owner else "N/A",
                owner["email"] if owner else "N/A",
                now.strftime("%d-%m-%Y"),
                now.strftime("%H:%M:%S"),
                violation,
            ])
    except Exception as e:
        log.error(f"[Log] Failed: {e}")

# ══════════════════════════════════════════════════════════════
# EMAIL ALERT
# ══════════════════════════════════════════════════════════════
def _send_email_worker(to_email, owner_name, plate, violation):
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    subject   = "Traffic Violation Alert - Fine Issued"
    body = (
        f"Dear {owner_name},\n\n"
        "This is an automated notice from the Smart Traffic Monitoring System.\n\n"
        f"Your vehicle with number plate {plate} has been detected committing a traffic violation:\n\n"
        f"  Violation  : {violation}\n"
        f"  Date/Time  : {timestamp}\n"
        "  Location   : Traffic Camera Unit - Junction 1\n\n"
        "A fine has been issued 500 RS. Please pay at your nearest traffic authority office within 15 days.\n\n"
        f"  Number Plate : {plate}\n"
        f"  Owner        : {owner_name}\n\n"
        "Regards,\nSmart Traffic Monitoring System\n(Auto-generated — do not reply)\n"
    )
    msg = MIMEMultipart()
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, to_email, msg.as_string())
        log.info(f"[Email] ✅ Fine sent to {to_email}")
    except Exception as e:
        log.error(f"[Email] Failed: {e}")

def send_email(owner, violation="Line Crossing"):
    plate = owner["plate"]
    email = owner.get("email", "")
    if not email: return
    last_sent = email_sent_times.get(plate)
    if last_sent and datetime.now() - last_sent < timedelta(seconds=EMAIL_COOLDOWN_SECONDS): return
    email_sent_times[plate] = datetime.now()
    threading.Thread(target=_send_email_worker, args=(email, owner["name"], plate, violation), daemon=True).start()

# ══════════════════════════════════════════════════════════════
# DISPLAY & OCR PIPELINE HELPERS
# ══════════════════════════════════════════════════════════════
def run_ocr_pipeline(cap, df):

    log.info("LINE CROSS — OCR PIPELINE STARTED")
    log.info("⏳ Scanning for vehicle plate...")
    time.sleep(1) 

    plate = None
    last_candidate = None
    confirm_count = 0
    MAX_FRAMES = 150
    frame_num = 0

    while frame_num < MAX_FRAMES:

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            frame_num += 1
            continue

        frame = cv2.resize(frame, (1280, 720))
        h, w = frame.shape[:2]

        cx, cy = w // 2, h // 2
        bw, bh = 700, 220

        gx1 = max(0, cx - bw // 2)
        gy1 = max(0, cy - bh // 2)
        gx2 = min(w, cx + bw // 2)
        gy2 = min(h, cy + bh // 2)

        crop_debug = frame[gy1:gy2, gx1:gx2]
        cv2.imshow("OCR AREA", crop_debug)

        display = frame.copy()
        cv2.rectangle(display, (gx1, gy1), (gx2, gy2), (0,255,255), 2)
        cv2.imshow(SECONDARY_WINDOW, display)

        if cv2.waitKey(1) == 27:
            break

        if frame_num % 2 == 0:
            candidate = extract_number_plate(frame, (gx1, gy1, gx2, gy2))

            if candidate:
                print("Trying:", candidate)
                if candidate not in processed_plates_this_session:
                    
                    db_check = lookup_owner(candidate, df)
                    if db_check:
                        plate = db_check["plate"]
                        print("✅ FINAL PLATE ACCEPTED (DB MATCH):", plate)
                        cv2.putText(frame, plate, (cx-200, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 4)
                        cv2.imshow(SECONDARY_WINDOW, frame)
                        cv2.waitKey(1000)
                        break
                    
                    if candidate == last_candidate:
                        confirm_count += 1
                    else:
                        last_candidate = candidate
                        confirm_count = 1

                    if confirm_count >= 2:
                        plate = candidate
                        print("✅ FINAL PLATE ACCEPTED (AFTER 2 READS):", plate)
                        cv2.putText(frame, plate, (cx-200, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 4)
                        cv2.imshow(SECONDARY_WINDOW, frame)
                        cv2.waitKey(1000)
                        break

        frame_num += 1

    try:
        cv2.destroyWindow(SECONDARY_WINDOW)
        cv2.destroyWindow("OCR AREA")
    except:
        pass

    if not plate:
        log.warning("❌ Plate not detected")
        return None, None

    processed_plates_this_session.add(plate)

    owner = lookup_owner(plate, df)

    if not owner:
        owner = {"plate": plate, "name": "Unknown", "phone": "N/A", "email": ""}
    else:
        plate = owner["plate"]

    send_email(owner, "Line Crossing")
    log_violation(plate, owner)

    return plate, owner

def draw_info_panel(frame, car_count, bus_count, bike_count, total_count, status, color_status):
    cv2.rectangle(frame, (10, 10), (520, 310), (0, 0, 0), -1)
    cv2.putText(frame, f"CAR  : {car_count}",   (20, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),    2)
    cv2.putText(frame, f"BUS  : {bus_count}",   (20, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0),    2)
    cv2.putText(frame, f"BIKE : {bike_count}",  (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),  2)
    cv2.putText(frame, f"TOTAL: {total_count}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, status,                  (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status,   2)

# ══════════════════════════════════════════════════════════════
# MAIN LOOP (WITH 10 SECONDS TIMEOUT & CAMERA CLOSE LOGIC)
# ══════════════════════════════════════════════════════════════
def main():
    global serial_conn
    log.info("=== Smart Traffic System PRO — Starting ===")

    init_serial()
    vehicle_model = YOLO(VEHICLE_MODEL_PATH)
    custom_model  = YOLO(CUSTOM_MODEL_PATH)

    df = read_csv_data()
    log.info(f"[CSV Database] {len(df)} records loaded.")
    init_log_csv()

    cap = None
    camera_active = False
    
    line_counter      = 0
    ambulance_counter = 0
    line_cross_cooldown = 0
    
    detection_start_time = 0
    max_total_count = 0
    ambulance_seen = False

    log.info("System Ready. Waiting for HW signal (1 or 2) to activate camera...")

    while True:
        # State 1: Waiting for Hardware Signal
        if not camera_active:
            hw_signal = check_hw_signal()
            if hw_signal in ['1', '2']:
                log.info(f"[HW] Signal '{hw_signal}' received! Activating camera for 10 SECONDS...")
                
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    time.sleep(2) # Camera warmup
                
                camera_active = True
                detection_start_time = time.time()
                max_total_count = 0
                ambulance_seen = False
            else:
                time.sleep(0.1)
                continue

        # State 2: Camera Active & Processing Frames
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (1280, 720))

        car_count = bus_count = bike_count = total_count = 0
        line_detected = ambulance_detected = False
        line_box = None

        if line_cross_cooldown > 0:
            line_cross_cooldown -= 1

        # ── VEHICLE DETECTION ──
        results1 = vehicle_model(frame, conf=0.4, verbose=False)
        for r in results1:
            for box in r.boxes:
                cls   = int(box.cls[0])
                label = vehicle_model.names[cls]
                conf  = float(box.conf[0])  # Added confidence extraction
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label == "car":
                    car_count  += 1; total_count += 1; color = (0, 255, 0)
                elif label == "bus":
                    bus_count  += 1; total_count += 1; color = (255, 0, 0)
                elif label in ("motorbike", "bicycle"):
                    bike_count += 1; total_count += 1; color = (0, 255, 255)
                elif label == "truck":
                    total_count += 1;                  color = (0, 200, 200)
                else:
                    continue

                # Formatting label with percentage
                display_text = f"{label.upper()} {conf*100:.0f}%"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ── CUSTOM DETECTION ──
        results2 = custom_model(frame, conf=0.10, verbose=False)
        for r in results2:
            for box in r.boxes:
                cls   = int(box.cls[0])
                label = custom_model.names[cls]
                conf  = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area  = (x2 - x1) * (y2 - y1)

                if label == "ambulance" and conf > 0.9 and area > 5000:
                    ambulance_detected = True
                    ambulance_counter  = 15
                    display_text = f"AMBULANCE {conf*100:.0f}%" # Added percentage
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

                elif label == "line_cross" and conf > 0.25 and area > 3000 and line_cross_cooldown == 0:
                    line_detected = True
                    line_counter  = 10
                    line_box      = (x1, y1, x2, y2)
                    display_text = f"LINE CROSS {conf*100:.0f}%" # Added percentage
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        if line_counter > 0:      line_detected = True;      line_counter -= 1
        if ambulance_counter > 0: ambulance_detected = True; ambulance_counter -= 1

        # Tracker for the 10-second window
        max_total_count = max(max_total_count, total_count)
        if ambulance_detected:
            ambulance_seen = True

        # Display Status Logic (Visual Only)
        if ambulance_detected:
            status, color_status = "AMBULANCE - PRIORITY", (0, 255, 0)
        elif total_count >= 10:
            status, color_status = "HIGH TRAFFIC", (0, 0, 255)
        elif 5 <= total_count < 10:
            status, color_status = "MEDIUM TRAFFIC", (0, 165, 255)
        else:
            status, color_status = "LOW TRAFFIC", (0, 255, 255)

        # Draw 10 Second Timer
        elapsed_time = time.time() - detection_start_time
        remaining_time = max(0, 10 - int(elapsed_time))
        cv2.putText(frame, f"TIME LEFT: {remaining_time}s", (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # ── LINE CROSS PIPELINE ──
        if line_detected and line_box is not None and line_cross_cooldown == 0:
            line_cross_cooldown = 300
            plate, owner = run_ocr_pipeline(cap, df)
            line_box      = None
            line_detected = False
            line_counter  = 0
            continue

        draw_info_panel(frame, car_count, bus_count, bike_count, total_count, status, color_status)
        cv2.imshow(PRIMARY_WINDOW, frame)

        # State 3: 10 Seconds Completed
        if elapsed_time >= 10.0:
            log.info(f"⏱️ 10 Seconds Up! Max vehicles counted: {max_total_count}")
            
            # SEND UART DATA STRICTLY ONCE
            if ambulance_seen:
                send_uart('4', "AMBULANCE DETECTED (10s Window)")
            elif max_total_count >= 10:
                send_uart('3', f"HIGH TRAFFIC (Max: {max_total_count})")
            elif 5 <= max_total_count < 10:
                send_uart('2', f"MEDIUM TRAFFIC (Max: {max_total_count})")
            else:
                send_uart('1', f"LOW TRAFFIC (Max: {max_total_count})")

            log.info("🛑 Closing camera and waiting for next hardware signal...")
            
            # Close Camera
            if cap:
                cap.release()
                cap = None
            cv2.destroyAllWindows()
            
            # Flush Serial Buffer to prevent instant looping from old signals
            if serial_conn and serial_conn.in_waiting > 0:
                serial_conn.reset_input_buffer()
                
            camera_active = False
            continue

        if cv2.waitKey(1) == 27:
            break

    if cap: cap.release()
    cv2.destroyAllWindows()
    close_serial()
    log.info("=== System stopped ===")

if __name__ == "__main__":
    main()