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
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
TESSERACT_PATH     = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
EMAIL_SENDER       = "sekaran2405@gmail.com"
EMAIL_PASSWORD     = "wrrf awsj dlkm smrn"
EXCEL_DB_PATH      = "vehicle_data.xlsx"
LOG_CSV_PATH       = "violation_log.csv"
VEHICLE_MODEL_PATH = "yolov8n.pt"
CUSTOM_MODEL_PATH  = "models/best.pt"
EMAIL_COOLDOWN_SECONDS = 120

# ──────────────────────────────────────────────────────────────
# UART / SERIAL CONFIGURATION
# ──────────────────────────────────────────────────────────────
SERIAL_PORT     = "COM5"      # ← Device Manager-ல பாத்து மாத்துங்க
SERIAL_BAUDRATE = 9600
SERIAL_ENABLED  = True        # False பண்ணா serial skip ஆகும்

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

# ──────────────────────────────────────────────────────────────
# SERIAL / UART SETUP
# ──────────────────────────────────────────────────────────────
serial_conn = None

def init_serial():
    """Serial connection திறக்கும். Fail ஆனா log மட்டும் பண்ணும்."""
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
        time.sleep(2)   # Arduino reset time
        log.info(f"[UART] ✅ Serial connected: {SERIAL_PORT} @ {SERIAL_BAUDRATE} baud")
    except serial.SerialException as e:
        log.error(f"[UART] ❌ Cannot open {SERIAL_PORT}: {e}")
        log.warning("[UART] Running without serial — traffic signals won't be sent.")
        serial_conn = None


def send_uart(value: bytes = b'1', label: str = ""):
    """
    Hardware-க்கு 1 byte send பண்ணும்.
    HIGH TRAFFIC or AMBULANCE detect ஆகும்போது call பண்ணப்படும்.
    """
    global serial_conn
    if serial_conn is None or not serial_conn.is_open:
        print(f"[UART] Not connected — skip send ({label})")
        return
    try:
        serial_conn.write(value)
        serial_conn.flush()
        print(f"[UART] ✅ Sent: {value} | Reason: {label}")
        log.info(f"[UART] Sent {value} | {label}")
    except serial.SerialException as e:
        print(f"[UART] ❌ Send failed: {e}")
        log.error(f"[UART] Send error: {e}")


def close_serial():
    global serial_conn
    if serial_conn and serial_conn.is_open:
        serial_conn.close()
        log.info("[UART] Serial connection closed.")


# ══════════════════════════════════════════════════════════════
# OCR FUNCTIONS
# ══════════════════════════════════════════════════════════════

def preprocess_plate_image(crop):
    crop  = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, sharp_kernel)
    gray  = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return thresh


def clean_plate_text(raw):
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def validate_plate(text):
    pattern = r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$"
    return re.match(pattern, text) is not None


def extract_number_plate(frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    try:
        variants = []
        v1 = preprocess_plate_image(crop)
        variants.append(("OTSU", v1))
        variants.append(("INV", cv2.bitwise_not(v1)))

        tmp  = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        tmp  = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        tmp  = cv2.bilateralFilter(tmp, 11, 17, 17)
        adap = cv2.adaptiveThreshold(tmp, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 2)
        variants.append(("ADAP", adap))

        for vname, processed in variants:
            pil_img = Image.fromarray(processed)
            for psm in [8, 7, 6, 13]:
                config = (
                    f"--psm {psm} --oem 3 "
                    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )
                raw   = pytesseract.image_to_string(pil_img, config=config)
                plate = clean_plate_text(raw)
                if validate_plate(plate):
                    return plate

        return None
    except Exception as e:
        print(f"  [OCR] Error: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# EXCEL DATABASE
# ══════════════════════════════════════════════════════════════

def create_sample_excel():
    pd.DataFrame({
        "Number Plate": ["TN22AB1234", "KA01CD5678", "MH12EF9012"],
        "Owner Name":   ["Arun Kumar",  "Priya Nair",  "Rahul Mehta"],
        "Phone Number": ["9876543210",  "8765432109",  "7654321098"],
        "Email ID":     ["arun@example.com", "priya@example.com", "rahul@example.com"],
    }).to_excel(EXCEL_DB_PATH, index=False)


def read_excel_data():
    if not Path(EXCEL_DB_PATH).exists():
        create_sample_excel()
    try:
        df = pd.read_excel(EXCEL_DB_PATH, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        df["Number Plate"] = df["Number Plate"].str.upper().str.replace(r"\s+", "", regex=True)
        return df
    except Exception as e:
        log.error(f"[Excel] Failed: {e}")
        return pd.DataFrame()


def lookup_owner(plate, df):
    if df.empty:
        return None
    match = df[df["Number Plate"] == plate]
    if not match.empty:
        row = match.iloc[0]
        return {
            "plate": plate,
            "name":  row.get("Owner Name",   "Unknown"),
            "phone": row.get("Phone Number", "N/A"),
            "email": row.get("Email ID",     ""),
        }
    return None


# ══════════════════════════════════════════════════════════════
# EMAIL
# ══════════════════════════════════════════════════════════════

def _send_email_worker(to_email, owner_name, plate, violation):
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    subject   = "Traffic Violation Alert - Fine Issued"
    body = (
        f"Dear {owner_name},\n\n"
        "This is an automated notice from the Smart Traffic Monitoring System.\n\n"
        f"Your vehicle with number plate {plate} has been detected committing "
        "a traffic violation:\n\n"
        f"  Violation  : {violation}\n"
        f"  Date/Time  : {timestamp}\n"
        "  Location   : Traffic Camera Unit - Junction 1\n\n"
        "A fine has been issued 500 RS. Please pay at your nearest traffic authority\n"
        "office within 15 days to avoid additional penalties.\n\n"
        f"  Number Plate : {plate}\n"
        f"  Owner        : {owner_name}\n\n"
        "Regards,\nSmart Traffic Monitoring System\n"
        "(Auto-generated — do not reply)\n"
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
        print(f"  [Email] Fine sent to {to_email}")
    except Exception as e:
        print(f"  [Email] Failed: {e}")


def send_email(owner, violation="Line Crossing"):
    plate = owner["plate"]
    email = owner.get("email", "")
    if not email:
        return
    last_sent = email_sent_times.get(plate)
    if last_sent and datetime.now() - last_sent < timedelta(seconds=EMAIL_COOLDOWN_SECONDS):
        return
    email_sent_times[plate] = datetime.now()
    threading.Thread(
        target=_send_email_worker,
        args=(email, owner["name"], plate, violation),
        daemon=True
    ).start()


# ══════════════════════════════════════════════════════════════
# VIOLATION LOG
# ══════════════════════════════════════════════════════════════

def init_log_csv():
    if not Path(LOG_CSV_PATH).exists():
        with open(LOG_CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["Number Plate", "Owner Name", "Phone", "Email",
                 "Date", "Time", "Violation Type"]
            )


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
        print(f"  [Log] Failed: {e}")


# ══════════════════════════════════════════════════════════════
# CAMERA HELPERS
# ══════════════════════════════════════════════════════════════

def _reopen_primary():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    time.sleep(1)
    return cap


# ══════════════════════════════════════════════════════════════
# OCR PIPELINE
# ══════════════════════════════════════════════════════════════

def run_ocr_pipeline(cap, df):
    print("\n" + "="*60)
    print("LINE CROSS — OCR WINDOW OPENING")
    print("="*60)

    plate      = None
    MAX_FRAMES = 300
    frame_num  = 0

    while frame_num < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            frame_num += 1
            continue

        frame   = cv2.resize(frame, (1280, 720))
        display = frame.copy()
        _draw_secondary_overlay(display, plate, frame_num, MAX_FRAMES)
        cv2.imshow(SECONDARY_WINDOW, display)

        if cv2.waitKey(1) == 27:
            break

        if frame_num % 5 == 0:
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            bw, bh = 560, 160
            gx1 = max(0, cx - bw // 2)
            gy1 = max(0, cy - bh // 2)
            gx2 = min(w, cx + bw // 2)
            gy2 = min(h, cy + bh // 2)

            candidate = extract_number_plate(frame, (gx1, gy1, gx2, gy2))
            if candidate and candidate not in processed_plates_this_session:
                plate = candidate
                _show_success_overlay(cap, display, plate)
                break

        frame_num += 1

    try:
        cv2.destroyWindow(SECONDARY_WINDOW)
    except cv2.error:
        pass
    cv2.waitKey(1)

    if plate is None:
        return None, None

    processed_plates_this_session.add(plate)
    owner = lookup_owner(plate, df)
    if not owner:
        owner = {"plate": plate, "name": "Unknown", "phone": "N/A", "email": ""}

    send_email(owner, "Line Crossing")
    log_violation(plate, owner)
    return plate, owner


# ══════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════

def _draw_secondary_overlay(frame, plate_so_far, frame_num, max_frames):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "!! LINE CROSS DETECTED -- SHOW NUMBER PLATE !!",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(frame, "Hold the vehicle number plate clearly in front of this camera",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    progress = int((frame_num / max_frames) * w)
    cv2.rectangle(frame, (0, h - 10), (w, h),        (50, 50, 50), -1)
    cv2.rectangle(frame, (0, h - 10), (progress, h), (0, 200, 255), -1)
    cx, cy = w // 2, h // 2
    bw, bh = 560, 160
    box_x1, box_y1 = cx - bw // 2, cy - bh // 2
    box_x2, box_y2 = cx + bw // 2, cy + bh // 2
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2)
    cl = 25
    for px, py, dx, dy in [
        (box_x1, box_y1,  1,  1), (box_x2, box_y1, -1,  1),
        (box_x1, box_y2,  1, -1), (box_x2, box_y2, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx * cl, py),           (0, 255, 0), 3)
        cv2.line(frame, (px, py), (px,            py + dy * cl), (0, 255, 0), 3)
    cv2.putText(frame, "[ ALIGN NUMBER PLATE HERE ]",
                (cx - 130, box_y2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"OCR Scanning... frame {frame_num}",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def _show_success_overlay(sec_cap, last_frame, plate):
    deadline = time.time() + 2.0
    while time.time() < deadline:
        ret, f = sec_cap.read()
        frame  = f if ret else last_frame.copy()
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 80, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, "PLATE DETECTED!",
                    (300, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        cv2.putText(frame, plate,
                    (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 0), 4)
        cv2.putText(frame, "Issuing fine & sending email...",
                    (280, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)
        cv2.imshow(SECONDARY_WINDOW, frame)
        cv2.waitKey(1)


def draw_info_panel(frame, car_count, bus_count, bike_count,
                    total_count, status, color_status):
    cv2.rectangle(frame, (10, 10), (520, 310), (0, 0, 0), -1)
    cv2.putText(frame, f"CAR  : {car_count}",   (20, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),    2)
    cv2.putText(frame, f"BUS  : {bus_count}",   (20, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0),    2)
    cv2.putText(frame, f"BIKE : {bike_count}",  (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),  2)
    cv2.putText(frame, f"TOTAL: {total_count}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, status,                  (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status,   2)


def draw_result_overlay(frame, plate, owner, box):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y2 + 5), (x1 + 400, y2 + 100), (0, 0, 0), -1)
    cv2.putText(frame, f"PLATE : {plate}", (x1 + 5, y2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    if owner and owner["name"] != "Unknown":
        cv2.putText(frame, f"OWNER : {owner['name']}", (x1 + 5, y2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        cv2.putText(frame, "FINE ISSUED — EMAIL SENT", (x1 + 5, y2 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


# ══════════════════════════════════════════════════════════════
# UART COOLDOWN — double send தவிர்க்க
# ══════════════════════════════════════════════════════════════
_uart_last_sent_time  = 0
_uart_cooldown_sec    = 5   # 5 seconds-க்கு ஒரு முறை மட்டும் send ஆகும்


def _maybe_send_uart(reason: str):
    """Cooldown check பண்ணி UART send பண்ணும்."""
    global _uart_last_sent_time
    now = time.time()
    if now - _uart_last_sent_time >= _uart_cooldown_sec:
        send_uart(b'1', reason)
        _uart_last_sent_time = now


# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════

def main():
    log.info("=== Smart Traffic System PRO — Starting ===")

    # Serial init
    init_serial()

    vehicle_model = YOLO(VEHICLE_MODEL_PATH)
    custom_model  = YOLO(CUSTOM_MODEL_PATH)

    df = read_excel_data()
    log.info(f"[Excel] {len(df)} records loaded.")
    init_log_csv()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    time.sleep(2)

    if not cap.isOpened():
        log.error("Cannot open webcam. Exiting.")
        close_serial()
        return

    line_counter      = 0
    ambulance_counter = 0
    osd_plate  = None
    osd_owner  = None
    osd_box    = None
    osd_frames = 0
    line_cross_cooldown = 0

    log.info("Running — press ESC to quit.")

    while True:
        if not cap.isOpened():
            time.sleep(1)
            cap = _reopen_primary()
            continue

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

        # ── VEHICLE DETECTION ──────────────────────────────────
        results1 = vehicle_model(frame, conf=0.4, verbose=False)
        for r in results1:
            for box in r.boxes:
                cls   = int(box.cls[0])
                label = vehicle_model.names[cls]
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

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ── CUSTOM DETECTION ───────────────────────────────────
        results2 = custom_model(frame, conf=0.10, verbose=False)
        for r in results2:
            for box in r.boxes:
                cls   = int(box.cls[0])
                label = custom_model.names[cls]
                conf  = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area  = (x2 - x1) * (y2 - y1)

                print(f"[DETECT] {label:<12}  conf={conf:.2f}  area={area}")

                if label == "ambulance" and conf > 0.9 and area > 5000:
                    ambulance_detected = True
                    ambulance_counter  = 15
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(frame, "AMBULANCE", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

                elif label == "line_cross" and conf > 0.25 and area > 3000 and line_cross_cooldown == 0:
                    line_detected = True
                    line_counter  = 10
                    line_box      = (x1, y1, x2, y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    cv2.putText(frame, "LINE CROSS", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        # ── MEMORY / STABILITY ─────────────────────────────────
        if line_counter > 0:
            line_detected = True; line_counter -= 1
        if ambulance_counter > 0:
            ambulance_detected = True; ambulance_counter -= 1

        # ══════════════════════════════════════════════════════
        # 🔴 SIGNAL STATUS — UART SEND பண்ணும்
        # ══════════════════════════════════════════════════════
        if ambulance_detected:
            status, color_status = "AMBULANCE - GREEN SIGNAL", (0, 255, 0)
            # Ambulance detect → hardware-க்கு 1 send
            _maybe_send_uart("AMBULANCE DETECTED")

        elif total_count >= 10:
            status, color_status = "HIGH TRAFFIC", (0, 0, 255)
            # High traffic detect → hardware-க்கு 1 send
            _maybe_send_uart("HIGH TRAFFIC")

        else:
            status, color_status = "LOW TRAFFIC", (0, 255, 255)
            # Low traffic — send பண்ண வேண்டாம்

        # ══════════════════════════════════════════════════════
        # LINE CROSS → OCR PIPELINE
        # ══════════════════════════════════════════════════════
        if line_detected and line_box is not None and line_cross_cooldown == 0:
            line_cross_cooldown = 300
            plate, owner = run_ocr_pipeline(cap, df)
            if plate:
                osd_plate  = plate
                osd_owner  = owner
                osd_box    = line_box
                osd_frames = 120
            line_box      = None
            line_detected = False
            line_counter  = 0
            continue

        # ── OSD result overlay ─────────────────────────────────
        if osd_frames > 0:
            osd_frames -= 1
            if osd_box and osd_plate:
                draw_result_overlay(frame, osd_plate, osd_owner, osd_box)

        # ── INFO PANEL ─────────────────────────────────────────
        draw_info_panel(frame, car_count, bus_count, bike_count,
                        total_count, status, color_status)

        if line_detected:
            cv2.putText(frame, "LINE CROSS DETECTED!", (700, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        if ambulance_detected:
            cv2.putText(frame, "AMBULANCE PRIORITY!", (700, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        if osd_frames > 0 and osd_plate:
            cv2.putText(frame, f"OCR: {osd_plate}", (700, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        if line_cross_cooldown > 0:
            cd_pct = int((line_cross_cooldown / 300) * 200)
            cv2.rectangle(frame, (540, 10), (740, 30), (50, 50, 50), -1)
            cv2.rectangle(frame, (540, 10), (540 + cd_pct, 30), (0, 165, 255), -1)
            cv2.putText(frame, "COOLDOWN", (545, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(PRIMARY_WINDOW, frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    close_serial()
    log.info("=== System stopped ===")


if __name__ == "__main__":
    main()