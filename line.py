import cv2
from ultralytics import YOLO
import pytesseract

# -------------------------------
# TESSERACT PATH
# -------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------------
# LOAD MODELS
# -------------------------------
vehicle_model = YOLO("yolov8n.pt")
custom_model = YOLO("models/best.pt")

cap = cv2.VideoCapture(0)

# memory
line_counter = 0
plate_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    line_detected = False

    # -------------------------------
    # CUSTOM MODEL (IMPORTANT FIX 🔥)
    # -------------------------------
    results = custom_model(frame, conf=0.25)

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            label = custom_model.names[cls]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            area = (x2 - x1) * (y2 - y1)

            # -------------------------------
            # LINE CROSS FIX
            # -------------------------------
            if label == "line_cross" and conf > 0.25 and area > 2000:

                line_detected = True
                line_counter = 10

                # crop
                crop = frame[y1:y2, x1:x2]

                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                plate_text = pytesseract.image_to_string(gray)

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 4)
                cv2.putText(frame, "LINE CROSS", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)

    # -------------------------------
    # MEMORY FIX
    # -------------------------------
    if line_counter > 0:
        line_detected = True
        line_counter -= 1

    # -------------------------------
    # DISPLAY
    # -------------------------------
    if line_detected:
        cv2.putText(frame, "VIOLATION DETECTED!",
                    (400, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 3)

        cv2.putText(frame, f"PLATE: {plate_text}",
                    (400, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,255,255), 2)

    cv2.imshow("FINAL OCR SYSTEM", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()