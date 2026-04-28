# # Entry point
# import cv2
# from ultralytics import YOLO

# # -------------------------------
# # LOAD MODELS
# # -------------------------------
# vehicle_model = YOLO("yolov8n.pt")      # vehicle detection
# custom_model = YOLO("models/best.pt") # ambulance + line_cross

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.resize(frame, (1280, 720))

#     car_count = 0
#     bus_count = 0
#     bike_count = 0
#     total_count = 0

#     ambulance_detected = False
#     violation = False

#     # -------------------------------
#     # VEHICLE DETECTION
#     # -------------------------------
#     results1 = vehicle_model(frame, conf=0.4)

#     for r in results1:
#         for box in r.boxes:
#             cls = int(box.cls[0])
#             label = vehicle_model.names[cls]

#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             if label == "car":
#                 car_count += 1
#                 total_count += 1
#                 color = (0,255,0)

#             elif label == "bus":
#                 bus_count += 1
#                 total_count += 1
#                 color = (255,0,0)

#             elif label in ["motorbike","bicycle"]:
#                 bike_count += 1
#                 total_count += 1
#                 color = (0,255,255)

#             elif label == "truck":
#                 total_count += 1
#                 color = (0,200,200)

#             else:
#                 continue

#             cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
#             cv2.putText(frame, label, (x1,y1-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     # -------------------------------
#     # CUSTOM DETECTION
#     # -------------------------------
#     results2 = custom_model(frame, conf=0.5)

#     for r in results2:
#         for box in r.boxes:
#             cls = int(box.cls[0])
#             label = custom_model.names[cls]

#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             if label == "ambulance":
#                 ambulance_detected = True
#                 color = (0,255,0)

#             elif label == "line_cross":
#                 violation = True
#                 color = (0,0,255)

#             else:
#                 continue

#             cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
#             cv2.putText(frame, label.upper(), (x1,y1-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     # -------------------------------
#     # SIGNAL LOGIC
#     # -------------------------------
#     if ambulance_detected:
#         status = "AMBULANCE - GREEN SIGNAL"
#         color_status = (0,255,0)
#     else:
#         if total_count < 10:
#             status = "LOW TRAFFIC"
#             color_status = (0,255,255)
#         else:
#             status = "HIGH TRAFFIC"
#             color_status = (0,0,255)

#     # -------------------------------
#     # DISPLAY
#     # -------------------------------
#     cv2.rectangle(frame, (10,10), (500,250), (0,0,0), -1)

#     cv2.putText(frame, f"CAR: {car_count}", (20,50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

#     cv2.putText(frame, f"BUS: {bus_count}", (20,90),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

#     cv2.putText(frame, f"BIKE: {bike_count}", (20,130),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

#     cv2.putText(frame, f"TOTAL: {total_count}", (20,170),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

#     cv2.putText(frame, status, (20,210),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)

#     if violation:
#         cv2.putText(frame, "VIOLATION!", (800,60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

#     cv2.imshow("SMART TRAFFIC FINAL", frame)

#     if cv2.waitKey(1) == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

# -------------------------------
# LOAD MODELS
# -------------------------------
vehicle_model = YOLO("yolov8n.pt")
custom_model = YOLO("models/best.pt")

cap = cv2.VideoCapture(0)

# 🔥 memory
line_counter = 0
ambulance_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    car_count = 0
    bus_count = 0
    bike_count = 0
    total_count = 0

    line_detected = False
    ambulance_detected = False

    # -------------------------------
    # VEHICLE DETECTION
    # -------------------------------
    results1 = vehicle_model(frame, conf=0.4)

    for r in results1:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = vehicle_model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "car":
                car_count += 1
                total_count += 1
                color = (0,255,0)

            elif label == "bus":
                bus_count += 1
                total_count += 1
                color = (255,0,0)

            elif label in ["motorbike","bicycle"]:
                bike_count += 1
                total_count += 1
                color = (0,255,255)

            elif label == "truck":
                total_count += 1
                color = (0,200,200)

            else:
                continue

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -------------------------------
    # CUSTOM DETECTION
    # -------------------------------
    results2 = custom_model(frame, conf=0.25)  # 🔥 low for line_cross

    for r in results2:
        for box in r.boxes:

            cls = int(box.cls[0])
            label = custom_model.names[cls]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 🔥 AREA FILTER
            area = (x2 - x1) * (y2 - y1)

            if label == "ambulance":
                # 🔥 STRICT CONDITION
                if conf > 0.6 and area > 5000:
                    ambulance_detected = True
                    ambulance_counter = 15

                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 4)
                    cv2.putText(frame, "AMBULANCE", (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 3)

            elif label == "line_cross":
                # 🔥 EASY CONDITION
                if conf > 0.25 and area > 3000:
                    line_detected = True
                    line_counter = 10

                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 4)
                    cv2.putText(frame, "LINE CROSS", (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)

    # -------------------------------
    # MEMORY (STABILITY)
    # -------------------------------
    if line_counter > 0:
        line_detected = True
        line_counter -= 1

    if ambulance_counter > 0:
        ambulance_detected = True
        ambulance_counter -= 1

    # -------------------------------
    # SIGNAL LOGIC
    # -------------------------------
    if ambulance_detected:
        status = "AMBULANCE - GREEN SIGNAL"
        color_status = (0,255,0)
    else:
        if total_count < 10:
            status = "LOW TRAFFIC"
            color_status = (0,255,255)
        else:
            status = "HIGH TRAFFIC"
            color_status = (0,0,255)

    # -------------------------------
    # DISPLAY PANEL
    # -------------------------------
    cv2.rectangle(frame, (10,10), (520,280), (0,0,0), -1)

    cv2.putText(frame, f"CAR: {car_count}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(frame, f"BUS: {bus_count}", (20,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.putText(frame, f"BIKE: {bike_count}", (20,130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.putText(frame, f"TOTAL: {total_count}", (20,170),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(frame, status, (20,210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)

    # -------------------------------
    # ALERT DISPLAY
    # -------------------------------
    if line_detected:
        cv2.putText(frame, "LINE CROSS DETECTED!",
                    (700, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 3)

    if ambulance_detected:
        cv2.putText(frame, "AMBULANCE PRIORITY!",
                    (700, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 3)

    cv2.imshow("SMART TRAFFIC FINAL PRO", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()