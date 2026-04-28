from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data_custom.yaml",   # ✅ correct
    epochs=100,
    imgsz=640,
    batch=4,
    device="cpu",
    name="custom_model"
)