from ultralytics import YOLO

model = YOLO("yolo171.pt")  # ← change to your actual path

model.export(
    format="onnx",
    imgsz=640,
    half=False,        # keep FP32 for ONNX — TensorRT will do FP16 conversion
    dynamic=False,     # fixed batch size = 1 for deployment
    simplify=True,     # simplify ONNX graph
    opset=17,          # TensorRT 10.x supports up to opset 17
)