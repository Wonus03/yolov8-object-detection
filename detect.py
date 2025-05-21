from ultralytics import YOLO

# Load the model
model = YOLO("yolov8s.pt")

# Run detection on an image
results = model("sample.png", save=True)  # replace with your actual image path

# Print results
for r in results:
    print(r.boxes)
