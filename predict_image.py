from ultralytics import YOLO
import cv2
import os

# Load YOLO model
prediction_model = YOLO("best.pt")

# Define class names
class_names = ["infected_leaf"]

# Load a single image
image_path = "input_image/frame_01250.jpg"
frame = cv2.imread(image_path)

# Perform inference using the YOLO model
results = prediction_model(frame, conf=0.6)

# Iterate over the results
for result in results:
    boxes = result.boxes
    for box in boxes:
        conf = box.conf[0]
        cls = int(box.cls[0].item())  # Convert tensor to integer
        x1, y1, x2, y2 = box.xyxy[0]

        # Map class index to class name
        class_name = class_names[cls]

        # Draw bounding box and class name
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(frame, f"{class_name}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# Create output directory if it doesn't exist
output_folder = "output_image"
os.makedirs(output_folder, exist_ok=True)

# Save the annotated image
output_path = os.path.join(output_folder, "output.jpg")
cv2.imwrite(output_path, frame)

# Display the annotated image
cv2.imshow("Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Saved output image at: {output_path}")
