from ultralytics import YOLO
import cv2
import os

# Load YOLO model
prediction_model = YOLO("best.pt")

# Define class names
class_names = ["infected_leaf"]

# Input video path
video_path = "input_video/dataset_1.mp4"

# Output directory
output_folder = "output_video"
os.makedirs(output_folder, exist_ok=True)

# Open video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Width
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second (correct playback speed)

# Define output video writer
output_path = os.path.join(output_folder, "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Break if the video ends

    # Perform inference using YOLO
    results = prediction_model(frame, conf=0.55)

    # Iterate over results and draw bounding boxes
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

    # Write the processed frame to the output video
    out.write(frame)

    # Display the annotated video (optional)
    cv2.imshow("Detection", frame)

    # Maintain correct playback speed by waiting between frames
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved at: {output_path}")
