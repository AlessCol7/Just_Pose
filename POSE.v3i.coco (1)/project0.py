import cv2
import os
import time
import random
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n-pose.pt')

# Set video path (0 for webcam)
video_path = 0
cap = cv2.VideoCapture(video_path)

# Path to the extracted images folder
images_folder = r'images'  # Update this path
image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]

last_image_time = time.time()
display_interval = 8  # Interval in seconds

# Loop over the video frames
while cap.isOpened():
    # Read frame
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Iterate over the list of Results objects
        for result in results.xywh:
            # Draw the detections on the frame
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                label = f'{result.names[int(cls)]}: {conf:.2f}'
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('YOLOv8 Inference', frame)

        # Check if it's time to display a new random image
        if time.time() - last_image_time >= display_interval:
            random_image_path = random.choice(image_files)
            random_image = cv2.imread(random_image_path)
            cv2.imshow('Random Pose Image', random_image)
            last_image_time = time.time()

        # Break the loop if "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('Video has ended or failed, exiting the loop')
        break

# Release the video capture object and close the display windows
cap.release()
cv2.destroyAllWindows()
