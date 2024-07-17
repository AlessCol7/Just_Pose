import cv2
import os
import random
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n-pose.pt')

# Set video path (0 for webcam)
video_path = 0
cap = cv2.VideoCapture(video_path)

# List all images in the 'images' subfolder
current_folder = os.path.dirname(os.path.abspath(__file__))
images_folder = os.path.join(current_folder, 'images')
image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]

last_image_time = time.time()
display_interval = 8  # Interval in seconds

# Loop over the video frames
while cap.isOpened():
    # Read frame
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, save=True)

        # Visualize the results
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow('YOLOv8 Inference', annotated_frame)

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
