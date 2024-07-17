import cv2
import os
import time
import random
from ultralytics import YOLO
import numpy as np

# Load YOLO model
model = YOLO('yolov8n-pose.pt')

# Set video path (0 for webcam)
video_path = 0
cap = cv2.VideoCapture(video_path)

# Path to the extracted images folder
images_folder = r'/Users/alessiacolumban/Just_Pose/POSE.v3i.coco/train'  # Update this path
image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]

last_image_time = time.time()
display_interval = 8  # Interval in seconds

# Function to calculate IoU between two bounding boxes
def calculate_iou(boxA, boxB):
    # Extract coordinates
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB
    
    # Calculate intersection coordinates
    xA = max(x1A, x1B)
    yA = max(y1A, y1B)
    xB = min(x2A, x2B)
    yB = min(y2A, y2B)
    
    # Calculate intersection area
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # Calculate area of each box
    boxA_area = (x2A - x1A + 1) * (y2A - y1A + 1)
    boxB_area = (x2B - x1B + 1) * (y2B - y1B + 1)
    
    # Calculate union area
    union_area = boxA_area + boxB_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

# Loop over the video frames
while cap.isOpened():
    # Read frame
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, save=True)

        # Iterate over the list of Results objects
        for result in results:
            # Check if there are any detections
            if result.boxes.xyxy.shape[0] > 0:
                # Display the detections on the frame
                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    label = f'{result.names[int(cls)]}: {conf:.2f}'
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                # Compare with the last displayed random image
                random_image = cv2.imread("/Users/alessiacolumban/Just_Pose/POSE.v3i.coco/train)  # Load the last displayed random image")
                pose_detected = frame  # Replace with actual pose detection result
                pose_reference = random_image  # Replace with actual random image pose

                # Example comparison using IoU
                detected_box = result.boxes.xyxy[0]  # Assuming one detection per frame
                reference_box = [x1, y1, x2, y2]  # Adjust this to match your reference pose coordinates
                
                iou = calculate_iou(detected_box, reference_box)
                
                if iou >= 0.5:  # Adjust the threshold as needed
                    # Take screenshot and save it
                    screenshot_path = os.path.join(images_folder, f'screenshot_{time.time()}.jpg')
                    cv2.imwrite(screenshot_path, frame)
                    print(f'Saved screenshot: {screenshot_path}')

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
