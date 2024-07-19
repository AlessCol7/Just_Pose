import cv2
import os
import time
import random
import json
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import euclidean

# Load YOLO model
model = YOLO('yolov8n-pose.pt')

# Path to the annotations JSON file
annotations_path = '/Users/alessiacolumban/Just_Pose/POSE.v3i.coco (1)/images/annotations_coco.json'

# Path to the extracted images folder
images_folder = '/Users/alessiacolumban/Just_Pose/POSE.v3i.coco (1)/images'
image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]

# Load pose annotations from JSON
def load_pose_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['images']

annotations = load_pose_annotations(annotations_path)

# Function to get pose from image using YOLO model
def get_pose_from_image(image, model):
    results = model(image)
    keypoints = []
    for result in results:
        if hasattr(result, 'keypoints'):
            keypoints.append(result.keypoints.xy.flatten())
    return keypoints[0] if keypoints else []

# Function to compare poses
def compare_poses(pose1, pose2, threshold=50):
    if len(pose1) != len(pose2):
        print("Pose lengths do not match.")
        return False
    distances = [euclidean(p1, p2) for p1, p2 in zip(pose1, pose2)]
    mean_distance = np.mean(distances)
    print(f"Mean distance: {mean_distance}")
    return mean_distance < threshold

# Function to display image
def display_image(image_path, window_name='Random Pose Image', wait_time=8000):
    if not os.path.isfile(image_path):
        print(f"Error: File {image_path} not found.")
        return False
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}.")
        return False
    cv2.imshow(window_name, image)
    cv2.waitKey(wait_time)
    cv2.destroyWindow(window_name)
    return True

# Function to capture and save screenshot
def capture_screenshot(output_path, frame):
    cv2.imwrite(output_path, frame)

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Main loop
last_image_time = time.time()
display_interval = 8  # Interval in seconds

while cap.isOpened():
    # Read frame
    success, frame = cap.read()
    if not success:
        print('Video has ended or failed, exiting the loop')
        break

    # Check if the frame is read correctly
    if frame is None:
        print("Error: Frame is empty.")
        continue

    # Run YOLOv8 inference on the frame
    try:
        results = model(frame)
    except Exception as e:
        print(f"Error during model inference: {e}")
        continue

    # Draw the detections on the frame
    for result in results:
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
        if display_image(random_image_path):
            last_image_time = time.time()

            # Extract keypoints for the random image
            random_image = cv2.imread(random_image_path)
            for image_data in annotations:
                if image_data['file_name'] in random_image_path:
                    keypoints = np.array(image_data['annotations'][0]['keypoints']).flatten()
                    print(f"Random image keypoints: {keypoints}")

                    # Get current pose from the webcam frame
                    current_pose = get_pose_from_image(frame, model)
                    print(f"Current pose keypoints: {current_pose}")

                    # Compare the current pose with the dataset pose
                    if compare_poses(current_pose, keypoints):
                        screenshot_path = f'saved_ss/screenshot_{int(time.time())}.jpg'
                        capture_screenshot(screenshot_path, frame)
                        print(f"Pose match found. Screenshot saved to {screenshot_path}.")
                    else:
                        print("Poses do not match.")
                    break

    # Break the loop if "q" is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display windows
cap.release()
cv2.destroyAllWindows()
