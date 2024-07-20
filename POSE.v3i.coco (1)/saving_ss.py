import cv2
from ultralytics import YOLO
import os
import random
import time
from scipy.spatial import distance
import numpy as np

# Initialize the YOLOv8n-pose model
model = YOLO("yolov8n-pose.pt")  

# Keypoint indices for specific body parts (following COCO format)
keypoint_indices = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

# Function to extract keypoints from an image
def extract_keypoints(results):
    keypoints = []
    for result in results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints.append(result.keypoints.xy.cpu().numpy())  # Use xy to get the coordinates
    return keypoints

# Function to calculate distances between specific keypoints
def calculate_distances(keypoints):
    if len(keypoints) > 0:
        kp = keypoints[0]
        if kp.shape[0] == 17:  # Ensure we have all 17 keypoints
            distances = {
                'left_hand_to_head': distance.euclidean(kp[keypoint_indices['left_wrist']], kp[keypoint_indices['nose']]),
                'right_hand_to_head': distance.euclidean(kp[keypoint_indices['right_wrist']], kp[keypoint_indices['nose']]),
                'shoulder_width': distance.euclidean(kp[keypoint_indices['left_shoulder']], kp[keypoint_indices['right_shoulder']])
            }
            return distances
    return None

# Function to compare distances using a normalized approach
def compare_distances(distances1, distances2):
    if distances1 and distances2:
        norm1 = distances1['shoulder_width']
        norm2 = distances2['shoulder_width']
        
        if norm1 > 0 and norm2 > 0:
            left_hand_diff = abs((distances1['left_hand_to_head'] / norm1) - (distances2['left_hand_to_head'] / norm2))
            right_hand_diff = abs((distances1['right_hand_to_head'] / norm1) - (distances2['right_hand_to_head'] / norm2))
            avg_diff = (left_hand_diff + right_hand_diff) / 2
            return avg_diff
    return float('inf')

# Function to calculate percentage similarity based on normalized difference
def calculate_similarity(diff, threshold=0.1):
    similarity = max(0, (1 - diff) * 100)
    print(f"Difference: {diff}, Similarity: {similarity}%")  # Debug print for similarity and difference
    return similarity >= threshold * 100

# Load annotated images
annotated_images_folder = '/Users/alessiacolumban/Just_Pose/POSE.v3i.coco (1)/output_folder'  # Update this path
annotated_images = [os.path.join(annotated_images_folder, f) for f in os.listdir(annotated_images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Create the folder for saving screenshots if it doesn't exist
screenshots_folder = 'saved_ss'
os.makedirs(screenshots_folder, exist_ok=True)

# Open video file or capture device
video_path = 0
cap = cv2.VideoCapture(video_path)

last_switch_time = time.time()
current_image = None

# Loop over the video frames
while cap.isOpened():
    # Read frame
    success, frame = cap.read()

    if success:
        # Perform inference
        results = model(frame, save=False)

        # Extract keypoints from current frame
        current_keypoints = extract_keypoints(results)
        current_distances = calculate_distances(current_keypoints)
        print(f"Current distances: {current_distances}")  # Debug print for current distances

        # Display a new annotated image every 8 seconds
        if time.time() - last_switch_time > 8 or current_image is None:
            current_image_path = random.choice(annotated_images)
            current_image = cv2.imread(current_image_path)
            annotated_results = model(current_image)
            annotated_keypoints = extract_keypoints(annotated_results)
            annotated_distances = calculate_distances(annotated_keypoints)
            print(f"Annotated distances: {annotated_distances}")  # Debug print for annotated distances
            last_switch_time = time.time()
        
        # Compare distances and calculate similarity
        if current_distances and annotated_distances:
            diff = compare_distances(current_distances, annotated_distances)
            if calculate_similarity(diff, threshold=0.1):  # Adjust threshold as needed
                # Save the frame as a screenshot
                timestamp = int(time.time())
                screenshot_path = os.path.join(screenshots_folder, f"screenshot_{timestamp}.png")
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")

        # Visualize the results
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow('Live Feed', annotated_frame)
        if current_image is not None:
            cv2.imshow('Annotated Image', current_image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
