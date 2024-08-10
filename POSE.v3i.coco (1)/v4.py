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
            kps = result.keypoints.xy.cpu().numpy()
            # Ensure keypoints_with_visibility is in the format [x, y, visibility] or just [x, y]
            keypoints_with_visibility = []
            for kp in kps[0]:
                if kp.shape[0] == 2:
                    # If only x, y are available, assume visibility is 1
                    keypoints_with_visibility.append(np.append(kp, [1.0]))
                else:
                    # Handle cases where visibility is available
                    keypoints_with_visibility.append(kp)
            print(f"Extracted keypoints: {keypoints_with_visibility}")  # Debug print for extracted keypoints
            keypoints.append(keypoints_with_visibility)
    return keypoints

# Function to calculate distances between specific keypoints
def calculate_distances(keypoints):
    distances = {}
    if len(keypoints) > 0:
        kp = keypoints[0]
        if len(kp) > max(keypoint_indices.values()):  # Ensure enough keypoints are present
            for part1, part2 in [('left_wrist', 'nose'), ('right_wrist', 'nose'), ('left_shoulder', 'right_shoulder')]:
                index1, index2 = keypoint_indices.get(part1), keypoint_indices.get(part2)
                if index1 is not None and index2 is not None:
                    if index1 < len(kp) and index2 < len(kp):
                        if kp[index1][2] > 0 and kp[index2][2] > 0:  # Check visibility
                            dist = distance.euclidean(kp[index1][:2], kp[index2][:2])
                            distances[f'{part1}_to_{part2}'] = dist
                            print(f"Calculated distance for {part1} to {part2}: {dist}")
                        else:
                            print(f"Keypoints for {part1} or {part2} are not visible.")
                    else:
                        print(f"Indices for {part1} or {part2} are out of bounds (index1: {index1}, index2: {index2}).")
                else:
                    print(f"Invalid indices for {part1} or {part2}.")
        else:
            print(f"Not enough keypoints extracted (expected {max(keypoint_indices.values()) + 1}, got {len(kp)}).")
        print(f"Calculated distances: {distances}")  # Debug print for calculated distances
    return distances if distances else None

# Function to compare distances using a normalized approach
def compare_distances(distances1, distances2):
    if distances1 and distances2 and 'left_shoulder_to_right_shoulder' in distances1 and 'left_shoulder_to_right_shoulder' in distances2:
        norm1 = distances1['left_shoulder_to_right_shoulder']
        norm2 = distances2['left_shoulder_to_right_shoulder']
        
        if norm1 > 0 and norm2 > 0:
            left_hand_diff = abs((distances1.get('left_wrist_to_nose', 0) / norm1) - (distances2.get('left_wrist_to_nose', 0) / norm2))
            right_hand_diff = abs((distances1.get('right_wrist_to_nose', 0) / norm1) - (distances2.get('right_wrist_to_nose', 0) / norm2))
            avg_diff = (left_hand_diff + right_hand_diff) / 2
            return avg_diff
    return float('inf')

# Function to calculate percentage similarity based on normalized difference
def calculate_similarity(diff, threshold=0.5):
    similarity = max(0, (1 - diff) * 100)
    print(f"Difference: {diff}, Similarity: {similarity}%")  # Debug print for similarity and difference
    return similarity >= threshold * 100, similarity

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
last_frame_time = time.time()
current_image = None

# Loop over the video frames
while cap.isOpened():
    # Read frame
    success, frame = cap.read()

    if success:
        current_time = time.time()
        
        # Process a frame every 2 seconds
        if current_time - last_frame_time >= 2:
            last_frame_time = current_time

            # Perform inference
            results = model(frame, save=False)

            # Extract keypoints from current frame
            current_keypoints = extract_keypoints(results)
            current_distances = calculate_distances(current_keypoints)
            print(f"Current distances: {current_distances}")  # Debug print for current distances

            # Display a new annotated image every 8 seconds
            if current_time - last_switch_time > 8 or current_image is None:
                current_image_path = random.choice(annotated_images)
                current_image = cv2.imread(current_image_path)
                annotated_results = model(current_image)
                annotated_keypoints = extract_keypoints(annotated_results)
                annotated_distances = calculate_distances(annotated_keypoints)
                print(f"Annotated distances: {annotated_distances}")  # Debug print for annotated distances
                last_switch_time = current_time

            # Compare distances and calculate similarity
            if current_distances and annotated_distances:
                diff = compare_distances(current_distances, annotated_distances)
                similar, similarity = calculate_similarity(diff, threshold=0.3)  # Adjust threshold as needed
                
                print(f"Pose similarity: {similarity}%")  # Print similarity percentage

                if similar:
                    # Save the frame as a screenshot
                    timestamp = int(current_time)
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
