import cv2
import os
import time
import random
import json
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n-pose.pt')

# Set video path (0 for webcam)
video_path = 0
cap = cv2.VideoCapture(video_path)

# Path to the extracted images folder and annotations file
images_folder = '/Users/alessiacolumban/Just_Pose/POSE.v3i.coco/train'
annotations_file = os.path.join(images_folder, '_annotations.coco.json')

# Load annotations
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# Create a mapping from image filenames to keypoints
image_keypoints = {}
for annotation in annotations['annotations']:
    image_id = annotation['image_id']
    if 'keypoints' not in annotation:
        continue  # Skip annotations without keypoints
    keypoints = annotation['keypoints']
    image_filename = next((img['file_name'] for img in annotations['images'] if img['id'] == image_id), None)
    if image_filename:
        image_keypoints[image_filename] = keypoints

# List of image files
image_files = list(image_keypoints.keys())

last_image_time = time.time()
display_interval = 8  # Interval in seconds
reference_keypoints = None

# Function to calculate similarity between two sets of keypoints
def calculate_similarity(kp1, kp2, threshold=0.5):
    kp1 = np.array(kp1).reshape(-1, 3)[:, :2]  # Only (x, y) coordinates
    kp2 = np.array(kp2).reshape(-1, 3)[:, :2]
    
    # Euclidean distance
    distances = np.linalg.norm(kp1 - kp2, axis=1)
    
    # Normalize and calculate similarity
    similarity = np.mean(distances < threshold)
    
    return similarity

# Loop over the video frames
while cap.isOpened():
    # Read frame
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Extract keypoints from YOLO results
        for result in results:
            if result.boxes.xyxy.shape[0] > 0:
                # Extract detected keypoints
                detected_keypoints = result.keypoints.xy.cpu().numpy()[0].flatten().tolist()
                
                # Debug print to verify the structure of detected keypoints
                print(f"Detected keypoints: {detected_keypoints}")
                print(f"Length of detected keypoints: {len(detected_keypoints)}")
                
                # Check if it's time to display a new random image
                if time.time() - last_image_time >= display_interval or reference_keypoints is None:
                    random_image_path = random.choice(image_files)
                    random_image = cv2.imread(os.path.join(images_folder, random_image_path))
                    reference_keypoints = image_keypoints[random_image_path]
                    
                    # Ensure the reference keypoints have the same length as detected keypoints
                    if len(reference_keypoints) != len(detected_keypoints):
                        # Add default confidence of 1.0 to reference keypoints
                        reference_keypoints = [kp if (i + 1) % 3 != 0 else 1.0 for i, kp in enumerate(reference_keypoints * (len(detected_keypoints) // len(reference_keypoints)))]
                    
                    # Debug print to verify the structure of reference keypoints
                    print(f"Reference keypoints: {reference_keypoints}")
                    print(f"Length of reference keypoints: {len(reference_keypoints)}")
                    cv2.imshow('Random Pose Image', random_image)
                    last_image_time = time.time()

                # Compare detected keypoints with reference keypoints
                if reference_keypoints is not None:
                    if len(detected_keypoints) == len(reference_keypoints):
                        similarity = calculate_similarity(detected_keypoints, reference_keypoints)
                        if similarity >= 0.5:  # Adjust the threshold as needed
                            # Take screenshot and save it
                            screenshot_folder = '/Users/alessiacolumban/Just_Pose/POSE.v3i.coco/saved_screenshots'
                            os.makedirs(screenshot_folder, exist_ok=True)
                            screenshot_path = os.path.join(screenshot_folder, f'screenshot_{time.time()}.jpg')
                            cv2.imwrite(screenshot_path, frame)
                            print(f'Saved screenshot: {screenshot_path}')
                    else:
                        print("Keypoints length mismatch even after alignment. Skipping similarity check.")

            # Display the frame
            cv2.imshow('YOLOv8 Inference', frame)

        # Break the loop if "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('Video has ended or failed, exiting the loop')
        break

# Release the video capture object and close the display windows
cap.release()
cv2.destroyAllWindows()
