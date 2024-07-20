import cv2
from ultralytics import YOLO
import os
import random
import time
from scipy.spatial import distance

# Initialize the YOLOv8n-pose model
model = YOLO("yolov8n-pose.pt")  

# Function to extract keypoints from an image
def extract_keypoints(results):
    keypoints = []
    for result in results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints.append(result.keypoints.cpu().numpy())
    return keypoints

# Function to compare keypoints using Euclidean distance
def compare_keypoints(keypoints1, keypoints2):
    if keypoints1 and keypoints2:
        dist = distance.euclidean(keypoints1[0].flatten(), keypoints2[0].flatten())
        return dist
    return float('inf')

# Load annotated images
annotated_images_folder = '/Users/alessiacolumban/Just_Pose/POSE.v3i.coco (1)/output_folder'  # Update this path
annotated_images = [os.path.join(annotated_images_folder, f) for f in os.listdir(annotated_images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

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

        # Display a new annotated image every 8 seconds
        if time.time() - last_switch_time > 8 or current_image is None:
            current_image_path = random.choice(annotated_images)
            current_image = cv2.imread(current_image_path)
            annotated_results = model(current_image)
            annotated_keypoints = extract_keypoints(annotated_results)
            last_switch_time = time.time()
        
        # Compare keypoints
        if annotated_keypoints:
            dist = compare_keypoints(current_keypoints, annotated_keypoints)
            print(f"Distance to annotated pose: {dist}")

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
