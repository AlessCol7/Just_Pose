import cv2
import os
import json
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Initialize the YOLOv8n-pose model
model = YOLO("yolov8n-pose.pt")  # replace with your model path if different

# Define the path to your images folder and output folder
images_folder = '/Users/alessiacolumban/Just_Pose/POSE.v3i.coco (1)/images'
output_folder = '/Users/alessiacolumban/Just_Pose/POSE.v3i.coco (1)/output_folder'
coco_output_file = '/Users/alessiacolumban/Just_Pose/POSE.v3i.coco (1)/annotations.json'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the COCO keypoints order
COCO_KEYPOINTS = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
]

# Skeleton structure connecting keypoints
SKELETON = [
    [0, 1], [1, 3], [3, 5], [0, 2], [2, 4], [4, 6],
    [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
]

# COCO annotations template
coco_annotations = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": COCO_KEYPOINTS,
            "skeleton": SKELETON
        }
    ]
}

# Function to draw keypoints and skeleton on image
def draw_annotations(image, bbox, keypoints, keypoint_labels, skeleton, color=(0, 255, 0)):
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height

    # Draw bounding box
    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        if i < len(keypoint_labels):
            cv2.circle(image, (int(x), int(y)), 3, color, -1)
            cv2.putText(image, keypoint_labels[i], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

    # Draw skeleton
    for joint in skeleton:
        start_idx, end_idx = joint
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            cv2.line(image, 
                     (int(keypoints[start_idx][0]), int(keypoints[start_idx][1])), 
                     (int(keypoints[end_idx][0]), int(keypoints[end_idx][1])), 
                     color, 2)

# Process each image in the folder
for img_id, img_name in enumerate(os.listdir(images_folder), 1):
    if img_name.endswith(('.jpg', '.jpeg', '.png')):
        # Load image
        img_path = os.path.join(images_folder, img_name)
        img = cv2.imread(img_path)

        # Apply the YOLOv8n-pose model to get pose keypoints
        results = model(img_path)

        # Initialize list to collect all annotations for this image
        img_annotations = []

        # Get the pose keypoints
        for result_id, result in enumerate(results, 1):
            keypoints = result.keypoints.xy if result.keypoints else None
            if keypoints is not None:
                keypoints = keypoints.cpu().numpy().reshape(-1, 2)  # Ensure the keypoints are in the correct shape
                keypoints = keypoints.astype(float).tolist()  # Convert to float list

                # Get bounding box from result.boxes
                if result.boxes is not None:
                    bbox = result.boxes.xyxy.cpu().numpy()[0]
                    x_min, y_min, x_max, y_max = bbox
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    area = bbox_width * bbox_height

                    # Draw annotations on the image
                    draw_annotations(img, [x_min, y_min, bbox_width, bbox_height], keypoints, COCO_KEYPOINTS, SKELETON)

                    # Prepare COCO annotation
                    keypoints_with_visibility = []
                    for x, y in keypoints:
                        keypoints_with_visibility.extend([float(x), float(y), 2])  # Assume all keypoints are visible

                    annotation = {
                        "id": result_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": float(area),
                        "keypoints": keypoints_with_visibility,
                        "face_box": [],  # Placeholder, update with actual face box if available
                        "lefthand_box": [],  # Placeholder, update with actual left hand box if available
                        "righthand_box": [],  # Placeholder, update with actual right hand box if available
                        "foot_kpts": [],  # Placeholder, update with actual foot keypoints if available
                        "face_kpts": [],  # Placeholder, update with actual face keypoints if available
                        "lefthand_kpts": [],  # Placeholder, update with actual left hand keypoints if available
                        "righthand_kpts": [],  # Placeholder, update with actual right hand keypoints if available
                        "face_valid": False,  # Placeholder, update with actual validity if available
                        "lefthand_valid": False,  # Placeholder, update with actual validity if available
                        "righthand_valid": False,  # Placeholder, update with actual validity if available
                        "foot_valid": False  # Placeholder, update with actual validity if available
                    }
                    img_annotations.append(annotation)

        # Save the processed image
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, img)
        print(f"Processed and saved: {output_path}")

        # Add image info to COCO annotations
        coco_annotations["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": int(img.shape[1]),
            "height": int(img.shape[0])
        })

        # Add annotations for this image to COCO annotations
        coco_annotations["annotations"].extend(img_annotations)

# Convert all float32 values in coco_annotations to float
def convert_float32(o):
    if isinstance(o, np.float32):
        return float(o)
    raise TypeError

# Save COCO annotations to file
with open(coco_output_file, 'w') as f:
    json.dump(coco_annotations, f, indent=4, default=convert_float32)
