import os
import cv2
import json

# Function to annotate images using YOLOv8n-pose
def annotate_images(image_folder):
    annotations = []
    image_files = os.listdir(image_folder)
    
    for filename in image_files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            
            # Replace with your YOLOv8n-pose inference code
            # Example: perform inference with YOLOv8n-pose
            detected_poses = yolo_v8n_pose_inference(image)
            
            # Example: structure annotations (replace with actual output from YOLOv8n-pose)
            for pose in detected_poses:
                keypoints = [(kp[0], kp[1]) for kp in pose['keypoints']]
                annotation = {
                    "image_path": image_path,
                    "pose_keypoints": keypoints,
                    "bounding_box": pose['bbox']
                }
                annotations.append(annotation)
    
    return annotations

# Dummy function to simulate YOLOv8n-pose inference (replace with actual code)
def yolo_v8n_pose_inference(image):
    # Simulated output, replace with actual YOLOv8n-pose inference
    detected_poses = [
        {
            "keypoints": [(100, 200), (150, 250), (200, 300)],  # Example keypoints
            "bbox": [50, 100, 200, 300]  # Example bounding box [x_min, y_min, width, height]
        },
        # Add more poses as necessary
    ]
    return detected_poses

# Example usage
if __name__ == "__main__":
    image_folder = "images"
    
    annotations = annotate_images(image_folder)
    
    # Print annotations (for verification)
    for annotation in annotations:
        print(annotation)
    
    # Save annotations to JSON file
    with open('annotations.json', 'w') as f:
        json.dump(annotations, f, indent=4, default=str)  # Use default=str to handle non-serializable types
