import os
import cv2
import json

# Function to annotate images using YOLOv8n-pose
def annotate_images(image_folder):
    annotations = []
    image_files = os.listdir(image_folder)
    
    image_id = 1  # Starting image ID
    
    for filename in image_files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            
            # Replace with your YOLOv8n-pose inference code
            # Example: perform inference with YOLOv8n-pose
            detected_poses = yolo_v8n_pose_inference(image)
            
            # Example: structure annotations (replace with actual output from YOLOv8n-pose)
            image_annotations = {
                "id": image_id,
                "file_name": filename,
                "width": image.shape[1],
                "height": image.shape[0],
                "annotations": []
            }
            
            for pose_id, pose in enumerate(detected_poses, 1):
                keypoints = [(kp[0], kp[1]) for kp in pose['keypoints']]
                bbox = pose['bbox']
                
                # Create annotation entry for each pose
                annotation = {
                    "id": pose_id,
                    "image_id": image_id,
                    "category_id": 1,  # Assuming one category for poses
                    "keypoints": keypoints,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],  # Area calculation for COCO format
                    "iscrowd": 0  # Assuming not crowd in COCO format
                }
                
                image_annotations["annotations"].append(annotation)
            
            annotations.append(image_annotations)
            image_id += 1
    
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
    image_folder = "/Users/alessiacolumban/Downloads/POSE.v3i.coco (1)/images"
    
    annotations = annotate_images(image_folder)
    
    # Save annotations to COCO-style JSON file
    coco_annotations = {
        "images": annotations,
        "categories": [{"id": 1, "name": "person"}]  # Example category definition
    }
    
    with open('annotations_coco.json', 'w') as f:
        json.dump(coco_annotations, f, indent=4, default=str)  # Use default=str to handle non-serializable types
