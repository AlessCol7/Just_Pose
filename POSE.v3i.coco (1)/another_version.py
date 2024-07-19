import os
import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLO model
model = YOLO('yolov8n-pose.pt')

# Define the path to the images folder
images_folder = "/Users/alessiacolumban/Just_Pose/POSE.v3i.coco (1)/images/"

# Define the path to the new folder where annotated images will be saved
annotated_images_folder = os.path.join(images_folder, 'annotated_images')

# Create the new folder if it does not exist
os.makedirs(annotated_images_folder, exist_ok=True)

# List all image files in the images folder
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Check if there are any images to process
if not image_files:
    print("No images found in the specified folder.")
else:
    print(f"Found {len(image_files)} image(s) to process.")
    
    # Process each image
    for idx, image_file in enumerate(image_files):
        print(f"Processing file {idx + 1}/{len(image_files)}: {image_file}")
        image_path = os.path.join(images_folder, image_file)
        frame = cv2.imread(image_path)

        if frame is not None:
            print(f"Successfully read {image_file}")

            # Perform inference
            results = model(frame, save=False)  # save=False to avoid saving intermediate results
            
            if results:
                # Visualize the results
                annotated_frame = results[0].plot()

                # Convert PIL image to numpy array
                annotated_frame = np.array(annotated_frame)

                # Convert RGB to BGR
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

                # Save the annotated frame to the new folder
                output_path = os.path.join(annotated_images_folder, 'annotated_' + image_file)
                success = cv2.imwrite(output_path, annotated_frame)
                if success:
                    print(f"Processed and saved {image_file} to {output_path}")
                else:
                    print(f"Failed to save {image_file} to {output_path}")

            else:
                print(f"No results found for {image_file}")
        else:
            print(f"Error: Unable to open image file {image_path}")

    # Clean up
    cv2.destroyAllWindows()
