import cv2
import os

def draw_bounding_boxes(image_dir, label_dir):
    # For each file in the image directory
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load the image
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            # Load the corresponding label file
            label_filename = filename.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(label_dir, label_filename)
            with open(label_path, 'r') as f:
                labels = f.read()
            # For each line in the labels
            for line in labels.splitlines():
                # Parse the class, center x, center y, width, and height
                parts = list(map(float, line.split()))
                class_, cx, cy, w, h = parts[:5]

                # Convert to pixel coordinates
                x = (cx - w / 2) * width
                y = (cy - h / 2) * height
                w *= width
                h *= height

                # Draw the bounding box
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

            # Show the image
            cv2.imshow('Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

image_dir = "/Users/alessiacolumban/Desktop/Just_Pose/POSE.v3i.yolov8/train/images"
label_dir = "/Users/alessiacolumban/Desktop/Just_Pose/POSE.v3i.yolov8/train/labels"
draw_bounding_boxes(image_dir, label_dir)