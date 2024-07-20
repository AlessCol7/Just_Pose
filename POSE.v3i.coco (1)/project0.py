import cv2
from ultralytics import YOLO

# Initialize the YOLOv8n-pose model
model = YOLO("yolov8n-pose.pt")  

# open file
video_path = 0
cap = cv2.VideoCapture(video_path)

# Loop over the video frames
while cap.isOpened():
    # Read frame
    success, frame = cap.read()

    if success:
        # Perform inference
        results = model(frame, save = True)

        # visualize the results
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow('frame', annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Release the video capture object
cap.release()
