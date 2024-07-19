import json

json_file_path = r'/Users/alessiacolumban/Just_Pose/POSE.v3i.coco (1)/annotations.json'  # Update this path

with open(json_file_path, 'r') as f:
    annotations = json.load(f)

# Print the type and structure of the JSON data
print(type(annotations))
print(annotations)
