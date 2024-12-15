import cv2
import json

# Path to your video file and JSON file
video_path = '/home/client/jellyfish/datasets/dds/trafficcam_1.mp4'
json_path = '/home/client/jellyfish/pytorch_yolov4/ground_truth/dds/trafficcam_1/model_1280_704/output_dets.json' #Change to client output_dets directory

# Load the JSON file
with open(json_path, 'r') as f:
    annotations = json.load(f)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the frame width, height, and frames per second
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Calculate the total number of frames to process (30 seconds)
total_frames = fps * 80

# Define the codec and create a VideoWriter object
output_path = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Define distinct colors for each category_id
category_colors = {
    0: (255, 0, 0),       
    1: (0, 255, 0),       
    2: (0, 0, 255),       
    3: (255, 165, 0),     
    4: (75, 0, 130),      
    5: (148, 0, 211),     
    6: (64, 224, 208),    
    7: (255, 20, 147),    
    8: (220, 20, 60),     
    9: (128, 128, 0),     
}

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_id >= total_frames:
        break

    # Check if the frame has an associated bounding box annotation
    for annotation in annotations:
        if annotation['image_id'] == frame_id:
            # Extract bounding box and other data
            x, y, w, h = annotation['bbox']
            category_id = annotation['category_id']
            score = annotation['score']
            
            # Get color for the category (default to white if category_id not in the color map)
            color = category_colors.get(category_id, (255, 255, 255))
            
            # Draw the bounding box (convert to integer since OpenCV requires ints)
            start_point = (int(x), int(y))
            end_point = (int(x + w), int(y + h))
            
            # Draw the rectangle (color depends on category_id, thickness of 2)
            cv2.rectangle(frame, start_point, end_point, color, 2)
            
            # Add text for category ID and score
            label = f'Car: {category_id}, Score: {score:.2f}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            
            # Calculate position for text
            text_position = (int(x), int(y) - 10)
            
            # Draw the text on the frame
            cv2.putText(frame, label, text_position, font, font_scale, color, font_thickness, cv2.LINE_AA)

    # Write the frame to the output video
    out.write(frame)
    frame_id += 1

    # Uncomment this if you want to see the video as it's processed
    # cv2.imshow('Frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
