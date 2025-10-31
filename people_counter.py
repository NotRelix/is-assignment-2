# =======================================================
# 5.1 INITIALIZATION
# =======================================================

# Import necessary packages
from centroidtracker import CentroidTracker
import numpy as np
import cv2
import os # For checking file existence

# --- MODEL AND CONFIGURATION SETUP ---
# Paths to YOLO model files (must be in the yolo_model/ directory)
YOLO_CONFIG = "yolo_model/yolov3.cfg"
YOLO_WEIGHTS = "yolo_model/yolov3.weights"
CLASSES_FILE = "yolo_model/coco.names"

# Ensure all YOLO files are present before proceeding
if not all(os.path.exists(f) for f in [YOLO_CONFIG, YOLO_WEIGHTS, CLASSES_FILE]):
    print("[ERROR] One or more YOLO model files are missing. Check yolo_model/ directory.")
    exit()

# Load the COCO class labels our YOLO model was trained on
with open(CLASSES_FILE, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)

# Get the output layer names that we need from YOLO
layer_names = net.getLayerNames()
# Determine the output layer names from the YOLO model
# This handles both older and newer OpenCV versions
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Filter to keep only the 'person' class ID (COCO dataset has 80 classes, person is often index 0)
# We find the index of "person" in the loaded classes list
PERSON_CLASS_ID = classes.index("person")
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3 # Non-Maximum Suppression threshold

# --- VIDEO I/O AND TRACKER SETUP ---
INPUT_VIDEO_PATH = "input/input_video.mp4"
OUTPUT_VIDEO_PATH = "output/output_video.mp4"

# Initialize video capture from file and get its properties
vs = cv2.VideoCapture(INPUT_VIDEO_PATH)
writer = None
(W, H) = (None, None)

# Initialize our centroid tracker and a list to store each tracked object
ct = CentroidTracker(maxDisappeared=40)

# Dictionary to store the trajectory history and counted status of each object
# { objectID: { "centroids": [centroid1, centroid2, ...], "counted": False } }
trackableObjects = {} 

# Initialize the total number of people who have moved up or down
totalDown = 0   # Corresponds to "Entering" if line is horizontal and camera is elevated
totalUp = 0     # Corresponds to "Exiting" if line is horizontal and camera is elevated

# Position of the virtual counting line (e.g., center of the frame)
# We use H // 2 for a horizontal line in the middle
Y_LINE_POS = 0 


# =======================================================
# 5.2 THE MAIN PROCESSING LOOP
# =======================================================

print("[INFO] starting video stream...")
# Loop over frames from the video stream
while True:
    # Read the next frame from the file
    (grabbed, frame) = vs.read()

    # If the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break

    # If the frame dimensions are empty, set them and the line position
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        # Set the virtual line position to the center of the frame
        Y_LINE_POS = H // 2

    # --- DETECTION PHASE (YOLO) ---
    rects = [] # List to store valid bounding boxes for the tracker

    # Construct a blob from the input frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    # Lists for Non-Maximum Suppression (NMS)
    boxes = []
    confidences = []
    classIDs = []

    # Loop over each of the layer outputs
    for output in detections:
        # Loop over each of the detections
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter detections to keep only "person" class with high confidence
            if classID == PERSON_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
                # Scale the bounding box coordinates back to the image size
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Use the center (x, y)-coordinates to derive the top and left corner
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Add the detection to the NMS lists
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply Non-Maximum Suppression (NMS) to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # Loop over the remaining detections after NMS
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            # Add the bounding box in [startX, startY, endX, endY] format to our list for the tracker
            rects.append((x, y, x + w, y + h))

    # --- TRACKING PHASE ---
    # Update our centroid tracker using the computed set of bounding box rectangles
    objects = ct.update(rects)

    # --- COUNTING AND VISUALIZATION PHASE ---
    # Draw a horizontal line in the center of the frame (the virtual gate)
    cv2.line(frame, (0, Y_LINE_POS), (W, Y_LINE_POS), (0, 255, 255), 2) # Yellow Line

    # Loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # Check to see if a trackable object exists for the current object ID
        to = trackableObjects.get(objectID, None)

        # 1. If there is no existing trackable object, create one
        if to is None:
            # We store the centroid history and a 'counted' flag
            to = {"centroids": [centroid], "counted": False}
        
        # 2. The object was tracked. Check direction and count.
        else:
            # Calculate direction by comparing current y-centroid to the mean of previous y-centroids
            # A negative value means the object is moving "up" (decreasing y-coordinate)
            # A positive value means the object is moving "down" (increasing y-coordinate)
            
            # Get the y-coordinates of the centroid history
            y_centroids = [c[1] for c in to["centroids"]]
            
            # Calculate the direction vector (current_y - mean_previous_y)
            direction_y = centroid[1] - np.mean(y_centroids)

            # Store the current centroid for next frame's comparison
            to["centroids"].append(centroid)

            # Check to see if the object has been counted or not
            if not to["counted"]:
                
                # Entering Event (Moving DOWN and crossed the line)
                # Previous mean Y-coord was ABOVE the line, current Y-coord is BELOW the line
                # direction_y > 0 means moving down
                if direction_y > 0 and centroid[1] > Y_LINE_POS:
                    totalDown += 1 # Increment "Entering" counter
                    to["counted"] = True # Prevent double counting

                # Exiting Event (Moving UP and crossed the line)
                # Previous mean Y-coord was BELOW the line, current Y-coord is ABOVE the line
                # direction_y < 0 means moving up
                elif direction_y < 0 and centroid[1] < Y_LINE_POS:
                    totalUp += 1 # Increment "Exiting" counter
                    to["counted"] = True # Prevent double counting
        
        # Store the trackable object back in our dictionary
        trackableObjects[objectID] = to

        # Draw both the ID of the object and the centroid of the object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Green ID
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # --- VISUALIZATION (COUNTERS) ---
    # Construct a tuple of information we will display on the frame
    info = [
        ("Exiting", totalUp),
        ("Entering", totalDown),
    ]

    # Loop over the info tuples and draw them