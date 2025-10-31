# people_counter.py
# =======================================================
# 5.1 INITIALIZATION
# =======================================================

from centroidtracker import CentroidTracker
import numpy as np
import cv2
import os
import datetime

# --- MODEL AND CONFIGURATION SETUP ---
YOLO_CONFIG = "yolo_model/yolov3.cfg"
YOLO_WEIGHTS = "yolo_model/yolov3.weights"
CLASSES_FILE = "yolo_model/coco.names"

# Ensure all YOLO files exist
for f in [YOLO_CONFIG, YOLO_WEIGHTS, CLASSES_FILE]:
    if not os.path.exists(f):
        print(f"[ERROR] Missing: {f}")
        exit()

# Load COCO class labels
with open(CLASSES_FILE, "r") as f:
    classes = [line.strip() for line in f.readlines()]

print(f"[INFO] Loaded {len(classes)} classes (first 5): {classes[:5]}")

# --- Load YOLO network ---
print("[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG, YOLO_WEIGHTS)

# Try enabling GPU (if OpenCV built with CUDA); fall back to CPU
try:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("[INFO] Using GPU (CUDA) backend for DNN")
except Exception as e:
    print("[INFO] Using CPU backend for DNN:", e)

# --- Output layers (robust to OpenCV versions) ---
layer_names = net.getLayerNames()
outs = net.getUnconnectedOutLayers()
outs_flat = np.array(outs).flatten()
output_layers = [layer_names[i - 1] for i in outs_flat]
print("[DEBUG] Output layers:", output_layers)

# --- Detection thresholds (you can tweak) ---
if "person" not in classes:
    print("[ERROR] 'person' not found in classes. Check coco.names.")
    exit()
PERSON_CLASS_ID = classes.index("person")

CONFIDENCE_THRESHOLD = 0.2   # final confidence threshold (objectness * class_prob)
NMS_THRESHOLD = 0.4

# --- VIDEO INPUT/OUTPUT ---
INPUT_VIDEO_PATH = "input/input_video.mov"
OUTPUT_VIDEO_PATH = "output/output_video.mp4"

vs = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not vs.isOpened():
    print(f"[ERROR] Could not open input video: {INPUT_VIDEO_PATH}")
    exit()

writer = None
(W, H) = (None, None)

# Initialize tracker and variables
ct = CentroidTracker(maxDisappeared=40)
trackableObjects = {}
totalLeft = 0
totalRight = 0

print("[INFO] Starting video stream...")

# =======================================================
# 5.2 MAIN LOOP (Turnstile-Style)
# =======================================================

GATE_LEFT = GATE_RIGHT = GATE_TOP = GATE_BOTTOM = None
GATE_WIDTH = 250
FRAME_MIN_DIM = 416  # minimum width/height to ensure detection scale is okay

frame_idx = 0
no_detection_saved = False

while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        print("[INFO] End of video file reached.")
        break

    frame_idx += 1

    # Upscale very small frames so YOLO sees reasonable input
    if frame.shape[1] < FRAME_MIN_DIM or frame.shape[0] < FRAME_MIN_DIM:
        scale_x = max(1, FRAME_MIN_DIM / frame.shape[1])
        scale_y = max(1, FRAME_MIN_DIM / frame.shape[0])
        scale = max(scale_x, scale_y)
        new_w = int(frame.shape[1] * scale)
        new_h = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if W is None or H is None:
        (H, W) = frame.shape[:2]
        X_CENTER = W // 2
        GATE_LEFT = X_CENTER - (GATE_WIDTH // 2)
        GATE_RIGHT = X_CENTER + (GATE_WIDTH // 2)
        GATE_TOP = int(H * 0.7)
        GATE_BOTTOM = int(H * 1.5)

    rects = []

    # Optional: brighten dark frames
    gray_tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray_tmp.mean()
    if mean_brightness < 50:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enh = clahe.apply(gray_tmp)
        frame = cv2.cvtColor(gray_enh, cv2.COLOR_GRAY2BGR)
        print(f"[DEBUG] Applied CLAHE on frame {frame_idx} (mean brightness {mean_brightness:.1f})")

    # --- YOLO detection ---
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes, confidences = [], []

    # debug counters
    raw_detections = 0
    person_candidates = 0

    for output in detections:
        for detection in output:
            raw_detections += 1
            # detection format: [center_x, center_y, width, height, objectness, class_probs...]
            objectness = float(detection[4])
            scores = detection[5:]
            classID = int(np.argmax(scores))
            class_prob = float(scores[classID])
            # IMPORTANT: multiply objectness by class probability
            confidence = objectness * class_prob

            # debug: show some detected values for first few raw detections
            if raw_detections <= 5:
                print(f"[RAW] det#{raw_detections}: obj={objectness:.3f} cls={classes[classID]}({classID}) cls_prob={class_prob:.3f} final_conf={confidence:.3f}")

            # only consider final confidence and 'person' class
            if classID == PERSON_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
                person_candidates += 1
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # clamp & sanity-check
                w = max(1, int(width))
                h = max(1, int(height))
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))
                x = max(0, x)
                y = max(0, y)
                if x + w > W:
                    w = W - x
                if y + h > H:
                    h = H - y

                # if tiny, skip (you can lower these if your people are very small)
                if w < 10 or h < 20:
                    continue

                boxes.append([x, y, int(w), int(h)])
                confidences.append(confidence)

    # NMS handling
    idxs = []
    if len(boxes) > 0:
        try:
            idxs_raw = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            if isinstance(idxs_raw, (tuple, list, np.ndarray)) and len(idxs_raw) > 0:
                idxs = np.array(idxs_raw).flatten()
            else:
                idxs = []
        except Exception as e:
            print("[WARN] NMSBoxes exception:", e)
            idxs = list(range(len(boxes)))

    print(f"[DEBUG] Frame {frame_idx}: raw={raw_detections} person_cand={person_candidates} boxes={len(boxes)} kept_after_nms={len(idxs)}")

    # Save one debug frame if nothing detected to inspect visually
    if len(idxs) == 0 and (frame_idx % 30 == 0) and not no_detection_saved:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"debug_no_detections_{ts}.jpg"
        cv2.imwrite(save_path, frame)
        print(f"[DEBUG] Saved frame with no detections to {save_path} for inspection")
        no_detection_saved = True

    if len(idxs) > 0:
        for i in idxs:
            i = int(i)
            (x, y, w, h) = boxes[i]
            rects.append((x, y, x + w, y + h))
            conf = confidences[i] if i < len(confidences) else 0.0
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"person {conf:.2f}", (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # --- Tracker update ---
    objects = ct.update(rects)

    # Draw gate zone
    cv2.rectangle(frame, (GATE_LEFT, GATE_TOP), (GATE_RIGHT, GATE_BOTTOM), (0, 255, 255), 2)
    cv2.putText(frame, "GATE ZONE", (GATE_LEFT - 10, GATE_TOP - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)
        if to is None:
            to = {"centroids": [centroid], "counted": False}
        else:
            prev_x = [c[0] for c in to["centroids"]]
            direction_x = centroid[0] - np.mean(prev_x) if len(prev_x) > 0 else 0
            to["centroids"].append(centroid)

            if not to["counted"]:
                inside_gate = GATE_LEFT <= centroid[0] <= GATE_RIGHT and GATE_TOP <= centroid[1] <= GATE_BOTTOM

                if direction_x < -5 and not inside_gate and np.mean(prev_x) > GATE_RIGHT:
                    totalLeft += 1
                    to["counted"] = True
                    print(f"[GATE] ID {objectID} passed RIGHT→LEFT | totalLeft={totalLeft}")
                elif direction_x > 5 and not inside_gate and np.mean(prev_x) < GATE_LEFT:
                    totalRight += 1
                    to["counted"] = True
                    print(f"[GATE] ID {objectID} passed LEFT→RIGHT | totalRight={totalRight}")

        trackableObjects[objectID] = to

        cv2.putText(frame, f"ID {objectID}", (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # --- Display counters ---
    info = [("Exited (Right→Left)", totalLeft), ("Entered (Left→Right)", totalRight)]
    for (i, (k, v)) in enumerate(info):
        cv2.putText(frame, f"{k}: {v}", (10, (i * 25) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- Init writer ---
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30, (W, H), True)

    writer.write(frame)
    cv2.imshow("Processing...", cv2.resize(frame, (800, 600)))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("[INFO] Cleaning up...")
writer.release()
vs.release()
cv2.destroyAllWindows()
