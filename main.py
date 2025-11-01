from flask import Flask, render_template, Response, jsonify
import cv2, time, numpy as np, mediapipe as mp, threading
from collections import deque
from scipy.spatial import distance as dist

app = Flask(__name__)

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 3
BLINK_RATE_WINDOW = 60.0
MOTION_THRESHOLD = 30000
MOTION_STILL_SECONDS = 15

blink_timestamps = deque()
stress_history = deque(maxlen=5)
ear_consec = 0
total_blinks = 0
low_motion_start = None
prev_gray = None
final_stress = "LOW"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("⚠️ Camera not found. Try index 1 or check permissions.")

lock = threading.Lock()

def eye_aspect_ratio(landmarks, eye_idxs, image_w, image_h):
    pts = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in eye_idxs]
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    if C == 0:
        return 0
    return (A + B) / (2.0 * C)

def process_frame():
    global total_blinks, ear_consec, prev_gray, low_motion_start, final_stress
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        now = time.time()

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            ear_l = eye_aspect_ratio(face_landmarks, LEFT_EYE, w, h)
            ear_r = eye_aspect_ratio(face_landmarks, RIGHT_EYE, w, h)
            ear_val = (ear_l + ear_r) / 2.0
            if ear_val < EAR_THRESHOLD:
                ear_consec += 1
            else:
                if ear_consec >= CONSEC_FRAMES:
                    total_blinks += 1
                    blink_timestamps.append(time.time())
                ear_consec = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21,21), 0)
        if prev_gray is None:
            prev_gray = gray.copy()
            motion_level = 99999
        else:
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_level = np.sum(thresh)
            prev_gray = cv2.addWeighted(prev_gray, 0.5, gray, 0.5, 0)

        if motion_level < MOTION_THRESHOLD:
            if low_motion_start is None:
                low_motion_start = now
        else:
            low_motion_start = None

        low_motion_duration = (now - low_motion_start) if low_motion_start else 0

        while blink_timestamps and (now - blink_timestamps[0]) > BLINK_RATE_WINDOW:
            blink_timestamps.popleft()
        blink_rate_per_min = len(blink_timestamps) * (60.0 / BLINK_RATE_WINDOW)

        cond_blink = blink_rate_per_min > 30
        cond_still = low_motion_duration > MOTION_STILL_SECONDS

        score = cond_blink + cond_still
        if score >= 2:
            stress_level = "HIGH"
        elif score == 1:
            stress_level = "MEDIUM"
        else:
            stress_level = "LOW"

        stress_history.append(stress_level)
        final_stress = max(set(stress_history), key=stress_history.count)

        with lock:
            global_data["blink_rate"] = round(blink_rate_per_min, 2)
            global_data["stress"] = final_stress

        time.sleep(0.05)

global_data = {"blink_rate": 0, "stress": "LOW"}

threading.Thread(target=process_frame, daemon=True).start()

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_data')
def get_data():
    with lock:
        return jsonify(global_data)

if __name__ == "__main__":
    app.run(debug=True)
