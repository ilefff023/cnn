import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import time
import sqlite3
import threading
import queue
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import h5py
import json
import re

IMG_SIZE = 80 

# ---------------- MODEL LOADER ----------------
MODEL_PATH     = 'best_model.h5'
NEW_MODEL_PATH = 'best_model.keras'

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def load_model_safe():
    if os.path.exists(NEW_MODEL_PATH):
        try:
            m = keras.models.load_model(NEW_MODEL_PATH, compile=False)
            print("Loaded best_model.keras successfully.")
            return m
        except Exception as e:
            print(f"keras load failed: {e}")
    if os.path.exists(MODEL_PATH):
        try:
            m = keras.models.load_model(MODEL_PATH, compile=False)
            print("Loaded .h5 successfully.")
            return m
        except Exception as e:
            print(f"h5 load failed: {e}")
        try:
            m = build_model()
            m.load_weights(MODEL_PATH)
            print("Loaded weights into rebuilt model.")
            return m
        except Exception as e:
            print(f"weights load failed: {e}")
        try:
            with h5py.File(MODEL_PATH, 'r') as f:
                raw = f.attrs.get('model_config', None)
            if raw is None:
                raise ValueError("No model_config in h5")
            cfg_str = raw if isinstance(raw, str) else raw.decode('utf-8')
            cfg_str = re.sub(r'"batch_input_shape":\s*\[[^\]]*\],?\s*', '', cfg_str)
            cfg_str = re.sub(r'"batch_shape":\s*\[[^\]]*\],?\s*', '', cfg_str)
            cfg = json.loads(cfg_str)
            m = keras.models.model_from_config(cfg)
            m.load_weights(MODEL_PATH)
            print("Patched h5 config loaded.")
            return m
        except Exception as e:
            print(f"patch load failed: {e}")
    print("Could not load model. Current folder:", os.getcwd())
    exit(1)

model = load_model_safe()

# ---- Warm up model so first real frame isn't slow ----
_dummy = np.zeros((2, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
model.predict(_dummy, verbose=0)
print("Model warmed up.")

OPEN_THRESHOLD = 0.50
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))  # create once, reuse

# ---------------- DATABASE ----------------
conn   = sqlite3.connect("eye_data.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS eye_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    eye_state TEXT,
    blink INTEGER,
    blink_rate REAL
)
""")
conn.commit()

# ---------------- CASCADES ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---------------- EYE CROP ----------------
def get_eye_crops(face_gray, face_color):
    fh, fw = face_gray.shape[:2]
    y1, y2 = int(fh*0.18), int(fh*0.52)
    lx1, lx2 = int(fw*0.04), int(fw*0.46)
    rx1, rx2 = int(fw*0.54), int(fw*0.96)
    return (
        face_gray[y1:y2, lx1:lx2],
        face_gray[y1:y2, rx1:rx2],
        {'left':(lx1,y1,lx2-lx1,y2-y1), 'right':(rx1,y1,rx2-rx1,y2-y1)}
    )

def preprocess(eye_gray):
    if eye_gray is None or eye_gray.size==0 or eye_gray.shape[0]<5 or eye_gray.shape[1]<5:
        return None
    eye = cv2.resize(eye_gray, (IMG_SIZE, IMG_SIZE))
    eye = CLAHE.apply(eye)
    eye = eye.astype(np.float32) / 255.0
    return eye.reshape(IMG_SIZE, IMG_SIZE, 1)

# ================================================================== #
#  INFERENCE THREAD
#  Runs model.predict in background. Main loop sends eye crops via
#  a queue and reads results from result_queue — zero blocking.
# ================================================================== #
infer_queue  = queue.Queue(maxsize=1)   # only keep latest frame
result_queue = queue.Queue(maxsize=1)

def inference_worker():
    while True:
        item = infer_queue.get()
        if item is None:
            break
        eyes = [preprocess(e) for e in item]
        eyes = [e for e in eyes if e is not None]
        if eyes:
            batch = np.stack(eyes, axis=0)          # shape (N,80,80,1)
            preds = model.predict(batch, verbose=0)  # ONE call for both eyes
            avg   = float(np.mean(preds))
        else:
            avg = None
        # replace old result if not consumed
        if not result_queue.empty():
            try: result_queue.get_nowait()
            except: pass
        result_queue.put(avg)

worker = threading.Thread(target=inference_worker, daemon=True)
worker.start()

# ---------------- FACE DETECTION THREAD ----------------
face_queue   = queue.Queue(maxsize=1)
face_result  = queue.Queue(maxsize=1)

def face_worker():
    while True:
        gray_eq = face_queue.get()
        if gray_eq is None:
            break
        faces = face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(100,100)
        )
        result = max(faces, key=lambda r: r[2]*r[3]) if len(faces)>0 else None
        if not face_result.empty():
            try: face_result.get_nowait()
            except: pass
        face_result.put(result)

fworker = threading.Thread(target=face_worker, daemon=True)
fworker.start()

# ---------------- CALIBRATION ----------------
def calibrate(cap):
    global OPEN_THRESHOLD
    print("\n=== CALIBRATION ===")
    print("Eyes OPEN — press SPACE when ready...")
    last_face = None

    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame   = cv2.flip(frame, 1)
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        faces   = face_cascade.detectMultiScale(gray_eq, 1.1, 5, minSize=(100,100))
        if len(faces)>0:
            last_face = max(faces, key=lambda r: r[2]*r[3])
        msg = "Face found! Press SPACE" if last_face is not None else "No face..."
        cv2.putText(frame, msg, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, "CALIBRATION - EYES OPEN", (20,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("FINAL SYSTEM", frame)
        key = cv2.waitKey(1)&0xFF
        if key==ord(' ') and last_face is not None: break
        if key==ord('q'): return

    def collect(cap, label, color, duration=2.0):
        scores=[]; t0=time.time()
        while time.time()-t0 < duration:
            ret,frame=cap.read()
            if not ret: continue
            frame=cv2.flip(frame,1)
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face_cascade.detectMultiScale(cv2.equalizeHist(gray),1.1,5,minSize=(100,100))
            if len(faces)>0:
                fx,fy,fw,fh=max(faces,key=lambda r:r[2]*r[3])
                lg,rg,_=get_eye_crops(gray[fy:fy+fh,fx:fx+fw], frame[fy:fy+fh,fx:fx+fw])
                eyes=[preprocess(e) for e in [lg,rg]]
                eyes=[e for e in eyes if e is not None]
                if eyes:
                    batch=np.stack(eyes,axis=0)
                    preds=model.predict(batch,verbose=0)
                    scores.append(float(np.mean(preds)))
            remaining=duration-(time.time()-t0)
            cv2.putText(frame,f"{label}  {remaining:.1f}s",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)
            cv2.imshow("FINAL SYSTEM",frame)
            cv2.waitKey(1)
        return scores

    open_scores = collect(cap, "KEEP EYES OPEN", (0,255,0))

    print("Now CLOSE your eyes — press SPACE...")
    while True:
        ret,frame=cap.read()
        if not ret: continue
        frame=cv2.flip(frame,1)
        cv2.putText(frame,"CLOSE eyes then press SPACE",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,165,255),2)
        cv2.imshow("FINAL SYSTEM",frame)
        key=cv2.waitKey(1)&0xFF
        if key==ord(' '): break
        if key==ord('q'): return

    closed_scores = collect(cap, "KEEP EYES CLOSED", (0,0,255))

    if open_scores and closed_scores:
        om = float(np.mean(open_scores))
        cm = float(np.mean(closed_scores))
        OPEN_THRESHOLD = round((om+cm)/2.0, 3)
        print(f"Open={om:.3f}  Closed={cm:.3f}  => Threshold={OPEN_THRESHOLD}")
    else:
        print("Calibration failed — using default 0.5")
    print("=== CALIBRATION DONE ===\n")

# ---------------- VARIABLES ----------------
blink_count     = 0
start_time      = time.time()
last_blink_time = 0.0
BLINK_COOLDOWN  = 0.25
MIN_BLINK_TIME  = 0.05
eye_closed      = False
blink_start     = 0.0
pred_buffer     = []
SMOOTH_SIZE     = 4
last_db_write   = time.time()
DB_INTERVAL     = 0.5
last_face       = None
last_raw        = None   # last inference result
frame_count     = 0
FACE_SKIP       = 4

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
if not cap.isOpened():
    print("Camera not found!"); exit(1)

calibrate(cap)
start_time = time.time()
label      = "No Face"
print("SYSTEM RUNNING... Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame                = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    gray_full            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq              = cv2.equalizeHist(gray_full)
    frame_count         += 1
    blink_event          = 0
    eye_found            = False

    # ---- send frame to face worker every FACE_SKIP frames ----
    if frame_count % FACE_SKIP == 0:
        if face_queue.empty():
            face_queue.put(gray_eq)

    # ---- read face result if available ----
    try:
        f = face_result.get_nowait()
        if f is not None:
            last_face = f
    except queue.Empty:
        pass

    if last_face is not None:
        fx,fy,fw,fh = last_face
        fx=max(0,fx); fy=max(0,fy)
        fw=min(fw,frame_w-fx); fh=min(fh,frame_h-fy)

        face_gray  = gray_full[fy:fy+fh, fx:fx+fw]
        face_color = frame[fy:fy+fh, fx:fx+fw]

        lg, rg, boxes = get_eye_crops(face_gray, face_color)

        # ---- send eyes to inference worker ----
        if infer_queue.empty():
            infer_queue.put([lg, rg])

        # ---- read latest inference result ----
        try:
            last_raw = result_queue.get_nowait()
        except queue.Empty:
            pass

        if last_raw is not None:
            eye_found = True
            pred_buffer.append(last_raw)
            if len(pred_buffer) > SMOOTH_SIZE:
                pred_buffer.pop(0)
            avg_pred = float(np.mean(pred_buffer))
            label    = "Open" if avg_pred >= OPEN_THRESHOLD else "Closed"

            eye_color = (0,255,0) if label=="Open" else (0,0,255)
            for side,(bx,by,bw,bh) in boxes.items():
                cv2.rectangle(frame,(fx+bx,fy+by),(fx+bx+bw,fy+by+bh),eye_color,1)

            cv2.putText(frame, f"Score:{last_raw:.2f} T:{OPEN_THRESHOLD}",
                        (20,frame_h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

        cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(255,200,0),2)

    # ---------------- BLINK DETECTION ----------------
    current_time = time.time()
    if eye_found:
        if label=="Closed":
            if not eye_closed:
                eye_closed=True; blink_start=current_time
        else:
            if eye_closed:
                duration=current_time-blink_start
                if duration>MIN_BLINK_TIME and (current_time-last_blink_time)>BLINK_COOLDOWN:
                    blink_count+=1; blink_event=1; last_blink_time=current_time
                eye_closed=False
    else:
        eye_closed=False; pred_buffer.clear(); last_face=None

    elapsed    = time.time()-start_time
    blink_rate = (blink_count/elapsed)*60.0 if elapsed>0 else 0.0

    # ---------------- DATABASE ----------------
    if time.time()-last_db_write > DB_INTERVAL:
        cursor.execute(
            "INSERT INTO eye_tracking (timestamp,eye_state,blink,blink_rate) VALUES (?,?,?,?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), label, blink_event, round(blink_rate,2))
        )
        conn.commit()
        last_db_write=time.time()

    # ---------------- DISPLAY ----------------
    state_color=(0,255,0) if label=="Open" else (0,0,255) if label=="Closed" else (180,180,180)
    cv2.putText(frame,f"State: {label}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.9,state_color,2)
    cv2.putText(frame,f"Blinks: {blink_count} | Rate: {int(blink_rate)}/min",(20,80),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

    cv2.imshow("FINAL SYSTEM", frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

# cleanup
infer_queue.put(None)
face_queue.put(None)
cap.release()
conn.close()
cv2.destroyAllWindows()