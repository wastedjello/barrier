#!/usr/bin/env python3
import cv2
import subprocess
import numpy as np
import pytesseract
import pandas as pd
import re
import serial
import time
import os
import csv
from datetime import datetime
import warnings
from collections import deque, Counter
from statistics import mean

# ================= CONFIG =================
WIDTH, HEIGHT = 640, 480
FPS = 30

CSV_FILE = "plates.csv"
PHOTO_DIR = "photos"
ROI_DIR = "rois"
FAILED_ROI_DIR = "rois_failed"

SERIAL_PORT = "/dev/ttyACM0"   # change if needed
BAUD = 9600

SAMPLE_RATE = 3        # OCR every 3 frames (faster detection)
DEDUP_SECONDS = 5      # avoid duplicate logging
# Voting parameters
VOTE_BUFFER_LEN = 7
VOTE_IOU_THRESH = 0.45
VOTE_ACCEPT_SCORE = 0.65
VOTE_MIN_CONSENSUS = 3
VOTE_COOLDOWN = 4     # seconds between logs for same plate
# ==========================================

# Suppress annoying libpng warnings
warnings.filterwarnings("ignore", message=".*iCCP.*")

# Create folders if missing
os.makedirs(PHOTO_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)
os.makedirs(FAILED_ROI_DIR, exist_ok=True)

# ---------- Arduino ----------
try:
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
    time.sleep(2)
    print(f"🔌 Arduino connected on {SERIAL_PORT}")
except Exception as e:
    print(f"⚠️ Arduino not connected: {e}")
    ser = None

# ---------- CSV helper (append safely) ----------
CSV_HEADER = ["timestamp", "plate", "image", "roi_image"]
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()

def append_csv_row(row):
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writerow(row)

# ---------- Plate helpers ----------
def clean_plate_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

def valid_plate(text):
    # Only accept likely plates
    return (
        6 <= len(text) <= 10 and
        any(c.isdigit() for c in text) and
        any(c.isalpha() for c in text)
    )

# ---------- Preprocessing ----------
def preprocess_plate(plate_img):
    """CLAHE, gamma, resize, threshold, morphology -> binary image for OCR"""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # gamma correction
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0,256)]).astype("uint8")
    gray = cv2.LUT(gray, table)
    # upscale to help OCR
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # Otsu threshold
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    clean = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return clean

# ---------- OCR with confidence ----------
def tesseract_with_confidence(img):
    """
    Returns (text, avg_confidence 0..1).
    Uses pytesseract.image_to_data to extract confidences.
    """
    # Ensure grayscale or binary input
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT,
                                         config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    except Exception:
        # fallback to simple call
        txt = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return clean_plate_text(txt), 0.0

    texts = []
    confs = []
    for i, txt in enumerate(data.get('text', [])):
        if not txt or txt.strip() == "":
            continue
        conf = data.get('conf', [])[i]
        try:
            conf = float(conf)
        except Exception:
            continue
        # ignore negative confidences
        if conf < 0:
            continue
        texts.append(txt)
        confs.append(max(0.0, min(100.0, conf)) / 100.0)
    if not texts:
        return "", 0.0
    full = "".join(texts)
    return clean_plate_text(full), (mean(confs) if confs else 0.0)

# ---------- IoU helper ----------
def iou(boxA, boxB):
    # boxes: (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0

# ---------- Voting buffer classes ----------
class PlateBuffer:
    def __init__(self, bbox, maxlen=VOTE_BUFFER_LEN):
        self.bbox = bbox  # representative bbox
        self.entries = deque(maxlen=maxlen)  # (text, conf, ts, roi_path, bbox)
        self.last_logged = 0

    def add(self, text, conf, ts, roi_path, bbox):
        self.entries.append((text, conf, ts, roi_path, bbox))
        # update representative bbox to running average
        bx, by, bw, bh = self.bbox
        nx, ny, nw, nh = bbox
        self.bbox = (int((bx + nx)/2), int((by + ny)/2), int((bw + nw)/2), int((bh + nh)/2))

    def voted_text_and_score(self):
        if not self.entries:
            return "", 0.0
        texts = [e[0] for e in self.entries if e[0]]
        confs = [e[1] for e in self.entries if e[0]]
        if not texts:
            return "", 0.0
        maxlen = max(len(t) for t in texts)
        voted_chars = []
        char_conf_list = []
        for i in range(maxlen):
            counter = {}
            confs_for_char = {}
            for (t, c, *_ ) in self.entries:
                if i < len(t):
                    ch = t[i]
                    counter[ch] = counter.get(ch, 0) + 1
                    confs_for_char.setdefault(ch, []).append(c)
            if not counter:
                continue
            best_char = max(counter.items(), key=lambda x: x[1])[0]
            voted_chars.append(best_char)
            char_conf_list.append(mean(confs_for_char.get(best_char, [0.0])))
        voted = "".join(voted_chars)
        avg_char_conf = mean(char_conf_list) if char_conf_list else 0.0
        exact_matches = sum(1 for t in texts if t == voted)
        frac_match = exact_matches / len(texts)
        score = avg_char_conf * (0.6 + 0.4 * frac_match)
        return voted, score

    def recent_consensus_count(self):
        voted, _ = self.voted_text_and_score()
        if not voted:
            return 0
        return sum(1 for e in self.entries if e[0] == voted)

class VoteManager:
    def __init__(self, iou_thresh=VOTE_IOU_THRESH, buffer_len=VOTE_BUFFER_LEN,
                 accept_score=VOTE_ACCEPT_SCORE, min_consensus=VOTE_MIN_CONSENSUS, cooldown=VOTE_COOLDOWN):
        self.buffers = []
        self.iou_thresh = iou_thresh
        self.buffer_len = buffer_len
        self.accept_score = accept_score
        self.min_consensus = min_consensus
        self.cooldown = cooldown

    def add_detection(self, text, conf, bbox, roi_path):
        ts = time.time()
        for buf in self.buffers:
            if iou(buf.bbox, bbox) >= self.iou_thresh:
                buf.add(text, conf, ts, roi_path, bbox)
                return buf
        nb = PlateBuffer(bbox, maxlen=self.buffer_len)
        nb.add(text, conf, ts, roi_path, bbox)
        self.buffers.append(nb)
        now = time.time()
        # prune buffers with no recent entries
        self.buffers = [b for b in self.buffers if (not b.entries) or (now - b.entries[-1][2] < 6)]
        return nb

    def check_accept(self, buf):
        voted, score = buf.voted_text_and_score()
        if not voted:
            return None, None
        consensus = buf.recent_consensus_count()
        now = time.time()
        if (score >= self.accept_score and consensus >= self.min_consensus) or (consensus >= max(4, self.min_consensus+1)):
            if now - buf.last_logged > self.cooldown:
                buf.last_logged = now
                for e in reversed(buf.entries):
                    if e[0] == voted and e[3]:
                        return voted, e[3]
                return voted, buf.entries[-1][3]
        return None, None

vote_manager = VoteManager()

# ---------- Detection function (returns text, bbox, roi_path, conf) ----------
def detect_plate_and_ocr(frame):
    # Convert to gray and denoise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Adaptive threshold for edge detection
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Edge detection
    edged = cv2.Canny(thresh, 30, 200)

    # Contours
    contours, _ = cv2.findContours(
        edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if w < 80 or h < 25:
                continue

            # Expand ROI slightly
            pad_x = int(w * 0.05)
            pad_y = int(h * 0.10)
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(frame.shape[1], x + w + pad_x)
            y1 = min(frame.shape[0], y + h + pad_y)

            plate = frame[y0:y1, x0:x1]
            proc = preprocess_plate(plate)

            # OCR with confidence
            text, conf = tesseract_with_confidence(proc)

            ts = int(time.time())
            roi_fname = f"{ROI_DIR}/roi_{ts}_{x0}_{y0}.jpg"
            cv2.imwrite(roi_fname, plate)

            if valid_plate(text):
                return text, (x0, y0, x1 - x0, y1 - y0), roi_fname, conf

            failed_fname = f"{FAILED_ROI_DIR}/failed_{ts}_{x0}_{y0}.jpg"
            cv2.imwrite(failed_fname, plate)

    return None, None, None, 0.0

# ---------- rpicam ----------
cmd = [
    "rpicam-vid",
    "--codec", "yuv420",
    "--inline",
    "--nopreview",
    "--timeout", "0",
    "--width", str(WIDTH),
    "--height", str(HEIGHT),
    "--framerate", str(FPS),
    "-o", "-"
]

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    bufsize=WIDTH * HEIGHT * 3
)

seen = {}
records = []
frame_idx = 0

flash_text = ""
flash_until = 0

print("📷 System running — press Q to quit, S to save manual frame")

# ---------- Main loop ----------
try:
    while True:
        raw = proc.stdout.read(WIDTH * HEIGHT * 3 // 2)
        if len(raw) != WIDTH * HEIGHT * 3 // 2:
            time.sleep(0.005)
            continue

        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT * 3 // 2, WIDTH))
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        bbox = None

        if frame_idx % SAMPLE_RATE == 0:
            plate_text, bbox, roi_path, conf = detect_plate_and_ocr(frame)
            now = time.time()

            if plate_text and bbox:
                # Add to voting manager
                buf = vote_manager.add_detection(plate_text, conf, bbox, roi_path)
                accepted_text, accepted_roi = vote_manager.check_accept(buf)
                if accepted_text:
                    last_seen = seen.get(accepted_text, 0)
                    if now - last_seen > DEDUP_SECONDS:
                        seen[accepted_text] = now

                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        fname = f"{PHOTO_DIR}/{accepted_text}_{int(now)}.jpg"
                        cv2.imwrite(fname, frame)

                        row = {
                            "timestamp": ts,
                            "plate": accepted_text,
                            "image": fname,
                            "roi_image": accepted_roi or ""
                        }
                        append_csv_row(row)
                        records.append(row)

                        # Terminal notification
                        print(f"✅ ACCEPTED | Plate: {accepted_text} | Saved as: {fname}")

                        # On-screen notification
                        flash_text = f"PHOTO SAVED: {accepted_text}"
                        flash_until = time.time() + 2

                        # Arduino trigger
                        if ser:
                            try:
                                ser.write(b"TRIGGER\n")
                            except Exception:
                                pass

        # Draw bounding box
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Flash notification
        if time.time() < flash_until:
            cv2.putText(
                frame,
                flash_text,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        cv2.imshow("License Plate Camera", frame)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = int(time.time())
            manual_fname = f"{PHOTO_DIR}/manual_{ts}.jpg"
            cv2.imwrite(manual_fname, frame)
            print(f"💾 Manual save: {manual_fname}")

finally:
    # ---------- Cleanup ----------
    proc.terminate()
    if ser:
        ser.close()
    cv2.destroyAllWindows()

    # Save records to CSV (redundant with append but keeps a full DataFrame)
    try:
        df = pd.DataFrame(records)
        if not df.empty:
            df.to_csv(CSV_FILE, index=False, mode='a', header=False)
    except Exception:
        pass

    print(f"📄 Done. CSV saved at {CSV_FILE}")
