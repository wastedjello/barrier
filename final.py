import cv2
import subprocess
import numpy as np
import pytesseract
import pandas as pd
import re
import serial
import time
import os
from datetime import datetime
import warnings

# ================= CONFIG =================
WIDTH, HEIGHT = 640, 480
FPS = 30

CSV_FILE = "plates.csv"
PHOTO_DIR = "photos"

SERIAL_PORT = "/dev/ttyACM0"   # change if needed
BAUD = 9600

SAMPLE_RATE = 3        # OCR every 3 frames (faster detection)
DEDUP_SECONDS = 5      # avoid duplicate logging
# ==========================================

# Suppress annoying libpng warnings
warnings.filterwarnings("ignore", message=".*iCCP.*")

# Create photos folder if missing
os.makedirs(PHOTO_DIR, exist_ok=True)

# ---------- Arduino ----------
try:
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
    time.sleep(2)
    print(f"🔌 Arduino connected on {SERIAL_PORT}")
except Exception as e:
    print(f"⚠️ Arduino not connected: {e}")
    ser = None

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

def detect_plate_and_ocr(frame):
    # Convert to gray and denoise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Adaptive threshold for screen reading
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
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if w < 80 or h < 25:
                continue

            plate = frame[y:y+h, x:x+w]
            plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

            # Threshold for OCR
            plate_thresh = cv2.adaptiveThreshold(
                plate_gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # OCR
            text = pytesseract.image_to_string(
                plate_thresh,
                config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )
            text = clean_plate_text(text)
            if valid_plate(text):
                return text, (x, y, w, h)

    return None, None

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

print("📷 System running — press Q to quit")

# ---------- Main loop ----------
while True:
    raw = proc.stdout.read(WIDTH * HEIGHT * 3 // 2)
    if len(raw) != WIDTH * HEIGHT * 3 // 2:
        continue

    yuv = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT * 3 // 2, WIDTH))
    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    bbox = None

    if frame_idx % SAMPLE_RATE == 0:
        plate_text, bbox = detect_plate_and_ocr(frame)
        now = time.time()

        if plate_text:
            last_seen = seen.get(plate_text, 0)
            if now - last_seen > DEDUP_SECONDS:
                seen[plate_text] = now

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                fname = f"{PHOTO_DIR}/{plate_text}_{int(now)}.jpg"
                cv2.imwrite(fname, frame)

                records.append({
                    "timestamp": ts,
                    "plate": plate_text,
                    "image": fname
                })

                # Terminal notification
                print(f"📸 PHOTO TAKEN | Plate: {plate_text} | Saved as: {fname}")

                # On-screen notification
                flash_text = f"PHOTO SAVED: {plate_text}"
                flash_until = time.time() + 2

                # Arduino trigger
                if ser:
                    ser.write(b"TRIGGER\n")

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------- Cleanup ----------
proc.terminate()
if ser:
    ser.close()
cv2.destroyAllWindows()

df = pd.DataFrame(records)
df.to_csv(CSV_FILE, index=False)
print(f"📄 Saved {len(df)} entries to {CSV_FILE}")
