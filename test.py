from picamera2 import Picamera2
import cv2
import time
import csv
import re
from datetime import datetime
import easyocr

# ---------------- SETTINGS ----------------
CSV_FILE = "plates.csv"
COOLDOWN = 20  # seconds
MIN_PLATE_LEN = 5
# ------------------------------------------

reader = easyocr.Reader(['en'], gpu=False)
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

time.sleep(2)

last_seen = {}

def clean(text):
    text = text.upper()
    return re.sub(r'[^A-Z0-9]', '', text)

def log_plate(plate):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([now, plate])
    print(f"[LOGGED] {plate}")

# Create CSV if missing
try:
    with open(CSV_FILE, 'x', newline='') as f:
        csv.writer(f).writerow(["time", "plate"])
except FileExistsError:
    pass

print("🚗 License Plate OCR running (Ctrl+C to stop)")

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        results = reader.readtext(gray)

        for _, text, conf in results:
            plate = clean(text)

            if len(plate) < MIN_PLATE_LEN:
                continue

            now = time.time()
            if plate not in last_seen or now - last_seen[plate] > COOLDOWN:
                last_seen[plate] = now
                log_plate(plate)

        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    picam2.stop()

