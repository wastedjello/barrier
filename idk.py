from picamera2 import Picamera2
import cv2
import pytesseract
import time
import csv
import re
from datetime import datetime

CSV_FILE = "plates.csv"
COOLDOWN = 20
MIN_LEN = 5

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format":"RGB888","size":(1280,720)})
picam2.configure(config)
picam2.start()
time.sleep(2)

last_seen = {}

def clean(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def log_plate(plate):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([ts, plate])
    print(f"[LOGGED] {plate}")

# create csv header if needed
try:
    with open(CSV_FILE, 'x', newline='') as f:
        csv.writer(f).writerow(["time","plate"])
except FileExistsError:
    pass

print("License plate scanner running...")

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray,11,17,17)
        gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

        text = pytesseract.image_to_string(
            gray,
            config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        for raw in text.split():
            plate = clean(raw)
            if len(plate)<MIN_LEN:
                continue
            now = time.time()
            if plate not in last_seen or now - last_seen[plate] > COOLDOWN:
                last_seen[plate] = now
                log_plate(plate)

        time.sleep(1)

except KeyboardInterrupt:
    print("Stopped")

finally:
    picam2.stop()
