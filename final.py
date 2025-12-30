from picamera2 import Picamera2
import cv2
import pytesseract
import re
import time
import csv
from datetime import datetime

CSV_FILE = "plates.csv"
COOLDOWN_SECONDS = 20

# ------------------ FILTERS ------------------

def clean_plate(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def is_valid_plate(text):
    """
    Indian license plate formats:
    MH12AB1234
    KA03CD9988
    DL01AX9999
    """
    if not (re.search(r'[A-Z]', text) and re.search(r'\d', text)):
        return False

    if not (8 <= len(text) <= 11):
        return False

    pattern = r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{3,4}$'
    return re.match(pattern, text) is not None

# ------------------ CSV ------------------

def init_csv():
    try:
        with open(CSV_FILE, "x", newline="") as f:
            csv.writer(f).writerow(["timestamp", "plate"])
    except FileExistsError:
        pass

def log_plate(plate):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ts, plate])
    print(f"[LOGGED] {plate}")

# ------------------ OCR ------------------

def detect_plate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)

            # Plate shape filter
            aspect_ratio = w / float(h)
            if aspect_ratio < 2 or aspect_ratio > 6:
                continue
            if w < 120 or h < 30:
                continue

            plate_roi = frame[y:y+h, x:x+w]
            gray_roi = cv2.cvtColor(plate_roi, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(
                gray_roi, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            text = pytesseract.image_to_string(
                thresh,
                config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            text = clean_plate(text)

            if is_valid_plate(text):
                return text, (x, y, w, h)

    return None, None

# ------------------ MAIN ------------------

def main():
    init_csv()

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    seen = {}

    print("📷 License plate scanner running (press Q to quit)")

    while True:
        frame = picam2.capture_array()
        plate, bbox = detect_plate(frame)

        if plate:
            now = time.time()
            if now - seen.get(plate, 0) >= COOLDOWN_SECONDS:
                seen[plate] = now
                log_plate(plate)

            x, y, w, h = bbox
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, plate, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("License Plate Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
