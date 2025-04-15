import cv2
import numpy as np
import easyocr
import mss
import time
import os
from datetime import datetime

# Initialize EasyOCR Reader with GPU support
reader = easyocr.Reader(['en'], gpu=True)

# Screen capture tool
sct = mss.mss()
monitor = sct.monitors[1]  # Full primary screen

# Target downscaled resolution
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# File storage setup
save_dir = os.path.dirname(os.path.abspath(__file__))
file_index = 1

# Helper: get next available filename
def get_next_filename():
    global file_index
    while True:
        filename = f"ocr_capture_{file_index:03d}.txt"
        filepath = os.path.join(save_dir, filename)
        if not os.path.exists(filepath):
            return filepath
        file_index += 1

prev_time = 0
print("Press 'q' to exit | Press 'c' to capture OCR text to file")

while True:
    current_time = time.time()

    # Capture full screen
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)

    # Convert BGRA to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Resize to 720p
    frame_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Perform OCR
    results = reader.readtext(rgb_frame, detail=1, paragraph=False)

    # Draw OCR results
    for (bbox, text, prob) in results:
        if prob > 0.5:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            cv2.rectangle(frame_resized, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame_resized, text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # FPS counter
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show screen
    cv2.imshow("Full Screen OCR (720p)", frame_resized)

    # Handle key events
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # 'q' to quit
        break
    elif key == ord('c'):  # 'c' to capture
        captured_words = [text for (_, text, prob) in results if prob > 0.5]
        if captured_words:
            filepath = get_next_filename()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(captured_words))
            print(f"OCR text saved to: {filepath}")
        else:
            print("No confident text detected to save.")

cv2.destroyAllWindows()
