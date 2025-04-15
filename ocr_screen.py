import cv2
import numpy as np
import easyocr
import mss
import time

# Initialize EasyOCR Reader with GPU support
reader = easyocr.Reader(['en'], gpu=True)

# Screen capture tool
sct = mss.mss()
monitor = sct.monitors[1]  # Full primary screen

# Target resolution
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

prev_time = 0

print("Press 'q' to quit.")

while True:
    current_time = time.time()

    # Capture full screen
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)

    # Convert BGRA to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Resize to 720p (scaling full screen to lower resolution)
    frame_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    # Convert to RGB for OCR
    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Perform OCR
    results = reader.readtext(rgb_frame, detail=1, paragraph=False)

    # Draw results
    for (bbox, text, prob) in results:
        if prob > 0.5:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            cv2.rectangle(frame_resized, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame_resized, text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # FPS
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show result
    cv2.imshow("Full Screen OCR (720p)", frame_resized)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
