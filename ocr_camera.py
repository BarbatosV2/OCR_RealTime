import cv2
import easyocr
import time

# Initialize EasyOCR Reader with GPU support
reader = easyocr.Reader(['en'], gpu=True)

# Open webcam (0 = default webcam)
cap = cv2.VideoCapture(0)

# Set webcam resolution to 720p (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

print("Press 'q' to exit.")

# Variables for FPS calculation
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame.")
        break

    # Start timer for FPS
    current_time = time.time()

    # Convert image to RGB (EasyOCR uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform OCR
    results = reader.readtext(rgb_frame, detail=1, paragraph=False)

    # Loop over results and draw bounding boxes + text
    for (bbox, text, prob) in results:
        if prob > 0.5:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # FPS calculation
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display FPS on screen
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show the annotated frame
    cv2.imshow('EasyOCR Live', frame)

    # Break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
