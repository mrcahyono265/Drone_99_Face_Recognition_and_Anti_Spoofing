import cv2
import time
from datetime import datetime

# URL RSTP Drone E99
RTSP_URL = "rtsp://192.168.1.1:7070/webcam"

# Initialize connection with FFMPEG
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
# Configure buffer size (Avoiding Lagging)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("[INFO]: Connecting to the drone's camera...")
time.sleep(2.0)

# Varible to count frames
prev_time = time.time()
fps_smoothed = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR]: Unconnected from the drone's camera. Check the connection")
        break

    # Rotating camera 90 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Adding timestamp to the frame
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_time, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Calculate dan show FPS
    time_diff = time.time() - prev_time
    if time_diff > 0:
        fps_smoothed = (fps_smoothed * 0.9) + ((1 / time_diff) * 0.1) if fps_smoothed > 0 else (1 / time_diff)
    prev_time = time.time()
    cv2.putText(frame, f"FPS: {int(fps_smoothed)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Drone Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()