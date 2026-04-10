import cv2
import time
import threading
import os
from datetime import datetime

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"

class cameraDroneThread:
    def __init__(self, rtsp_url):
        print("[INFO]: Connecting drone...")
        self.stream = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # read first frame
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # Thread starting
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        # Looping
        while True:
            if self.stopped:
                self.stream.release()
                return
            # read next frame
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        # Call looping 
        if self.frame is not None:
            return cv2.rotate(self.frame, cv2.ROTATE_90_CLOCKWISE)
        return None

    def stop(self):
        self.stopped = True


def main():
    # URL RTSP drone camera
    RTSP_URL = "rtsp://192.168.1.1:7070/webcam"
    
    # Initialize and start
    camera = cameraDroneThread(RTSP_URL).start()
    time.sleep(2.0)

    prev_time = time.time()
    fps_smoothed = 0

    print("[INFO] Running...")

    while True:
        frame = camera.read()
        
        if frame is None:
            continue

        # Display timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Count FPS
        time_diff = time.time() - prev_time
        if time_diff > 0:
            fps_smoothed = (fps_smoothed * 0.9) + ((1 / time_diff) * 0.1) if fps_smoothed > 0 else (1 / time_diff)
        prev_time = time.time()
        
        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps_smoothed)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show App
        cv2.imshow("Drone E99 Face Recognition and Anti Spoofing", frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()