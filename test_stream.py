import cv2
import time
import requests

SERVER_URL = "http://127.0.0.1:8000/predict"   # change to your server IP
CLIENT_ID = "python-tester"

def main():
    cap = cv2.VideoCapture(0)  # webcam
    if not cap.isOpened():
        print("Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed.")
            break

        # encode as JPEG
        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            print("JPEG encode failed.")
            continue

        files = {
            "frame": ("frame.jpg", jpeg.tobytes(), "image/jpeg")
        }

        data = {
            "client_id": CLIENT_ID
        }

        try:
            response = requests.post(SERVER_URL, files=files, data=data, timeout=5)
            print(response.json())
        except Exception as e:
            print("Error:", e)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.1)  # simulate ~10 FPS

    cap.release()

if __name__ == "__main__":
    main()