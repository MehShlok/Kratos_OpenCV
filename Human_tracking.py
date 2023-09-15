import cv2
import numpy as np
import time

# Initialize the laptop's camera (use the appropriate camera source, e.g., 0 for built-in camera)
cap = cv2.VideoCapture(2)

# Load the pre-trained OpenCV human detection model (Haar Cascade)
# Download the XML file from: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_fullbody.xml
human_cascade = cv2.CascadeClassifier('/home/ubuntu/Downloads/haarcascade_fullbody.xml')

def main():
    fps = 1

    while True:
        start_time = time.time()

        # Capture a frame from the laptop's camera
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for human detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect humans in the frame
        humans = human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        # Draw rectangles around detected humans and calculate the center of the first detected human
        for (x, y, w, h) in humans:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            human_center_x = x + w // 2
            human_center_y = y + h // 2
            break  # Only consider the first detected human

        # Simulate rover-like actions based on the detected human's position
        if 'human_center_x' in locals():
            print("Human Detected")
            if human_center_x < frame.shape[1] // 2:
                print("Turn left")
            elif human_center_x > frame.shape[1] // 2:
                print("Turn right")
            else:
                print("Move forward")
        else:
            print("Stop")

        # Display the frame with detected humans
        cv2.imshow("Human Detection", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        fps = round(1.0 / (time.time() - start_time), 1)
        print("*********FPS: ", fps, "************")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
