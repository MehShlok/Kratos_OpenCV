import cv2
import numpy as np
import rover_control_library as rover
import time

rover_control = rover.RoverControl()
cap = cv2.VideoCapture(0) 

# Load the pre-trained OpenCV human detection model (Haar Cascade)
human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

def main():
    fps = 1

    while True:
        start_time = time.time()

        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for human detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect humans in the frame
        humans = human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected humans and calculate the center of the first detected human
        for (x, y, w, h) in humans:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            human_center_x = x + w // 2
            human_center_y = y + h // 2
            break  # Only consider the first detected human

        # Control the rover based on the detected human's position
        if 'human_center_x' in locals():
            if human_center_x < frame.shape[1] // 3:
                rover_control.turn_left()
            elif human_center_x > 2 * frame.shape[1] // 3:
                rover_control.turn_right()
            else:
                rover_control.move_forward()
        else:
            rover_control.stop()

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
    rover_control.stop()

if __name__ == '__main__':
    main()
