import cv2
import mediapipe as mp
import math
import rospy
from geometry_msgs.msg import Twist

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

# Initialize ROS
rospy.init_node('hand_gesture_robot_controller')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=2)
move = Twist()

# Initialize the hand gesture detection
cap = cv2.VideoCapture(0)
hands = mphands.Hands()

# Define hand gesture thresholds
forward_threshold = 0.10   # Use an open hand for moving forward
backward_threshold = 0.70  # Use a closed fist for moving backward
rotate_right_threshold = 0.20  # Use a "V" gesture for rotating right
rotate_left_threshold = 0.50   # Use an "L" gesture for rotating left

while True:
    data, image = cap.read()

    # Flip the image
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Storing the results
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks, mphands.HAND_CONNECTIONS)

            # Calculate the distance between thumb tip (landmark 4) and index finger tip (landmark 8)
            thumb_tip = hand_landmarks.landmark[4]
            index_finger_tip = hand_landmarks.landmark[8]
            distance = math.sqrt((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2)

            # Check for hand gestures and control the robot accordingly
            if distance <= forward_threshold:
                # Move forward
                move.linear.x = 0.2
                move.angular.z = 0.0
            elif distance >= backward_threshold:
                # Move backward
                move.linear.x = -0.2
                move.angular.z = 0.0
            elif thumb_tip.x < index_finger_tip.x and distance <= rotate_right_threshold:
                # Rotate right (rotateR)
                move.linear.x = 0.0
                move.angular.z = -0.3  # Adjust the angular speed as needed for your robot
            elif thumb_tip.x > index_finger_tip.x and distance >= rotate_left_threshold:
                # Rotate left (rotateL)
                move.linear.x = 0.0
                move.angular.z = 0.3  # Adjust the angular speed as needed for your robot
            else:
                move.linear.x = 0.0  # Stop the robot
                move.angular.z = 0.0

    cv2.imshow('Handtracker', image)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    # Publish the control commands to the robot
    pub.publish(move)

# When everything is done, release the capture and shut down ROS
cap.release()
cv2.destroyAllWindows()
rospy.signal_shutdown("Shutdown")
