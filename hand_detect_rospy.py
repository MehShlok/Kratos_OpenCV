import cv2
import mediapipe as mp
import math
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

# Initialize ROS node
rospy.init_node('hand_detection_node')

# Create a publisher for "Hand detected" and Twist messages (cmd_vel)
hand_detected_pub = rospy.Publisher('/hand_detected', String, queue_size=10)
cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

# Create a logger for gesture information
gesture_logger = rospy.Publisher('/gesture_info', String, queue_size=10)

cap = cv2.VideoCapture(0)
hands = mphands.Hands()

while not rospy.is_shutdown():
    data, image = cap.read()

    # Flip the image
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Storing the results
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    gesture_info = ""  # Initialize gesture information string

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks, mphands.HAND_CONNECTIONS)

            # Calculate the distance between thumb tip (landmark 4) and index finger tip (landmark 8)
            thumb_tip = hand_landmarks.landmark[4]
            index_finger_tip = hand_landmarks.landmark[8]
            distance = math.sqrt((thumb_tip.x - index_finger_tip.x)**2 + (thumb_tip.y - index_finger_tip.y)**2)

            # Check for gestures to control movement
            twist_msg = Twist()

            # Add tolerance for gesture detection
            if distance <= 0.1:
                twist_msg.linear.x = 0.0  # Stop
                gesture_info = "closed hand (Stop)"
            elif distance >= 0.2:
                twist_msg.linear.x = 0.2  # Forward
                gesture_info = "open hand (Forward)"
            else:
                twist_msg.linear.x = -0.2  # Reverse
                gesture_info = "Thumb left (Reverse)"

            # Publish Twist message to cmd_vel
            cmd_vel_pub.publish(twist_msg)

            # Publish "Hand detected" with the approximate distance
            hand_detected_pub.publish(f"Hand detected - Distance: {distance:.2f}")

    # Log gesture information
    gesture_logger.publish(gesture_info)

    # Print where the robot is going
    print("Robot is:", gesture_info)

    cv2.imshow('Handtracker', image)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
