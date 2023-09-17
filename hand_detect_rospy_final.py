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
            second_finger_tip = hand_landmarks.landmark[12]
            third_finger_tip = hand_landmarks.landmark[16]
            last_finger_tip = hand_landmarks.landmark[20]
            distance_1 = math.sqrt((thumb_tip.x - index_finger_tip.x)**2 + (thumb_tip.y - index_finger_tip.y)**2)
            distance_2 = math.sqrt((index_finger_tip.x - second_finger_tip.x)**2 + (index_finger_tip.y - second_finger_tip.y)**2)
            distance_3 = math.sqrt((second_finger_tip.x - third_finger_tip.x)**2 + (second_finger_tip.y - third_finger_tip.y)**2)
            distance_4 = math.sqrt((third_finger_tip.x - last_finger_tip.x)**2 + (third_finger_tip.y - last_finger_tip.y)**2)
            # Check for gestures to control movement
            twist_msg = Twist()
            if distance_4 <= 0.1:
                if distance_2 > 0.1 :
                    twist_msg.linear.x = 0.2  # Forward
                    gesture_info = "Peace (Forward)"
                elif distance_3 < 0.1:
                    twist_msg.linear.x = -0.2  # Stop
                    gesture_info = "closed hand (Reverse)"    
                elif distance_2 < 0.1:
                    twist_msg.linear.x = 0.0  # Stop
                    gesture_info = "glued hand(Stop)"
            else:
                if distance_3 <= 0.05:
                    twist_msg.angular.z = 0.2
                    gesture_info = "Thumb Left (Turn left)" 
                elif distance_2 > 0.05:
                    twist_msg.angular.z = -0.2
                    gesture_info = "Turn right (Open hand)"
            # Publish Twist message to cmd_vel
            cmd_vel_pub.publish(twist_msg)

            # Publish "Hand detected" with the approximate distance
            hand_detected_pub.publish(f"Hand detected - Distance: {distance_2:.2f}")

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
