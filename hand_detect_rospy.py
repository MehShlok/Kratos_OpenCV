import cv2
import mediapipe as mp
import math
import rospy
from std_msgs.msg import String

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

# Initialize ROS node
rospy.init_node('hand_detection_node')

# Create a publisher for "Hand detected" and "Move" messages
hand_detected_pub = rospy.Publisher('/hand_detected', String, queue_size=10)
move_pub = rospy.Publisher('/move', String, queue_size=10)

cap = cv2.VideoCapture(1)
hands = mphands.Hands()

while not rospy.is_shutdown():
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
            distance = math.sqrt((thumb_tip.x - index_finger_tip.x)**2 + (thumb_tip.y - index_finger_tip.y)**2)

            # Publish "Hand detected" with the approximate distance
            hand_detected_pub.publish(f"Hand detected - Distance: {distance:.2f}")

            # Check if the distance is between 0.5 and 1, and publish "Move"
            if 0.5 <= distance <= 1:
                move_pub.publish("Move")

    cv2.imshow('Handtracker', image)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
