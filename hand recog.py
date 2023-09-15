import rospy
from geometry_msgs.msg import Twist
import cv2
import numpy as np

# Initialize ROS node and publisher for cmd_vel topic
rospy.init_node("hand_gesture_robot_control")
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
rate = rospy.Rate(10)  # Publish rate in Hz

# Function to initialize the video capture
def init_camera():
    return cv2.VideoCapture(0)  # 0 corresponds to the default camera

# Function to detect and track the hand
def detect_hand(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to create a binary mask
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter the largest contour as the hand
    if contours:
        hand_contour = max(contours, key=cv2.contourArea)
        return hand_contour
    else:
        return None

# Function to recognize hand gestures
def recognize_gesture(hand_contour):
    # Initialize a convex hull
    hull = cv2.convexHull(hand_contour)

    # Find the center of mass of the hand
    M = cv2.moments(hand_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    # Calculate the distance between the fingertips and the center of mass
    fingertips = []
    for point in hull:
        if point[0][0] < cx:
            fingertips.append(tuple(point[0]))

    # Count the number of fingertips
    num_fingertips = len(fingertips)

    # Define gesture based on the number of fingertips and orientation
    if num_fingertips == 0:
        return "Closed Hand"
    elif num_fingertips >= 4:
        return "Thumbs Up"
    elif num_fingertips == 5 and cx > 300:  # Assuming right-handed user
        return "Thumbs Down"
    else:
        return "Open Hand"

# Initialize the camera
cap = init_camera()

while not rospy.is_shutdown():
    # Read a frame from the camera
    ret, frame = cap.read()

    # Detect and track the hand
    hand_contour = detect_hand(frame)

    if hand_contour is not None:
        # Recognize hand gestures
        gesture = recognize_gesture(hand_contour)

        # Publish velocity commands based on gesture
        twist = Twist()
        if gesture == "Closed Hand":
            # Stop the robot
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        elif gesture == "Open Hand":
            # Move the robot forward
            twist.linear.x = 0.2  # Adjust the linear velocity as needed
            twist.angular.z = 0.0
        elif gesture == "Thumbs Up":
            # Turn the robot right
            twist.linear.x = 0.0
            twist.angular.z = -0.2  # Adjust the angular velocity as needed
        elif gesture == "Thumbs Down":
            # Turn the robot left
            twist.linear.x = 0.0
            twist.angular.z = 0.2  # Adjust the angular velocity as needed

        # Publish the velocity command
        pub.publish(twist)
        rate.sleep()

    # Display the recognized gesture on the frame
    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the processed frame
    cv2.imshow("Hand Gesture Control", frame)

    # Check for key press to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
