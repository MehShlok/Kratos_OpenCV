import cv2
import imutils
import rospy
from geometry_msgs.msg import Twist
import numpy as np

# Initialize rospy node
rospy.init_node('human_following_robot')

# Create publisher to control robot movement
cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

# Load the MobileNet SSD model and class labels
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define the class labels and classes to ignore
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Generate random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Initialize robot motion state and human distance
robot_in_motion = False
human_distance = 0

# Define the camera parameters
focal_length = 720  # Focal length of the camera in pixels
person_height = 1.7  # Height of the person in meters

while not rospy.is_shutdown():
    # Reading the video stream
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width=min(600, image.shape[1]))

        # Convert the image to a blob
        blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)

        # Set the blob as input to the network
        net.setInput(blob)

        # Forward pass through the network to get detections
        detections = net.forward()

        # Initialize variables for tracking the closest person
        closest_person = None
        min_distance = float('inf')

        # Calculate the center of the image
        image_center_x = image.shape[1] / 2
        image_center_y = image.shape[0] / 2

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])

            # Filter out detections that are not persons
            if CLASSES[class_id] == 'person' and confidence > 0.2:
                box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (x, y, w, h) = box.astype(int)

                person_center_x = x + w / 2
                person_center_y = y + h / 2

                # Person height in pixels
                person_height_pixels = h

                # Calculate the distance between robot and person using the focal length method
                distance = (person_height * focal_length) / person_height_pixels

                # Check if closest person or not
                if distance < min_distance:
                    min_distance = distance
                    closest_person = (x, y, w, h)

        if closest_person is not None:
            x, y, w, h = closest_person
            closest_person_center_x = x + w / 2
            closest_person_center_y = y + h / 2

            # Calculate the error (difference between person's and image center)
            error = closest_person_center_x - image_center_x

            # Calculate the human distance
            human_distance = min_distance

            # Check if the human is within the desired range
            min_distance_threshold = 1  # Minimum distance in meters
            max_distance_threshold = 4  # Maximum distance in meters

            if (min_distance_threshold <= human_distance <= max_distance_threshold):

                # Define the robot control parameters
                linear_vel = 0.2
                angular_vel = -0.01 * error

                # Create Twist message
                cmd_vel_msg = Twist()
                cmd_vel_msg.linear.x = linear_vel
                cmd_vel_msg.angular.z = angular_vel

                # Draw tracking rectangle around closest person
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Publish Twist message
                cmd_vel_pub.publish(cmd_vel_msg)

                # Set robot motion state to True
                robot_in_motion = True
            
            else:
                # If the human is outside the desired range, stop the robot
                robot_in_motion = False
                cmd_vel_msg = Twist()
                cmd_vel_pub.publish(cmd_vel_msg)
        else:
            # If no person detected and robot was in motion, stop the robot
            if robot_in_motion:
                robot_in_motion = False
                cmd_vel_msg = Twist()
                cmd_vel_pub.publish(cmd_vel_msg)

        # Showing the output Image
        cv2.imshow("Image", image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
