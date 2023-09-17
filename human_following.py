import cv2
import imutils
import rospy
import math
from geometry_msgs.msg import Twist

# Initialize rospy node
rospy.init_node('human_following_robot')

# Create publisher to control robot movement
cmd_vel_pub = rospy.Publisher("/cmd_vel",Twist,queue_size=10)

# Initializing the HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize video capture
cap = cv2.VideoCapture(2)

while not rospy.is_shutdown():
	# Reading the video stream
	ret, image = cap.read()
	if ret:
		image = imutils.resize(image, width=min(600, image.shape[1]))

		# Detecting all the regions in the Image that has a person inside it
		(regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)

		if len(regions) > 0:
			# Initialize variables for tracking the closest person
			closest_person = None
			min_distance = float('inf')

			# Calculate the center of the image
			image_center_x = image.shape[1] / 2
			image_center_y = image.shape[0] / 2

			for region in regions:
				x, y, w, h = region
				person_center_x = x + w / 2
				person_center_y = y + h / 2

				# Calculate distance between robot and person
				distance = math.sqrt(pow((person_center_x - image_center_x),2) + pow((person_center_y - image_center_y),2))

				# Check if closest person or not 
				if distance < min_distance:
					min_distance = distance
					closest_person = region
				
			if closest_person is not None:
				x, y, w, h = closest_person
				closest_person_center_x = x + w / 2
				closest_person_center_y = y + h / 2

				# Calculate the error (difference between person's and image center)
				error = closest_person_center_x - image_center_x

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

		# Showing the output Image
		cv2.imshow("Image", image)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()
