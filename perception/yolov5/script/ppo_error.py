#!/usr/bin/env python3
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3

### ROS Subscriber Callback ###
TRACKING_ARRAY_RECEIVED = None


def fnc_callback(msg):
    global TRACKING_ARRAY_RECEIVED
    TRACKING_ARRAY_RECEIVED = msg


TARGET_CLASS_INDEX = 2
FREQ_LOW_LEVEL = 10

if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node("error_publisher_node")

    # Subscriber for YOLO predictions
    sub = rospy.Subscriber("/yolo_node/sort_mot_predictions", Float32MultiArray, fnc_callback)

    # Publisher for errors
    pub_error = rospy.Publisher("/controller_node/error", Vector3, queue_size=10)

    # Running rate
    rate = rospy.Rate(FREQ_LOW_LEVEL)

    while not rospy.is_shutdown():
        if TRACKING_ARRAY_RECEIVED is not None:
            height = TRACKING_ARRAY_RECEIVED.layout.dim[0].size
            width = TRACKING_ARRAY_RECEIVED.layout.dim[1].size
            np_tracking = np.array(TRACKING_ARRAY_RECEIVED.data).reshape((height, width))

            if len(np_tracking) > 0:
                the_obj = np_tracking[-1]
                x1, y1, x2, y2 = the_obj[0:4]

                # Calculate center and size
                x_ctr = (x1 + x2) / 2
                y_ctr = (y1 + y2) / 2
                size = (x2 - x1) * (y2 - y1) / 1000

                # Desired position and size
                CTR_X_POS = 224
                CTR_Y_POS = 224
                AREA_SIZE = 50

                # Calculate errors
                error_x = x_ctr - CTR_X_POS
                error_y = y_ctr - CTR_Y_POS
                error_z = size ** 0.5 - (AREA_SIZE) ** 0.5

                # Publish errors
                error_msg = Vector3()
                error_msg.x = error_x
                error_msg.y = error_y
                error_msg.z = error_z
                pub_error.publish(error_msg)
            else:
                # Publish zero errors if no object detected
                error_msg = Vector3(0, 0, 0)
                pub_error.publish(error_msg)

        rate.sleep()
