#!/usr/local/bin/py310ros
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3, PoseStamped

### ROS Subscriber Callbacks ###
TRACKING_ARRAY_RECEIVED = None
LAST_MSG_TIME = None
CURRENT_ALTITUDE = None  # meters, positive up — from /jaxguam/pose

def fnc_callback(msg):
    global TRACKING_ARRAY_RECEIVED, LAST_MSG_TIME
    TRACKING_ARRAY_RECEIVED = msg
    LAST_MSG_TIME = rospy.Time.now()

def pose_callback(msg):
    global CURRENT_ALTITUDE
    CURRENT_ALTITUDE = msg.pose.position.z

P_gain = 0.05
D_gain = 0.001

P_gain_z = 05.0

# Max correction velocity (m/s) added on top of the base velocity in
# node_vehicle.py — raising these speeds up centering/descent.
MAX_VEL_XY = 1.0
MAX_VEL_Z  = 1

FREQ_LOW_LEVEL = 10

# If no new tracking message has arrived within this many seconds, treat
# the data as stale and stop — prevents the UAV from continuing to move
# on an old bbox once detections/messages stop coming in (e.g. target
# leaves the FOV or the perception pipeline shuts down at trial end).
STALE_TIMEOUT_SEC = 0.5

# Camera image is IMG_SIZE x IMG_SIZE; at altitude h the camera's ground
# footprint is 2h x 2h meters, so meters-per-pixel = 2h / IMG_SIZE.
IMG_SIZE = 640
CTR_X_POS = IMG_SIZE / 2
CTR_Y_POS = IMG_SIZE / 2
AREA_SIZE = 360

# Real-world xy position-error deadband (meters). Above this, correct x/y
# and hold altitude (cmd_vz=0). Within this, stop x/y correction and
# descend directly with z velocity.
XY_CORRECTION_THRESHOLD_M = 4.0

if __name__ == '__main__':

    # rosnode node initialization
    rospy.init_node('controller_node')

    # subscriber init.
    sub = rospy.Subscriber('/yolo_node/sort_mot_predictions', Float32MultiArray, fnc_callback)
    pose_sub = rospy.Subscriber('/jaxguam/pose', PoseStamped, pose_callback)

    # publishers init.
    pub_vel_cmd = rospy.Publisher('/controller_node/vel_cmd', Vector3, queue_size=10)

    # Running rate
    rate = rospy.Rate(FREQ_LOW_LEVEL)

    # msg init.
    vel_cmd_tracking = Vector3()

    # Previous-error state for the derivative term (x/y only, used during
    # the centering phase to converge to zero error without overshoot).
    previous_error_x = 0.0
    previous_error_y = 0.0

    ##############################
    ### Instructions in a loop ###
    ##############################
    while not rospy.is_shutdown():

        is_stale = (LAST_MSG_TIME is None
                    or (rospy.Time.now() - LAST_MSG_TIME).to_sec() > STALE_TIMEOUT_SEC)

        if TRACKING_ARRAY_RECEIVED is not None and not is_stale and CURRENT_ALTITUDE is not None:
            height = TRACKING_ARRAY_RECEIVED.layout.dim[0].size
            width = TRACKING_ARRAY_RECEIVED.layout.dim[1].size
            np_tracking = np.array(TRACKING_ARRAY_RECEIVED.data).reshape((height, width))

            if len(np_tracking) > 0:
                the_obj = np_tracking[-1]
                x1, y1, x2, y2 = the_obj[0:4]

                x_ctr = (x1 + x2) / 2
                y_ctr = (y1 + y2) / 2
                size = (x2 - x1) * (y2 - y1) / 1000

                ### Calculate pixel error ###
                error_x = x_ctr - CTR_X_POS
                error_y = y_ctr - CTR_Y_POS
                error_z = size ** 0.5 - AREA_SIZE ** 0.5

                ### Convert xy pixel error to real-world meters using altitude ###
                meters_per_pixel = (2.0 * CURRENT_ALTITUDE) / IMG_SIZE
                error_x_m = error_x * meters_per_pixel
                error_y_m = error_y * meters_per_pixel

                ### Derivative term (x/y) ###
                derivative_x = error_x - previous_error_x
                derivative_y = error_y - previous_error_y
                previous_error_x = error_x
                previous_error_y = error_y

                if (abs(error_x_m) > XY_CORRECTION_THRESHOLD_M
                        or abs(error_y_m) > XY_CORRECTION_THRESHOLD_M):
                    ### Correction phase: PD converges to zero error, hold altitude ###
                    cmd_vx = P_gain * error_x + D_gain * derivative_x
                    cmd_vy = P_gain * -error_y + D_gain * -derivative_y
                    cmd_vz = 0.0
                else:
                    ### Descent phase: stop x/y correction, descend ###
                    cmd_vx = 0.0
                    cmd_vy = 0.0
                    cmd_vz = P_gain_z * error_z

                ### Clipping ###
                cmd_vx = np.clip(cmd_vx, -MAX_VEL_XY, MAX_VEL_XY)
                cmd_vy = np.clip(cmd_vy, -MAX_VEL_XY, MAX_VEL_XY)
                cmd_vz = np.clip(cmd_vz, -MAX_VEL_Z, MAX_VEL_Z)

                vel_cmd_tracking.y = cmd_vx  # if target is at the right then generate positive cmd_vx
                vel_cmd_tracking.x = cmd_vy  # if target is at the above then generate positive cmd_vy
                vel_cmd_tracking.z = cmd_vz  # if target is small then generate positive cmd_vz

            else:
                vel_cmd_tracking.x = 0
                vel_cmd_tracking.y = 0
                vel_cmd_tracking.z = 0

        else:
            # No message ever received, or the last one is stale (older
            # than STALE_TIMEOUT_SEC) — stop rather than reuse old bbox data.
            if is_stale and LAST_MSG_TIME is not None:
                rospy.logwarn_throttle(
                    1.0,
                    f"  [controller_node] Tracking data stale "
                    f"(> {STALE_TIMEOUT_SEC:.1f}s old) — holding position.")
            vel_cmd_tracking.x = 0
            vel_cmd_tracking.y = 0
            vel_cmd_tracking.z = 0

        ### Publish ###
        pub_vel_cmd.publish(vel_cmd_tracking)

        rate.sleep()
