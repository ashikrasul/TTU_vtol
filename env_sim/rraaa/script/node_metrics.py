#!/usr/bin/env python3

import os
import sys
import math
import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, Twist

sys.path.append(os.path.abspath('/home/sim/simulator/utils'))
from config import load_yaml_file
import constants

class MetricTracker:
    def __init__(self, configs) -> None:
        self.configs = configs

        # Initialize the ROS node
        rospy.init_node("metric_tracker")
        
        # Pose & target pose
        self.pose = None
        self.target_pose = None
        self.target_reached_pub = rospy.Publisher('/target_reached', Bool, queue_size=10)
        self.pose_threshold = 1e-1 # If the distance between the target pose and the current pose is less than this number in meters, then the target is reached.
        self.pose_sub = rospy.Subscriber(f"/{configs['ego_vehicle']['type']}/pose", PoseStamped, self.pose_callback)
        self.target_pose_sub = rospy.Subscriber(f"/target/pose", Twist, self.target_pose_callback)

    def pose_callback(self, msg):
        # Record the current pose
        self.pose = msg
        
        # Check if the target point has been reached
        if self.pose_is_within_threshold():
            reached = True
        else:
            reached = False

        # Publish the data related to reaching the target point
        self.target_reached_pub.publish(reached)

    def target_pose_callback(self, msg):
        self.target_pose = msg

    def pose_is_within_threshold(self) -> bool:
        # Check that both current and target poses are not None
        if self.pose is None or self.target_pose is None:
            return False

        # Current location
        curr_x = self.pose.pose.position.x
        curr_y = self.pose.pose.position.y
        curr_z = self.pose.pose.position.z

        # Target location
        targ_x = self.target_pose.linear.x
        targ_y = self.target_pose.linear.y
        targ_z = self.target_pose.linear.z

        # Find the distance
        dist = math.sqrt((curr_x - targ_x) ** 2 + (curr_y - targ_y) ** 2 + (curr_z - targ_z) ** 2)

        if dist <= self.pose_threshold:
            return True
        else:
            return False

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    # Extract the configs
    configs = load_yaml_file(constants.merged_config_path)

    # Initialize the metric tracker
    metric_tracker = MetricTracker(configs)

    # Run the metric tracker
    metric_tracker.run()