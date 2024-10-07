#!/usr/bin/env python3

import os
import sys
import rospy
from geometry_msgs.msg import Twist

sys.path.append(os.path.abspath('/home/sim/simulator/utils'))
import constants
from config import load_yaml_file

def main():
    pub = rospy.Publisher('/target', Twist, queue_size=10)
    rospy.init_node('target')
    r = rospy.Rate(10) # 10hz
    target_point = Twist()

    # Load {x,y,z} from the config file
    config = load_yaml_file(constants.merged_config_path)
    x = config['target']['x']
    y = config['target']['y']
    z = config['target']['z']

    while not rospy.is_shutdown():
        target_point.linear.x = x
        target_point.linear.y = y
        target_point.linear.z = z
        pub.publish(target_point)
        r.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass