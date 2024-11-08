#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

def talker():
    # Initialize the node with the name 'talker'
    rospy.init_node('talker', anonymous=True)
    
    # Create a publisher that publishes to the '/chatter' topic
    pub = rospy.Publisher('chatter', String, queue_size=10)
    
    # Set the rate of publishing messages (10 Hz)
    rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        # Create a message
        message = "Hello ROS! Time: %s" % rospy.get_time()
        
        # Log the message to the console
        rospy.loginfo(message)
        
        # Publish the message
        pub.publish(message)
        
        # Sleep to maintain the 10 Hz rate
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
