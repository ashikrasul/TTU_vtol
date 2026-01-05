#!/usr/local/bin/py310ros

import rospy
from geometry_msgs.msg import PoseStamped, Twist
import logging
from utils.vehicle import Vehicle_Node  # Import the parent class

class SimpleVehicleNode(Vehicle_Node):
    def __init__(self, config):
        # Call the parent class constructor, which initializes the node and publishers
        super(SimpleVehicleNode, self).__init__(config)
        
        # Log the initialization (this leverages already initialized components)
        rospy.loginfo("SimpleVehicleNode initialized successfully.")
        
    def main(self):
        rospy.loginfo("Node and topics created. No messages will be published.")
        rospy.spin()  # Keep the node running without publishing

if __name__ == "__main__":
    # Example config dictionary for initialization
    config = {
        'ego_vehicle': {
            'type': 'jaxguam',
            'location': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0}
        }
    }

    # Configure basic logging
    logging.basicConfig(level=logging.INFO)

    try:
        node = SimpleVehicleNode(config)
        node.main()
    except rospy.ROSInterruptException:
        pass
