#!/usr/bin/env python3

import math
import time
import rospy
import heapq
import numpy as np
from typing import List
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, PoseStamped

from utils.config import load_yaml_file
from utils import constants


# A* algorithm for 3D space
def astar(start, goal, grid_size=50, plot_path=False):
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def neighbors(node):
        x, y, z = node
        # Move in all 6 directions: left, right, up, down, forward, backward
        directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size:
                yield (nx, ny, nz)

    # Scale the coordinates to fit the grid
    def scale_coordinates(coords, start, goal, grid_size):
        start, goal = np.array(start), np.array(goal)
        scale_factors = (np.array(goal) - np.array(start)) / (grid_size - 1)  # Corrected scaling factor
        scaled_coords = (np.array(coords) - start) / scale_factors
        return tuple(np.clip(scaled_coords.astype(int), 0, grid_size-1))  # Ensure within bounds

    # Unscale the path back to original coordinates
    def unscale_path(path, start, goal, grid_size):
        start, goal = np.array(start), np.array(goal)
        scale_factors = (np.array(goal) - np.array(start)) / (grid_size - 1)
        return [(start + np.array(p) * scale_factors).tolist() for p in path]

    # Scale the start and goal coordinates to fit within the grid size
    scaled_start = scale_coordinates(start, start, goal, grid_size)
    scaled_goal = scale_coordinates(goal, start, goal, grid_size)

    # Initialize the open list (priority queue)
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(scaled_start, scaled_goal), 0, scaled_start))  # (f, g, node)
    
    # Cost and parent tracking dictionaries
    g_costs = {scaled_start: 0}
    came_from = {scaled_start: None}

    while open_list:
        _, current_g, current = heapq.heappop(open_list)

        if current == scaled_goal:
            # Reconstruct the path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path = path[::-1]  # Reverse the path to get the correct order

            # Unscale the path to original coordinates
            return unscale_path(path, start, goal, grid_size)

        # Explore neighbors
        for neighbor in neighbors(current):
            tentative_g = current_g + np.linalg.norm(np.array(current) - np.array(neighbor))

            if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g
                f_cost = tentative_g + heuristic(neighbor, scaled_goal)
                heapq.heappush(open_list, (f_cost, tentative_g, neighbor))
                came_from[neighbor] = current

    return []  # No path found

# Optional plotting using matplotlib
def plot_path(path):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], marker='o', color='b', markersize=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
    else:
        print("No path found.")

def compute_velocities(points, velocity_magnitude=5.0):
    velocities = []
    
    # If there are no points or only one point, return an empty list or a list with one zero vector
    if len(points) <= 1:
        return [np.zeros(3)]
    
    for i in range(len(points)):
        if i == 0 or i == len(points) - 1:
            # For the first and last points, velocity is zero
            velocities.append(np.zeros(3))
        else:
            # Compute the velocity vector from points[i] to points[i+1]
            direction = points[i + 1] - points[i]  # Correct direction from points[i] to points[i+1]
            direction_norm = np.linalg.norm(direction)  # Calculate the norm of the direction vector
            
            # Normalize and scale the direction vector to have the desired velocity magnitude
            if direction_norm != 0:
                velocity_vector = (direction / direction_norm) * velocity_magnitude
            else:
                velocity_vector = np.zeros(3)
            
            velocities.append(velocity_vector)
    
    return velocities

class PathPlanner:
    """
    Path planner class. It utilizes the starting point and the target point to calculate the path.
    """
    def __init__(self, config) -> None:
        self.config = config

        # Start the ROS node
        rospy.init_node("planner")  

        # Subscribe to the global target topic
        self.global_target_sub = rospy.Subscriber(config['ego_vehicle']['reference_topic'], Twist, self.global_target_callback)
        self.pose_sub = rospy.Subscriber(f"/{config['ego_vehicle']['type']}/pose", PoseStamped, self.pose_callback)

        
        if self.config['ego_vehicle']['planner'] == 'simple':
            # Subscribe to perception topics if planner type is 'simple'
            self.perception_vel_sub = rospy.Subscriber(
                config['perception_vel_topic'], 
                Twist, 
                self.perception_vel_callback
            )
            self.perception_control_sub = rospy.Subscriber(
                config['perception_control_topic'], 
                Float32MultiArray, 
                self.perception_control_callback
            )
            rospy.loginfo("Subscribed to perception_vel_topic and perception_control_topic")













        # Publish to the target waypoint topic
        self.target_waypoint_pub = rospy.Publisher('/target/waypoint', Float32MultiArray, queue_size=1)

        # Calculate the path to the target
        self.start_point, self.end_point = self.get_start_end_points(config)
        self.waypoints, self.velocities = self.get_waypoints(self.start_point, self.end_point)

        # Waypoint counter
        self.waypoint_counter = 0

    def get_waypoints(self, start_point: np.array, end_point: np.array) -> List[np.array]:
        """
        Identify the waypoints towards the global target.

        Args:
            - start_point (np.array): starting point
            - end_point (np.array): end point

        Return:
            - waypoints (List[np.array]): a list of numpy arrays of shape (3,), where each entry contains 3D coordinates
        """
        if self.config['ego_vehicle']['planner'] == 'simple':
            waypoints = [end_point]
            velocities = [0]
        elif self.config['ego_vehicle']['planner'] == 'a_star':
            waypoints = astar(tuple(start_point.tolist()), tuple(end_point.tolist()))
            velocities = compute_velocities([np.array(x) for x in waypoints], velocity_magnitude=0.5)
        else:
            raise ValueError(f"Unknown planner {self.config['ego_vehicle']['planner']}")

        return waypoints, velocities

    def get_start_end_points(self, config):
        # Start point
        start_point = np.array([
            config['ego_vehicle']['location']['x'],
            config['ego_vehicle']['location']['y'],
            config['ego_vehicle']['location']['z']
        ])

        # End point
        end_point = np.array([
            config['target']['x'],
            config['target']['y'],
            config['target']['z']
        ])
        if config['target']['type'] == 'relative':
            end_point += start_point
        elif config['target']['type'] == 'absolute':
            pass
        else:
            raise ValueError(f"Unknown target type {config['target']['type']}.")

        return (start_point, end_point)

    def run(self):
        r = rospy.Rate(10)
        start_time = time.time()

        while not rospy.is_shutdown():
            # Publish the current waypoint
            # waypoint = Twist()
            # waypoint.linear.x = self.waypoints[self.waypoint_counter][0]
            # waypoint.linear.y = self.waypoints[self.waypoint_counter][1]
            # waypoint.linear.z = self.waypoints[self.waypoint_counter][2]
            message = Float32MultiArray()
            message.data = [
                self.waypoints[self.waypoint_counter][0],
                self.waypoints[self.waypoint_counter][1],
                self.waypoints[self.waypoint_counter][2],
                self.velocities[self.waypoint_counter][0],
                self.velocities[self.waypoint_counter][1],
                self.velocities[self.waypoint_counter][2]
            ]
            
            if time.time() - start_time < 5:
                # waypoint.linear.x = self.config['ego_vehicle']['location']['x']
                # waypoint.linear.y = self.config['ego_vehicle']['location']['y']
                # waypoint.linear.z = self.config['ego_vehicle']['location']['z']
                message.data = [
                    self.config['ego_vehicle']['location']['x'],
                    self.config['ego_vehicle']['location']['y'],
                    self.config['ego_vehicle']['location']['z'],
                    0,
                    0,
                    0
                ]
            
            # Publish the data
            self.target_waypoint_pub.publish(message)

            rospy.loginfo(f"Publishing message: {message}")

            r.sleep()
    
    def pose_callback(self, data):
        # Only do something if the waypoint counter is not the last counter
        if self.waypoint_counter < len(self.waypoints) - 1:
            # Current position
            x_curr = data.pose.position.x
            y_curr = data.pose.position.y
            z_curr = data.pose.position.z

            # Current waypoint
            x_waypoint = self.waypoints[self.waypoint_counter][0]
            y_waypoint = self.waypoints[self.waypoint_counter][1]
            z_waypoint = self.waypoints[self.waypoint_counter][2]

            # Compute the distance
            dist = math.sqrt(
                (x_curr - x_waypoint) ** 2 +
                (y_curr - y_waypoint) ** 2 +
                (z_curr - z_waypoint) ** 2
            )
            
            # If the distance is below a threshold, move to the next waypoint
            if dist < self.config['landing_threshold']:
                self.waypoint_counter += 1

    def global_target_callback(self, data):
        pass

    def perception_vel_callback(self, msg):
        pass
        #rospy.loginfo(f"Received velocity data: {msg}")

    def perception_control_callback(self, msg):
        pass
        #rospy.loginfo(f"Received control data: {msg}")


if __name__ == "__main__":
    # Load the config
    config = load_yaml_file(constants.merged_config_path, __file__)

    # Initialize the planner class
    planner = PathPlanner(
        config=config
    )

    # Run the planner
    planner.run()