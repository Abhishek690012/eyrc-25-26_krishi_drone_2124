#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose

# !! IMPORTANT !!    
# You must first create the 'waypoint_navigation' package and define
# 'GetWaypoints.srv' inside its 'srv/' folder.
# The .srv file should look like:
# bool get_waypoints
# ---
# geometry_msgs/PoseArray waypoints
#
# Then build your workspace (colcon build)
try:
    from waypoint_navigation.srv import GetWaypoints
except ImportError:
    print("CRITICAL: 'waypoint_navigation' package or 'GetWaypoints' service not found.")
    print("Please create this package and service, then 'colcon build' your workspace.")
    exit(1)

class WayPoints(Node):

    def __init__(self):
        super().__init__('waypoints_service')
        self.get_logger().info('Waypoint Service Node is running...')
        self.srv = self.create_service(GetWaypoints, 'waypoints', self.waypoint_callback)
        
        # The waypoints from the e-Yantra portal (in decimeters/Whycon units)
        self.waypoints = [
            [-7.00, 0.00, 29.22],  # hover
            [-7.64, 3.06, 29.22],  # wp1
            [-8.22, 6.02, 29.22],  # wp2
            [-9.11, 9.27, 29.27],  # wp3
            [-5.98, 8.81, 29.27],  # wp4
            [-3.26, 8.41, 29.88],  # wp5
            [0.87, 8.18, 29.05],   # wp6
            [3.93, 7.35, 29.05]    # wp7
        ]
        self.get_logger().info(f'{len(self.waypoints)} waypoints loaded.')

    
    def waypoint_callback(self, request, response):
        # The request object must have a 'get_waypoints' field (bool)
        
        if request.get_waypoints == True :
            self.get_logger().info("Incoming request for Waypoints... sending waypoints.")
            
            # Create a list of Pose objects
            response.waypoints.poses = [Pose() for _ in range(len(self.waypoints))]
            
            # Populate the Pose objects
            for i in range(len(self.waypoints)):
                response.waypoints.poses[i].position.x = self.waypoints[i][0]
                response.waypoints.poses[i].position.y = self.waypoints[i][1]
                response.waypoints.poses[i].position.z = self.waypoints[i][2]
            
            return response

        else:
            self.get_logger().warn("Request rejected: 'get_waypoints' was not True.")
            return response # Return an empty response

def main(args=None):
    rclpy.init()
    waypoints = WayPoints()

    try:
        rclpy.spin(waypoints)
    except KeyboardInterrupt:
        waypoints.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        waypoints.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()