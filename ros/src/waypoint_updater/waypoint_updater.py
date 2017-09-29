#!/usr/bin/env python

import sys
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from geometry_msgs.msg import TwistStamped

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
REDUCE_ZONE = 30 # Reduce zone before the stop line (m)
MOVE_ON_LINE = 5 # Car should move on if it passes the line beyond the stop_line (m)
STOP_BUFFER = 4 # Buffer for the overshoot

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.waypoints = None
        self.base_velocities = []
        self.current_pose = None
        self.stop_line_index = -1
        self.red_light_on = True
        self.last_waypoint_index = -1
        self.lane = Lane()

        self.loop()

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()
            if self.current_pose is None:
                continue

            start_index = self.get_closest_waypoint(self.current_pose)

            self.lane.waypoints = []
            next_index = start_index

            for _ in range(LOOKAHEAD_WPS):

                if next_index >= len(self.waypoints):
                    next_index = 0

                distance_to_stop = self.distance(self.waypoints, next_index, self.stop_line_index - STOP_BUFFER)
                distance_past_stop_line = self.distance(self.waypoints, self.stop_line_index, next_index)

                # Base Speed
                velocity = self.base_velocities[next_index]

                # Reduce Speed
                if distance_to_stop < REDUCE_ZONE and distance_past_stop_line < MOVE_ON_LINE:
                    velocity = min(math.sqrt(2*distance_to_stop), self.base_velocities[next_index])
                    rospy.loginfo("lazzzy: stop_line_index: %f, next_index: %f, velocity: %f", self.stop_line_index, next_index, velocity)

                # Green Light
                if self.red_light_on == False:
                    velocity = self.base_velocities[next_index]

                # Set the speed
                self.set_waypoint_velocity(self.waypoints, next_index, velocity)
                self.lane.waypoints.append(self.waypoints[next_index])

                next_index += 1

            # publish final waypoints
            self.final_waypoints_pub.publish(self.lane)

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        if self.waypoints == None:
            return 0

        closest_index = 0
        min_dist = float('inf')
        dist2 = lambda a, b: (a.x - b.x)**2 + (a.y - b.y)**2
        for i, wp in enumerate(self.waypoints):
            d = dist2(pose.position, wp.pose.pose.position)
            if d < min_dist:
                min_dist = d
                closest_index = i
        return closest_index

    def current_velocity_cb(self, msg):
        self.current_velocity = (msg.twist.linear.x, msg.twist.angular.z)

    def pose_cb(self, msg):
        self.current_pose = msg.pose

    def waypoints_cb(self, waypoints):
        if self.waypoints != None:
            return
        self.waypoints = waypoints.waypoints
        self.base_velocities = [self.waypoints[i].twist.twist.linear.x for i in range(len(self.waypoints))]

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.red_light_on = True
        if msg.data == -1:
            self.red_light_on = False
        else:
            self.red_light_on = True
            self.stop_line_index = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
