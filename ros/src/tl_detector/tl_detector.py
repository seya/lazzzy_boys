#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 2

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
REDUCE_ZONE = 30 # Reduce zone before the stop line (m)
MOVE_ON_LINE = 5 # Car should move on if it passes the line beyond the stop_line (m)
STOP_BUFFER = 4 # Buffer for the overshoot

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = None
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.has_image = False
        rospy.loginfo("TLDetector initialization start")
        self.light_classifier = TLClassifier()
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.stop_line_positions = None

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        
        self.listener = tf.TransformListener()       
        rospy.loginfo("TLDetector initialization done")
        
        self.loop()
        return
    def loop(self):
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            rate.sleep()
            if (self.lights is None) or (self.waypoints is  None) or (self.pose is None):
                continue
            light_wp, state = self.process_traffic_lights()

            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if (state == TrafficLight.RED)else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        self.stop_line_positions = [self.get_closest_waypoint(sp_line) for sp_line in self.config['stop_line_positions']]
        self.stop_line_positions.sort()

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        return


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

        if isinstance(pose, list):
            pose_ = Pose()
            pose_.position.x = pose[0]
            pose_.position.y = pose[1]
            pose = pose_

        closest_index = 0
        min_dist = float('inf')
        dist2 = lambda a, b: (a.x - b.x)**2 + (a.y - b.y)**2
        for i, wp in enumerate(self.waypoints.waypoints):
            d = dist2(pose.position, wp.pose.pose.position)
            if d < min_dist:
                min_dist = d
                closest_index = i
        return closest_index

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image

        x = 0
        y = 0

        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

#         x, y = self.project_to_image_plane(light.pose.pose.position)

        #TODO use light location to zoom in on traffic light in image

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        #TODO find the closest visible traffic light (if one exists)
        if (self.lights == None) or (self.waypoints ==  None):
            return -1, TrafficLight.UNKNOWN

        # Generate a list of lights with its closest waypoint
        ### Note: Lights are given by the simulator for test/development purpose
        ###       They are not available with Carla
        lights = []
        for tl in self.lights:
            closest_wp_position = self.get_closest_waypoint(tl.pose.pose)
            lights.append((closest_wp_position, tl))
        lights.sort()

        first_tl_position = lights[0][0]
        last_tl_position = lights[-1][0]

        # Find the closest traffic light to the car
        closest_light_wp_position = len(self.waypoints.waypoints) # set the max number
        for tl_position, tl in lights:
            if ((tl_position >= car_position) and (tl_position < closest_light_wp_position) or
                (car_position > last_tl_position) and (tl_position == first_tl_position)): 
                closest_light_wp_position = tl_position
                light = tl

        if light:
            
            #return light_wp, state
            stop_line_position = -1
            for position in self.stop_line_positions:
                if closest_light_wp_position > position:
                    stop_line_position = position
            distance_to_stop = self.distance(self.waypoints.waypoints, car_position, stop_line_position - STOP_BUFFER)
            if(distance_to_stop <= REDUCE_ZONE + 10):
                light.state = self.get_light_state(light)
            else:
                light.state = TrafficLight.UNKNOWN
            rospy.loginfo("=== lazzzy-lights === light: %d, light index: %d, stop_line: %d, car pos index: %d,distance_to_stop=%1f",
                           light.state, closest_light_wp_position, stop_line_position, car_position,distance_to_stop)
            return stop_line_position, light.state
#         self.waypoints = None
        return -1, TrafficLight.UNKNOWN
    def distance(self, waypoints, car_position, stop_line_position):
        if car_position <= stop_line_position + 200:
            return self.sub_distance(waypoints, car_position, stop_line_position)
        end_waypoint_ind = len(waypoints) -1
        start_waypoint_ind = 0
           
        return self.sub_distance(waypoints, car_position, end_waypoint_ind) + self.sub_distance(waypoints, start_waypoint_ind, stop_line_position)
    def sub_distance(self,waypoints, wp1,wp2):
        #this function assumes wp1 <= wp2
        if wp1 == wp2:
            return 0
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
