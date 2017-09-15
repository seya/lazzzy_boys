#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

class ParseTLBag(object):
    def __init__(self):
        rospy.init_node('parse_tl_bag')

        self.pose = None
        self.camera_image = None

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        
        rospy.Subscriber('/image_raw', Image, self.image_cb)

        
        self.bridge = CvBridge()
       
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def image_cb(self, msg):
        self.camera_image = msg
       

  


    

    

if __name__ == '__main__':
    try:
        ParseTLBag()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not parse traffic light bag node.')
