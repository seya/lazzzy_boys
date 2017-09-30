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
        self.image_id = 0
       
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def image_cb(self, msg):
        camera_image = msg
        cv_image = self.bridge.imgmsg_to_cv2(camera_image, "bgr8")
#         img_folder = '/home/student/workspace/system_integration/traffic_light_classifier/data/traffic_light_bag_files/images/'
        img_folder = '/home/student/workspace/system_integration/traffic_light_classifier/data/traffic_light_bag_files_test/images/'
        img_file_name = "{}B_{:07d}.jpg".format(img_folder, self.image_id)
        rospy.loginfo("{}".format(img_file_name))
        cv2.imwrite(img_file_name,cv_image)
        self.image_id += 1


       

  


    

    

if __name__ == '__main__':
    try:
        ParseTLBag()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not parse traffic light bag node.')
