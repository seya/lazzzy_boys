import tensorflow as tf
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../traffic_light_classifier')))
# print(sys.path)
from nets import inception_v4
from preprocessing import inception_preprocessing
# from datasets import trafficlight_to_tfrecords
from nets.tl_model import TLModel
import time
import glob

class_names_to_ids = {"RED":0, "YELLOW":1, "GREEN":2, "UNKNOWN":3}
class TLClassifier(object):
    def __init__(self):

        ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../traffic_light_classifier'))
#         model_file = ckpt_path + '/logs/deploy/graph_frozen.pb'
        model_file = ckpt_path + '/logs/deploy/graph_optimized.pb'
        
        with tf.gfile.GFile(model_file, 'rb') as f:
            graph_def_optimized = tf.GraphDef()
            graph_def_optimized.ParseFromString(f.read())
        
        G = tf.Graph()
        
        
        
        isess = tf.InteractiveSession(graph=G)
        tf.import_graph_def(graph_def_optimized, name='')
        self.predictions = G.get_tensor_by_name('predictions:0')
        self.img_input = G.get_tensor_by_name('inputs:0')
        
        self.isess = isess
        return

    def get_classification(self, image,show_time=False):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        start = time.time()
        predictons = self.isess.run(self.predictions,feed_dict={self.img_input: image})
        elapsed = time.time()
        elapsed = elapsed - start
        if show_time:
            print('inference : {:.3f} seconds.'.format(elapsed))
        
        return predictons[0]
    
    def get_image_paths(self):
        X = []
        traffic_clasiffication_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../traffic_light_classifier/data'))
#         dataset_dirs = [traffic_clasiffication_path + '/traffic_light_bag_files_test/images']
        dataset_dirs = [traffic_clasiffication_path + '/sim_images_test']
        for dataset_dir in dataset_dirs:
            for filename in glob.iglob(dataset_dir + '/**/*.jpg'):
                if 'UNKNOWN' in filename:
                    continue
#                 print(filename)
                base_file_name = filename.split('/')[-1][:-4]
                if  base_file_name.endswith('_318'):
                    light_distance = int(base_file_name.split('_')[-2])
                    if light_distance > 80:
                        # for traffic light 318, the traffic light is not within camera image when it's too far away
                        continue
                    
                X.append(filename)
        X.sort()
        return  X
    def run(self):
        
#         img_paths = ['sim_images/GREEN/0000120_147_784.jpg',
#                          'sim_images/GREEN/0000119_148_784.jpg',
#                          'sim_images/GREEN/0000119_148_784.jpg',
#                          'sim_images/RED/0000052_40_318.jpg',
#                          'sim_images/YELLOW/0000164_103_784.jpg']
#         img_paths = [traffic_clasiffication_path + '/'+ img_path for img_path in img_paths]
        correct_pred = 0
        img_paths = self.get_image_paths()
        for i, img_path in enumerate(img_paths):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            pred_label = self.get_classification(img, show_time=True) 
            gt_label = class_names_to_ids[img_path.split('/')[-2]]
            
            if pred_label == gt_label:
                correct_pred += 1.0
            else:
                print("actual={},pred={},image={}".format(gt_label, pred_label,img_path))
            base_file_name = img_path.split('/')[-2] + img_path.split('/')[-1][:-4]
            print("{}/{}: {}/{}, {}, acc={}".format(i+1, len(img_paths), gt_label,pred_label, base_file_name, correct_pred/float(i+1)))
        print("acc={}".format(correct_pred/len(img_paths)))
            
            
        return


if __name__ == "__main__":   
    obj= TLClassifier()
    obj.run()
