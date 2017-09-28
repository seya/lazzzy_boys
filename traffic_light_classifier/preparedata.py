from datasets import trafficlight_datasets
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
# from nets import nets_factory
from preprocessing import inception_preprocessing
from nets import inception_v4
import numpy as np
import cv2
import math




class PrepareData():
    def __init__(self):
        
        return
    def load_batch(self, dataset, batch_size=None, is_training=None):
        """Loads a single batch of data.
        
        Args:
          dataset: The dataset to load.
          batch_size: The number of images in the batch.
          height: The size of each image after preprocessing.
          width: The size of each image after preprocessing.
          is_training: Whether or not we're currently training or evaluating.
        
        Returns:
          images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
          images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
          labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
        """
        shuffle = False
        num_readers = 1
        if is_training:
            shuffle = True
            #make sure most samples can be fetched in one epoch
            num_readers = 1
       
            
        
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    shuffle=shuffle,
                    num_readers=num_readers,
                    common_queue_capacity=20 * batch_size,
                    common_queue_min=10 * batch_size)
        
#         data_provider = slim.dataset_data_provider.DatasetDataProvider(
#             dataset, common_queue_capacity=32,
#             common_queue_min=8)
        image_raw, label,filename = data_provider.get(['image', 'label','filename'])
        
        # Preprocess image for usage by Inception.
        #No image augmentation for now, but resize normalization
        image_size = inception_v4.inception_v4.default_image_size
        image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training=False)
        
        # Preprocess the image for display purposes.
        image_raw = tf.expand_dims(image_raw, 0)
        image_raw = tf.image.resize_images(image_raw, [image_size, image_size])
        image_raw = tf.squeeze(image_raw)
        
    
        # Batch it up.
        images, images_raw, labels, filenames = tf.train.batch(
              [image, image_raw, label, filename],
              batch_size=batch_size,
              num_threads=1,
              capacity=2 * batch_size)
        
        tf.summary.image('image_raw',images_raw)
        tf.summary.image('image',images)
        return images, images_raw, labels, filenames
    def get_input(self, split_name, is_training=True, batch_size=32):
#         if split_name == "train":
#             data_sources = "./data/tfrecords/sim_train*.tfrecord"
#             num_samples = trafficlight_datasets.DATASET_SIZE['sim_train']
#         else:
#             data_sources = "./data/tfrecords/sim_eval*.tfrecord"
#             num_samples = trafficlight_datasets.DATASET_SIZE['sim_eval']
#         if split_name == "train":
#             data_sources = "./data/tfrecords/site_train*.tfrecord"
#             num_samples = trafficlight_datasets.DATASET_SIZE['site_train']
#         else:
#             data_sources = "./data/tfrecords/site_eval*.tfrecord"
#             num_samples = trafficlight_datasets.DATASET_SIZE['site_eval']
        
        if split_name == "train":
            data_sources = "./data/tfrecords/*_train*.tfrecord"
            num_samples = trafficlight_datasets.DATASET_SIZE['sim_train'] + trafficlight_datasets.DATASET_SIZE['site_train']
        else:
            data_sources = "./data/tfrecords/*_eval*.tfrecord"
            num_samples = trafficlight_datasets.DATASET_SIZE['sim_eval'] + trafficlight_datasets.DATASET_SIZE['site_eval']
            
        self.dataset = trafficlight_datasets.get_dataset(data_sources, num_samples)
        return self.load_batch(self.dataset, batch_size=batch_size, is_training=is_training)
   
    def run(self):
        with tf.Graph().as_default():    
            batch_data = self.get_input("train", is_training=True, batch_size=32)
#             batch_data = self.get_input("eval", is_training=False, batch_size=32)
            
            with tf.Session('') as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                with slim.queues.QueueRunners(sess):  
#                     while True:  
                    filename_list = []
                    for _ in range(math.ceil(self.dataset.num_samples/32)):
                        images, images_raw, labels,filenames = sess.run(list(batch_data))
#                         print(labels)
                        print(filenames)
                        filename_list.extend(filenames)
                    fs = np.unique(filename_list)
                    print(len(fs))
                        
        
        return
    
    


if __name__ == "__main__":   
    obj= PrepareData()
    obj.run()