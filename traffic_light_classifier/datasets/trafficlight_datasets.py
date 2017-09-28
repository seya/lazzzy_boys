
"""Provides data for the Pascal VOC Dataset (images + annotations).
"""
import os

import tensorflow as tf

slim = tf.contrib.slim

DATASET_SIZE = {
    'site_eval': 186,
    'site_train': 744,
    'sim_eval': 1997,
    'sim_train': 7986,
}
NUM_CLASSES = 3

ITEMS_TO_DESCRIPTIONS = {
    'image': 'image data',
    'label': 'image label',
    'format': 'image format',
    'filename': 'image file name',
}
def get_dataset(data_sources, num_samples):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.

    Args:
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
   
    # Allowing None in the signature so that dataset_factory can use the default.
    
    reader = tf.TFRecordReader
    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value='000000'),
        'image/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),   
        'label': slim.tfexample_decoder.Tensor('image/label'),
        'format': slim.tfexample_decoder.Tensor('image/format'),
        'filename': slim.tfexample_decoder.Tensor('image/filename')
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None


    return slim.dataset.Dataset(
            data_sources=data_sources,
            reader=reader,
            decoder=decoder,
            num_samples=num_samples,
            items_to_descriptions=ITEMS_TO_DESCRIPTIONS,
            num_classes=NUM_CLASSES,
            labels_to_names=labels_to_names)

