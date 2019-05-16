from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import warnings
import math
import sys
import random
from utils.imagenet_classname_encoder import classname_to_ids
from utils.image_augmentor import image_augmentor


class ImageReader(object):
        def __init__(self):
            self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
            self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

        def decode_jpeg(self, sess, image_data):
            image = sess.run(self._decode_jpeg, feed_dict={
                self._decode_jpeg_data: image_data
            })
            assert len(image.shape) == 3
            assert image.shape[2] == 3
            return image

        def read_image_dims(self, sess, image_data):
            image = self.decode_jpeg(sess, image_data)
            return image.shape


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def dataset2tfrecord(img_dir, output_dir, name, total_shards=50):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)
        print(output_dir, 'does not exist, create it done')
    else:
        if len(tf.gfile.ListDirectory(output_dir)) == 0:
            print(output_dir, 'already exist, need not create new')
        else:
            warnings.warn(output_dir + ' is not empty!', UserWarning)
    image_reader = ImageReader()
    sess = tf.Session()
    outputfiles = []
    directories = []
    class_names = []
    for filename in os.listdir(img_dir):
        path = os.path.join(img_dir, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)
    imglist = []
    for directory in directories:
        for filename in os.listdir(directory):
            imgname = os.path.join(directory, filename)
            imglist.append(imgname)
    random.shuffle(imglist)
    num_per_shard = int(math.ceil(len(imglist)) / float(total_shards))
    for shard_id in range(total_shards):
        outputname = '%s_%05d-of-%05d.tfrecord' % (name, shard_id+1, total_shards)
        outputname = os.path.join(output_dir, outputname)
        outputfiles.append(outputname)
        with tf.python_io.TFRecordWriter(outputname) as tf_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id+1) * num_per_shard, len(imglist))
            for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d/%d' % (
                    i+1, len(imglist), shard_id+1, total_shards))
                sys.stdout.flush()
                image_data = tf.gfile.GFile(imglist[i], 'rb').read()
                shape = image_reader.read_image_dims(sess, image_data)
                shape = np.asarray(shape, np.int32)
                class_name = os.path.basename(os.path.dirname(imglist[i]))
                class_id = classname_to_ids[class_name]
                features = {
                    'image': bytes_feature(image_data),
                    'shape': bytes_feature(shape.tobytes()),
                    'label': int64_feature(class_id)
                }
                example = tf.train.Example(features=tf.train.Features(
                                           feature=features))
                tf_writer.write(example.SerializeToString())
            sys.stdout.write('\n')
            sys.stdout.flush()
    return outputfiles


def parse_function(data, config):
        features = tf.parse_single_example(data, features={
            'image': tf.FixedLenFeature([], tf.string),
            'shape': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })
        shape = tf.decode_raw(features['shape'], tf.int32)
        label = tf.cast(features['label'], tf.int64)
        shape = tf.reshape(shape, [3])
        images = tf.image.decode_jpeg(features['image'], channels=3)
        images = tf.cast(tf.reshape(images, shape), tf.float32)
        images = image_augmentor(image=images,
                                 input_shape=shape,
                                 **config
                                 )
        return images, label, # shape


def get_generator(tfrecords, batch_size, buffer_size, image_preprocess_config):
    data = tf.data.TFRecordDataset(tfrecords)
    data = data.map(lambda x: parse_function(x, image_preprocess_config)).shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True).repeat()
    iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
    init_op = iterator.make_initializer(data)
    return init_op, iterator

