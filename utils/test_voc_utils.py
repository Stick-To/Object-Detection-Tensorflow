from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import utils.tfrecord_voc_utils as voc_utils
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# annotations path, image path, save path, tfrecord prefix, shard

tfrecord = voc_utils.dataset2tfrecord('../VOC/Annotations', '../VOC/JPEGImages',
                                      '../data/', 'test', 10)
print(tfrecord)
