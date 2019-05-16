from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import utils.tfrecord_imagenet_utils as imagenet_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tfrecord = imagenet_utils.dataset2tfrecord('F:\\test\\',
                                           'F:\\tfrecord\\', 'test', 5)
print(tfrecord)
