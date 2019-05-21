import tensorflow as tf
import numpy as np
import os
import utils.tfrecord_voc_utils as voc_utils
import YOLOv3 as yolov3
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from skimage import io, transform
from utils.voc_classname_encoder import classname_to_ids

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device_name = tf.test.gpu_device_name()
if device_name is not '':
    print('Found GPU Device!')
else:
    print('Found GPU Device Failed!')

lr = 0.001
batch_size = 32
buffer_size = 1024
epochs = 160
reduce_lr_epoch = []
config = {
    'mode': 'train',                                    # 'train', 'test'
    'is_pretraining': False,                            # if True, train network as a classifier
    'data_shape': [448, 448, 3],
    'num_classes': 20,
    'weight_decay': 5e-4,
    'keep_prob': 0.5,                                   # not used
    'data_format': 'channels_last',                     # 'channels_last' 'channels_first'
    'batch_size': batch_size,
    'coord_scale': 1,
    'noobj_scale': 1,
    'obj_scale': 5.,
    'class_scale': 1.,

    'nms_score_threshold': 0.5,
    'nms_max_boxes': 10,
    'nms_iou_threshold': 0.5,

    'rescore_confidence': False,

    # priors are divided by downsampling rate , the first third are used by the coarsest feature maps
    # so they should divided by 8.0 in darknet53 ,
    # the middle third are divided by 16.0, the last third are divided by by 32.0,
    'priors': [[1.25, 1.625], [2., 3.75], [4.125, 2.875], [1.875, 3.8125], [3.875, 2.8125],
               [3.6875, 7.4375], [3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]
}

image_augmentor_config = {
    'data_format': 'channels_last',
    'output_shape': [448, 448],
    'zoom_size': [480, 480],
    'crop_method': 'random',
    'flip_prob': [0., 0.5],
    'fill_mode': 'BILINEAR',
    'keep_aspect_ratios': False,
    'constant_values': 0.,
    'color_jitter_prob': 0.5,
    'rotate': [0.5, -10., 10.],
    'pad_truth_to': 60,
}


data = ['./test/test_00000-of-00005.tfrecord',
        './test/test_00001-of-00005.tfrecord',]

train_gen = voc_utils.get_generator(data,
                                    batch_size, buffer_size, image_augmentor_config)
trainset_provider = {
    'data_shape': [448, 448, 3],
    'num_train': 5011,
    'num_val': 0,                                       # not used
    'train_generator': train_gen,
    'val_generator': None                               # not used
}

testnet = yolov3.YOLOv3(config, trainset_provider)
# testnet.load_weight()
for i in range(epochs):
    print('-'*25, 'epoch', i, '-'*25)
    if i in reduce_lr_epoch:
        lr = lr/10.
        print('reduce lr, lr=', lr, 'now')
    mean_loss = testnet.train_one_epoch(lr)
    print('>> mean loss', mean_loss)
    testnet.save_weight('latest', './weight/test')       # 'latest', 'best'

# img = io.imread()
# img = transform.resize(img, [448,448])
# img = np.expand_dims(img, 0)
# result = testnet.test_one_image(img)
# id_to_clasname = {k:v for (v,k) in classname_to_ids.items()}
# scores = result[0]
# bbox = result[1]
# class_id = result[2]
# print(scores, bbox, class_id)
# plt.figure(1)
# plt.imshow(np.squeeze(img))
# axis = plt.gca()
# for i in range(len(scores)):
#     rect = patches.Rectangle((bbox[i][1],bbox[i][0]), bbox[i][3]-bbox[i][1],bbox[i][2]-bbox[i][0],linewidth=2,edgecolor='b',facecolor='none')
#     axis.add_patch(rect)
#     plt.text(bbox[i][1],bbox[i][0], id_to_clasname[class_id[i]]+str(' ')+str(scores[i]), color='red', fontsize=12)
# plt.show()
