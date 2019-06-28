from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import tfrecord_voc_utils as voc_utils
import tensorflow as tf
import numpy as np
import LH_RCNN as net
import os
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from skimage import io, transform
# from utils.voc_classname_encoder import classname_to_ids
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
lr = 0.003
batch_size = 32
buffer_size = 1024
epochs = 1500
reduce_lr_epoch = []
config = {
    'data_shape': [700, 1100, 3],
    'mode': 'train',                            # 'train' ,'test'
    'is_pretraining': False,
    'data_format': 'channels_last',             # 'channels_last' ,'channels_first'
    'num_classes': 20,
    'weight_decay': 1e-4,
    'keep_prob': 0.5,                           # not used
    'batch_size': batch_size,
    'rpn_first_step': 60000,                    # iters 0 - rpn_first_step train  rpn only
    'rcnn_first_step': 100000,                   # iters rpn_first_step - rcnn_first_step train  rcnn only
    'rpn_second_step': 160000,                   # iters rcnn_first_step - rpn_second_step train  rpn only
                                                # iters rpn_second_step - end train  rcnn only
    'nms_score_threshold': 0.5,
    'nms_max_boxes': 20,
    'nms_iou_threshold': 0.45,
    'post_nms_proposal': 500                    # when test, how many proposal are kept after nms
}

image_augmentor_config = {
    'data_format': 'channels_last',
    'output_shape': [700, 1100],
    'zoom_size': [720, 1120],
    'crop_method': 'random',
    'flip_prob': [0., 0.5],
    'fill_mode': 'BILINEAR',
    'keep_aspect_ratios': False,
    'constant_values': 0.,
    'color_jitter_prob': 0.5,
    'rotate': [0.5, -5., -5.],
    'pad_truth_to': 60,
}

data = os.listdir('./voc2007/')
data = [os.path.join('./voc2007/', name) for name in data]

train_gen = voc_utils.get_generator(data,
                                    batch_size, buffer_size, image_augmentor_config)
trainset_provider = {
    'data_shape': [700, 1100, 3],
    'num_train': 5011,
    'num_val': 0,                               # not used
    'train_generator': train_gen,
    'val_generator': None                       # not used
}
rcnn = net.LHRCNN(config, trainset_provider)
# rcnn.load_pretraining_weight('./rcnnpretrain/test-30000')
# rcnn.load_weight('./lhrcnn/test-44304')
for i in range(epochs):
    print('-'*25, 'epoch', i, '-'*25)
    if i in reduce_lr_epoch:
        lr = lr/10.
        print('reduce lr, lr=', lr, 'now')
    mean_loss = rcnn.train_one_epoch(lr)
    print('>> mean loss', mean_loss, )
    rcnn.save_weight('latest', './lhrcnn/test')    # 'latest' 'best'

# img = io.imread('000026.jpg')
# img = transform.resize(img, [700,1100])
# img = np.expand_dims(img, 0)
# result = ssd300.test_one_image(img)
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
