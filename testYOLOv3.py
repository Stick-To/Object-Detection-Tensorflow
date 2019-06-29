import tensorflow as tf
import numpy as np
import os
import utils.tfrecord_voc_utils as voc_utils
import YOLOv3 as yolov3
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from skimage import io, transform
from utils.voc_classname_encoder import classname_to_ids
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

lr = 0.001
batch_size = 12
buffer_size = 256
epochs = 160
reduce_lr_epoch = []
config = {
    'mode': 'train',                                    # 'train', 'test'
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
    'num_priors': 3,

    'nms_score_threshold': 0.5,
    'nms_max_boxes': 10,
    'nms_iou_threshold': 0.5,


    'priors': [[[10., 13.], [16, 30.], [33., 23.]],
               [[30., 61.], [62., 45.], [59., 119.]],
              [[116., 90.], [156., 198.], [373.,326.]]]

}

image_augmentor_config = {
    'data_format': 'channels_last',
    'output_shape': [448, 448],
    # 'zoom_size': [520, 520],
    # 'crop_method': 'random',
    'flip_prob': [0., 0.5],
    'fill_mode': 'BILINEAR',
    'keep_aspect_ratios': False,
    'constant_values': 0.,
    # 'color_jitter_prob': 0.5,
    # 'rotate': [0.5, -10., 10.],
    'pad_truth_to': 60,
}


data = os.listdir('./voc2007/')
data = [os.path.join('./voc2007/', name) for name in data]

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
testnet.load_weight('./weight/test-40449')
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
