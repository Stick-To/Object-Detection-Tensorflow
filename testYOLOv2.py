import tensorflow as tf
import numpy as np
import os
import utils.tfrecord_voc_utils as voc_utils
import YOLOv2 as yolov2
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from skimage import io, transform
from utils.voc_classname_encoder import classname_to_ids

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device_name = tf.test.gpu_device_name()
if device_name is not '':
    print('Found GPU Device!')
else:
    print('Found GPU Device Failed!')

lr = 0.005
batch_size = 32
buffer_size = 1024
epochs = 280
input_shape = [480, 480, 3]
reduce_lr_epoch = []
config = {
    'mode': 'train',                                 # 'train', 'test'
    'is_pretraining': False,
    'data_shape': input_shape,
    'num_classes': 20,
    'weight_decay': 1e-4,
    'keep_prob': 0.5,
    'data_format': 'channels_last',                  # 'channels_last' 'channels_first'
    'batch_size': batch_size,
    'coord_scale': 1,
    'noobj_scale': 1,
    'obj_scale': 5.,
    'class_scale': 1.,

    'nms_score_threshold': 0.5,
    'nms_max_boxes': 10,
    'nms_iou_threshold': 0.5,

    'rescore_confidence': False,
    'priors': [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]]
}

data = os.listdir('./voc2007/')
data = [os.path.join('./voc2007/', name) for name in data]

image_augmentor_config = {
    'data_format': 'channels_last',
    'output_shape': [480, 480],
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
train_gen = voc_utils.get_generator(data,
                                    batch_size, buffer_size, image_augmentor_config)
train_provider = {
    'data_shape': input_shape,
    'num_train': 5011,
    'num_val': 0,
    'train_generator': train_gen,
    'val_generator': None
}


testnet = yolov2.YOLOv2(config, train_provider)
testnet.load_weight('./yolo2/test-5304')
for i in range(epochs):
    print('-'*25, 'epoch', i, '-'*25)
    if i in reduce_lr_epoch:
        lr = lr/10.
        print('reduce lr, lr=', lr, 'now')
    mean_loss = testnet.train_one_epoch(lr)
    print('>> mean loss', mean_loss)
    testnet.save_weight('latest', './yolo2/test')       # 'latest', 'best'


# img = io.imread('/home/test/Desktop/YOLO-TF-master/VOC/JPEGImages/000012.jpg')
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
#     rect = patches.Rectangle((bbox[i][1],bbox[i][0]), bbox[i][3]-bbox[i][1],bbox[i][2]-bbox[i][0],linewidth=1,edgecolor='c',facecolor='none')
#     axis.add_patch(rect)
#     plt.text(bbox[i][1],bbox[i][0], id_to_clasname[class_id[i]]+str(' ')+str(scores[i]), color='red', fontsize=12)
# plt.show()
