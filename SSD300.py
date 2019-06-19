from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow as wrap
import sys
import os
import numpy as np


class SSD300:
    def __init__(self, config, data_provider):
        assert config['mode'] in ['train', 'test']
        assert config['data_format'] in ['channels_first', 'channels_last']
        self.config = config
        self.data_provider = data_provider
        self.input_size = 300
        if config['data_format'] == 'channels_last':
            self.data_shape = [300, 300, 3]
        else:
            self.data_shape = [3, 300, 300]
        self.num_classes = config['num_classes'] + 1
        self.weight_decay = config['weight_decay']
        self.prob = 1. - config['keep_prob']
        self.data_format = config['data_format']
        self.mode = config['mode']
        self.batch_size = config['batch_size'] if config['mode'] == 'train' else 1
        self.nms_score_threshold = config['nms_score_threshold']
        self.nms_max_boxes = config['nms_max_boxes']
        self.nms_iou_threshold = config['nms_iou_threshold']
        self.reader = wrap.NewCheckpointReader(config['pretraining_weight'])

        if self.mode == 'train':
            self.num_train = data_provider['num_train']
            self.num_val = data_provider['num_val']
            self.train_generator = data_provider['train_generator']
            self.train_initializer, self.train_iterator = self.train_generator
            if data_provider['val_generator'] is not None:
                self.val_generator = data_provider['val_generator']
                self.val_initializer, self.val_iterator = self.val_generator

        self.global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)
        self.is_training = True

        self._define_inputs()
        self._build_graph()
        self._create_saver()
        if self.mode == 'train':
            self._create_summary()
        self._init_session()

    def _define_inputs(self):
        shape = [self.batch_size]
        shape.extend(self.data_shape)
        mean = tf.convert_to_tensor([123.68, 116.779, 103.979], dtype=tf.float32)
        if self.data_format == 'channels_last':
            mean = tf.reshape(mean, [1, 1, 1, 3])
        else:
            mean = tf.reshape(mean, [1, 3, 1, 1])
        if self.mode == 'train':
            self.images, self.ground_truth = self.train_iterator.get_next()
            self.images.set_shape(shape)
            self.images = self.images - mean
        else:
            self.images = tf.placeholder(tf.float32, shape, name='images')
            self.images = self.images - mean
            self.ground_truth = tf.placeholder(tf.float32, [self.batch_size, None, 5], name='labels')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

    def _build_graph(self):
        with tf.variable_scope('feature_extractor'):
            feat1, feat2, feat3, feat4, feat5, feat6 = self._feature_extractor(self.images)
            feat1 = tf.nn.l2_normalize(feat1, axis=3 if self.data_format == 'channels_last' else 1)
            norm_factor = tf.get_variable('l2_norm_factor', initializer=tf.constant(20.))
            feat1 = norm_factor * feat1
        with tf.variable_scope('regressor'):
            pred1 = self._conv_layer(feat1, 4*(self.num_classes+4), 3, 1, 'pred1')
            pred2 = self._conv_layer(feat2, 6*(self.num_classes+4), 3, 1, 'pred2')
            pred3 = self._conv_layer(feat3, 6*(self.num_classes+4), 3, 1, 'pred3')
            pred4 = self._conv_layer(feat4, 6*(self.num_classes+4), 3, 1, 'pred4')
            pred5 = self._conv_layer(feat5, 4*(self.num_classes+4), 3, 1, 'pred5')
            pred6 = self._conv_layer(feat6, 4*(self.num_classes+4), 3, 1, 'pred6')
            if self.data_format == 'channels_first':
                pred1 = tf.transpose(pred1, [0, 2, 3, 1])
                pred2 = tf.transpose(pred2, [0, 2, 3, 1])
                pred3 = tf.transpose(pred3, [0, 2, 3, 1])
                pred4 = tf.transpose(pred4, [0, 2, 3, 1])
                pred5 = tf.transpose(pred5, [0, 2, 3, 1])
                pred6 = tf.transpose(pred6, [0, 2, 3, 1])
            p1shape = tf.shape(pred1)
            p2shape = tf.shape(pred2)
            p3shape = tf.shape(pred3)
            p4shape = tf.shape(pred4)
            p5shape = tf.shape(pred5)
            p6shape = tf.shape(pred6)
        with tf.variable_scope('inference'):
            p1bbox_yx, p1bbox_hw, p1conf = self._get_pbbox(pred1)
            p2bbox_yx, p2bbox_hw, p2conf = self._get_pbbox(pred2)
            p3bbox_yx, p3bbox_hw, p3conf = self._get_pbbox(pred3)
            p4bbox_yx, p4bbox_hw, p4conf = self._get_pbbox(pred4)
            p5bbox_yx, p5bbox_hw, p5conf = self._get_pbbox(pred5)
            p6bbox_yx, p6bbox_hw, p6conf = self._get_pbbox(pred6)

            s = [(0.2 + (0.9 - 0.2) / 5 * (i-1)) * self.input_size for i in range(1, 8)]
            s = [[s[i], (s[i]*s[i+1])**0.5] for i in range(0, 6)]
            a1bbox_y1x1, a1bbox_y2x2, a1bbox_yx, a1bbox_hw = self._get_abbox(s[0], [2, 1/2], p1shape)
            a2bbox_y1x1, a2bbox_y2x2, a2bbox_yx, a2bbox_hw = self._get_abbox(s[1], [2, 1/2, 3, 1/3], p2shape)
            a3bbox_y1x1, a3bbox_y2x2, a3bbox_yx, a3bbox_hw = self._get_abbox(s[2], [2, 1/2, 3, 1/3], p3shape)
            a4bbox_y1x1, a4bbox_y2x2, a4bbox_yx, a4bbox_hw = self._get_abbox(s[3], [2, 1/2, 3, 1/3], p4shape)
            a5bbox_y1x1, a5bbox_y2x2, a5bbox_yx, a5bbox_hw = self._get_abbox(s[4], [2, 1/2], p5shape)
            a6bbox_y1x1, a6bbox_y2x2, a6bbox_yx, a6bbox_hw = self._get_abbox(s[5], [2, 1/2], p6shape)

            pbbox_yx = tf.concat([p1bbox_yx, p2bbox_yx, p3bbox_yx, p4bbox_yx, p5bbox_yx, p6bbox_yx], axis=1)
            pbbox_hw = tf.concat([p1bbox_hw, p2bbox_hw, p3bbox_hw, p4bbox_hw, p5bbox_hw, p6bbox_hw], axis=1)
            pconf = tf.concat([p1conf, p2conf, p3conf, p4conf, p5conf, p6conf], axis=1)
            abbox_y1x1 = tf.concat([a1bbox_y1x1, a2bbox_y1x1, a3bbox_y1x1, a4bbox_y1x1, a5bbox_y1x1, a6bbox_y1x1], axis=0)
            abbox_y2x2 = tf.concat([a1bbox_y2x2, a2bbox_y2x2, a3bbox_y2x2, a4bbox_y2x2, a5bbox_y2x2, a6bbox_y2x2], axis=0)
            abbox_yx = tf.concat([a1bbox_yx, a2bbox_yx, a3bbox_yx, a4bbox_yx, a5bbox_yx, a6bbox_yx], axis=0)
            abbox_hw = tf.concat([a1bbox_hw, a2bbox_hw, a3bbox_hw, a4bbox_hw, a5bbox_hw, a6bbox_hw], axis=0)
            if self.mode == 'train':
                i = 0.
                loss = 0.
                cond = lambda loss, i: tf.less(i, tf.cast(self.batch_size, tf.float32))
                body = lambda loss, i: (
                    tf.add(loss, self._compute_one_image_loss(
                        tf.squeeze(tf.gather(pbbox_yx, tf.cast(i, tf.int32))),
                        tf.squeeze(tf.gather(pbbox_hw, tf.cast(i, tf.int32))),
                        abbox_y1x1,
                        abbox_y2x2,
                        abbox_yx,
                        abbox_hw,
                        tf.squeeze(tf.gather(pconf, tf.cast(i, tf.int32))),
                        tf.squeeze(tf.gather(self.ground_truth, tf.cast(i, tf.int32))),
                    )),
                    tf.add(i, 1.)
                )
                init_state = (loss, i)
                state = tf.while_loop(cond, body, init_state)
                total_loss, _ = state
                total_loss = total_loss / self.batch_size
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=.9)
                self.loss = total_loss + self.weight_decay * tf.add_n(
                    [tf.nn.l2_loss(var) for var in tf.trainable_variables()]
                ) 
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                train_op = optimizer.minimize(self.loss, global_step=self.global_step)
                self.train_op = tf.group([update_ops, train_op])
            else:
                pbbox_yxt = pbbox_yx[0, ...]
                pbbox_hwt = pbbox_hw[0, ...]
                confidence = tf.nn.softmax(pconf[0, ...])
                class_id = tf.argmax(confidence, axis=-1)
                conf_mask = tf.less(class_id, self.num_classes - 1)
                pbbox_yxt = tf.boolean_mask(pbbox_yxt, conf_mask)
                pbbox_hwt = tf.boolean_mask(pbbox_hwt, conf_mask)
                confidence = tf.boolean_mask(confidence, conf_mask)[:, :self.num_classes - 1]
                abbox_yxt = tf.boolean_mask(abbox_yx, conf_mask)
                abbox_hwt = tf.boolean_mask(abbox_hw, conf_mask)
                dpbbox_yxt = pbbox_yxt * abbox_hwt + abbox_yxt
                dpbbox_hwt = abbox_hwt * tf.exp(pbbox_hwt)
                dpbbox_y1x1 = dpbbox_yxt - dpbbox_hwt / 2.
                dpbbox_y2x2 = dpbbox_yxt + dpbbox_hwt / 2.
                dpbbox_y1x1y2x2 = tf.concat([dpbbox_y1x1, dpbbox_y2x2], axis=-1)
                filter_mask = tf.greater_equal(confidence, self.nms_score_threshold)
                scores = []
                class_id = []
                bbox = []
                for i in range(self.num_classes - 1):
                    scoresi = tf.boolean_mask(confidence[:, i], filter_mask[:, i])
                    bboxi = tf.boolean_mask(dpbbox_y1x1y2x2, filter_mask[:, i])
                    selected_indices = tf.image.non_max_suppression(

                        bboxi, scoresi, self.nms_max_boxes, self.nms_iou_threshold,
                    )
                    scores.append(tf.gather(scoresi, selected_indices))
                    bbox.append(tf.gather(bboxi, selected_indices))
                    class_id.append(tf.ones_like(tf.gather(scoresi, selected_indices), tf.int32) * i)
                bbox = tf.concat(bbox, axis=0)
                scores = tf.concat(scores, axis=0)
                class_id = tf.concat(class_id, axis=0)

                self.detection_pred = [scores, bbox, class_id]

    def _feature_extractor(self, images):
        conv1_1 = self._load_conv_layer(images,
                                        tf.get_variable(name='kernel_conv1_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv1/conv1_1/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv1_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv1/conv1_1/biases"),
                                                        trainable=True),
                                        name="conv1_1")
        conv1_2 = self._load_conv_layer(conv1_1,
                                        tf.get_variable(name='kernel_conv1_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv1/conv1_2/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv1_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv1/conv1_2/biases"),
                                                        trainable=True),
                                        name="conv1_2")
        pool1 = self._max_pooling(conv1_2, 2, 2, name="pool1")

        conv2_1 = self._load_conv_layer(pool1,
                                        tf.get_variable(name='kenrel_conv2_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv2/conv2_1/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv2_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv2/conv2_1/biases"),
                                                        trainable=True),
                                        name="conv2_1")
        conv2_2 = self._load_conv_layer(conv2_1,
                                        tf.get_variable(name='kernel_conv2_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv2/conv2_2/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv2_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv2/conv2_2/biases"),
                                                        trainable=True),
                                        name="conv2_2")
        pool2 = self._max_pooling(conv2_2, 2, 2, name="pool2")
        conv3_1 = self._load_conv_layer(pool2,
                                        tf.get_variable(name='kernel_conv3_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv3/conv3_1/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv_3_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv3/conv3_1/biases"),
                                                        trainable=True),
                                        name="conv3_1")
        conv3_2 = self._load_conv_layer(conv3_1,
                                        tf.get_variable(name='kernel_conv3_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv3/conv3_2/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv3_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv3/conv3_2/biases"),
                                                        trainable=True),
                                        name="conv3_2")
        conv3_3 = self._load_conv_layer(conv3_2,
                                        tf.get_variable(name='kernel_conv3_3',
                                                        initializer=self.reader.get_tensor("vgg_16/conv3/conv3_3/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv3_3',
                                                        initializer=self.reader.get_tensor("vgg_16/conv3/conv3_3/biases"),
                                                        trainable=True),
                                        name="conv3_3")
        pool3 = self._max_pooling(conv3_3, 2, 2, name="pool3")

        conv4_1 = self._load_conv_layer(pool3,
                                        tf.get_variable(name='kernel_conv4_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv4/conv4_1/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv4_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv4/conv4_1/biases"),
                                                        trainable=True),
                                        name="conv4_1")
        conv4_2 = self._load_conv_layer(conv4_1,
                                        tf.get_variable(name='kernel_conv4_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv4/conv4_2/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv4_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv4/conv4_2/biases"),
                                                        trainable=True),
                                        name="conv4_2")
        conv4_3 = self._load_conv_layer(conv4_2,
                                        tf.get_variable(name='kernel_conv4_3',
                                                        initializer=self.reader.get_tensor("vgg_16/conv4/conv4_3/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv4_3',
                                                        initializer=self.reader.get_tensor("vgg_16/conv4/conv4_3/biases"),
                                                        trainable=True),
                                        name="conv4_3")
        pool4 = self._max_pooling(conv4_3, 2, 2, name="pool4")
        conv5_1 = self._load_conv_layer(pool4,
                                        tf.get_variable(name='kernel_conv5_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv5/conv5_1/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv5_1',
                                                        initializer=self.reader.get_tensor("vgg_16/conv5/conv5_1/biases"),
                                                        trainable=True),
                                        name="conv5_1")
        conv5_2 = self._load_conv_layer(conv5_1,
                                        tf.get_variable(name='kernel_conv5_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv5/conv5_2/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv5_2',
                                                        initializer=self.reader.get_tensor("vgg_16/conv5/conv5_2/biases"),
                                                        trainable=True),
                                        name="conv5_2")
        conv5_3 = self._load_conv_layer(conv5_2,
                                        tf.get_variable(name='kernel_conv5_3',
                                                        initializer=self.reader.get_tensor("vgg_16/conv5/conv5_3/weights"),
                                                        trainable=True),
                                        tf.get_variable(name='bias_conv5_3',
                                                        initializer=self.reader.get_tensor("vgg_16/conv5/conv5_3/biases"),
                                                        trainable=True),
                                        name="conv5_3")
        pool5 = self._max_pooling(conv5_3, 3, 1, 'pool5')
        conv6 = self._conv_layer(pool5, 1024, 3, 1, 'conv6', dilation_rate=2, activation=tf.nn.relu)
        conv7 = self._conv_layer(conv6, 1024, 1, 1, 'conv7', activation=tf.nn.relu)
        conv8_1 = self._conv_layer(conv7, 256, 1, 1, 'conv8_1', activation=tf.nn.relu)
        conv8_2 = self._conv_layer(conv8_1, 512, 3, 2, 'conv8_2', activation=tf.nn.relu)
        conv9_1 = self._conv_layer(conv8_2, 128, 1, 1, 'conv9_1', activation=tf.nn.relu)
        conv9_2 = self._conv_layer(conv9_1, 256, 3, 2, 'conv9_2', activation=tf.nn.relu)
        conv10_1 = self._conv_layer(conv9_2, 128, 1, 1, 'conv10_1', activation=tf.nn.relu)
        conv10_2 = self._conv_layer(conv10_1, 256, 3, 1, 'conv10_2', activation=tf.nn.relu)
        conv11_1 = self._conv_layer(conv10_2, 128, 1, 1, 'conv11_1', activation=tf.nn.relu)
        conv11_2 = self._conv_layer(conv11_1, 256, 3, 2, 'conv11_2', activation=tf.nn.relu)
        return conv4_3, conv7, conv8_2, conv9_2, conv10_2, conv11_2

    def _get_pbbox(self, pred):
        pred = tf.reshape(pred, [self.batch_size, -1, self.num_classes+4])
        pconf = pred[..., :self.num_classes]
        pbbox_yx = pred[..., self.num_classes:self.num_classes+2]
        pbbox_hw = pred[..., self.num_classes+2:]
        return pbbox_yx, pbbox_hw, pconf

    def _get_abbox(self, size, aspect_ratio, pshape):
        topleft_y = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)
        topleft_x = tf.range(0., tf.cast(pshape[2], tf.float32), dtype=tf.float32)
        topleft_y = tf.reshape(topleft_y, [-1, 1, 1, 1]) + 0.5
        topleft_x = tf.reshape(topleft_x, [1, -1, 1, 1]) + 0.5
        topleft_y = tf.tile(topleft_y, [1, pshape[2], 1, 1]) * tf.cast(self.input_size, tf.float32) / tf.cast(pshape[1], tf.float32)
        topleft_x = tf.tile(topleft_x, [pshape[1], 1, 1, 1]) * tf.cast(self.input_size, tf.float32) / tf.cast(pshape[2], tf.float32)
        topleft_yx = tf.concat([topleft_y, topleft_x], -1)
        topleft_yx = tf.tile(topleft_yx, [1, 1, len(aspect_ratio)+2, 1])

        priors = [[size[0], size[0]], [size[1], size[1]]]
        for i in range(len(aspect_ratio)):
            priors.append([size[0]*(aspect_ratio[i]**0.5), size[0]/(aspect_ratio[i]**0.5)])
        priors = tf.convert_to_tensor(priors, tf.float32)
        priors = tf.reshape(priors, [1, 1, -1, 2])

        abbox_y1x1 = tf.reshape(topleft_yx - priors / 2., [-1, 2])
        abbox_y2x2 = tf.reshape(topleft_yx + priors / 2., [-1, 2])
        abbox_yx = abbox_y1x1 / 2. + abbox_y2x2 / 2.
        abbox_hw = abbox_y2x2 - abbox_y1x1
        return abbox_y1x1, abbox_y2x2, abbox_yx, abbox_hw

    def _compute_one_image_loss(self, pbbox_yx, pbbox_hw, abbox_y1x1, abbox_y2x2,
                                abbox_yx, abbox_hw, pconf, ground_truth):
        slice_index = tf.argmin(ground_truth, axis=0)[0]
        ground_truth = tf.gather(ground_truth, tf.range(0, slice_index, dtype=tf.int64))
        gbbox_yx = ground_truth[..., 0:2]
        gbbox_hw = ground_truth[..., 2:4]
        gbbox_y1x1 = gbbox_yx - gbbox_hw / 2.
        gbbox_y2x2 = gbbox_yx + gbbox_hw / 2.
        class_id = tf.cast(ground_truth[..., 4:5], dtype=tf.int32)
        label = class_id

        abbox_hwti = tf.reshape(abbox_hw, [1, -1, 2])
        abbox_y1x1ti = tf.reshape(abbox_y1x1, [1, -1, 2])
        abbox_y2x2ti = tf.reshape(abbox_y2x2, [1, -1, 2])
        gbbox_hwti = tf.reshape(gbbox_hw, [-1, 1, 2])
        gbbox_y1x1ti = tf.reshape(gbbox_y1x1, [-1, 1, 2])
        gbbox_y2x2ti = tf.reshape(gbbox_y2x2, [-1, 1, 2])
        ashape = tf.shape(abbox_hwti)
        gshape = tf.shape(gbbox_hwti)
        abbox_hwti = tf.tile(abbox_hwti, [gshape[0], 1, 1])
        abbox_y1x1ti = tf.tile(abbox_y1x1ti, [gshape[0], 1, 1])
        abbox_y2x2ti = tf.tile(abbox_y2x2ti, [gshape[0], 1, 1])
        gbbox_hwti = tf.tile(gbbox_hwti, [1, ashape[1], 1])
        gbbox_y1x1ti = tf.tile(gbbox_y1x1ti, [1, ashape[1], 1])
        gbbox_y2x2ti = tf.tile(gbbox_y2x2ti, [1, ashape[1], 1])

        gaiou_y1x1ti = tf.maximum(abbox_y1x1ti, gbbox_y1x1ti)
        gaiou_y2x2ti = tf.minimum(abbox_y2x2ti, gbbox_y2x2ti)
        gaiou_area = tf.reduce_prod(tf.maximum(gaiou_y2x2ti - gaiou_y1x1ti, 0), axis=-1)
        aarea = tf.reduce_prod(abbox_hwti, axis=-1)
        garea = tf.reduce_prod(gbbox_hwti, axis=-1)
        gaiou_rate = gaiou_area / (aarea + garea - gaiou_area)

        best_raindex = tf.argmax(gaiou_rate, axis=1)
        best_pbbox_yx = tf.gather(pbbox_yx, best_raindex)
        best_pbbox_hw = tf.gather(pbbox_hw, best_raindex)
        best_pconf = tf.gather(pconf, best_raindex)
        best_abbox_yx = tf.gather(abbox_yx, best_raindex)
        best_abbox_hw = tf.gather(abbox_hw, best_raindex)

        bestmask, _ = tf.unique(best_raindex)
        bestmask = tf.contrib.framework.sort(bestmask)
        bestmask = tf.reshape(bestmask, [-1, 1])
        bestmask = tf.sparse.SparseTensor(tf.concat([bestmask, tf.zeros_like(bestmask)], axis=-1),
                                          tf.squeeze(tf.ones_like(bestmask)), dense_shape=[ashape[1], 1])
        bestmask = tf.reshape(tf.cast(tf.sparse.to_dense(bestmask), tf.float32), [-1])

        othermask = 1. - bestmask
        othermask = othermask > 0.
        other_pbbox_yx = tf.boolean_mask(pbbox_yx, othermask)
        other_pbbox_hw = tf.boolean_mask(pbbox_hw, othermask)
        other_pconf = tf.boolean_mask(pconf, othermask)

        other_abbox_yx = tf.boolean_mask(abbox_yx, othermask)
        other_abbox_hw = tf.boolean_mask(abbox_hw, othermask)

        agiou_rate = tf.transpose(gaiou_rate)
        other_agiou_rate = tf.boolean_mask(agiou_rate, othermask)
        best_agiou_rate = tf.reduce_max(other_agiou_rate, axis=1)
        pos_agiou_mask = best_agiou_rate > 0.5
        neg_agiou_mask = (1. - tf.cast(pos_agiou_mask, tf.float32)) > 0.
        rgindex = tf.argmax(other_agiou_rate, axis=1)
        pos_rgindex = tf.boolean_mask(rgindex, pos_agiou_mask)
        pos_pbbox_yx = tf.boolean_mask(other_pbbox_yx, pos_agiou_mask)
        pos_pbbox_hw = tf.boolean_mask(other_pbbox_hw, pos_agiou_mask)
        pos_pconf = tf.boolean_mask(other_pconf, pos_agiou_mask)
        pos_abbox_yx = tf.boolean_mask(other_abbox_yx, pos_agiou_mask)
        pos_abbox_hw = tf.boolean_mask(other_abbox_hw, pos_agiou_mask)
        pos_label = tf.gather(label, pos_rgindex)
        pos_gbbox_yx = tf.gather(gbbox_yx, pos_rgindex)
        pos_gbbox_hw = tf.gather(gbbox_hw, pos_rgindex)
        pos_shape = tf.shape(pos_pconf)

        neg_pconf = tf.boolean_mask(other_pconf, neg_agiou_mask)
        neg_abbox_yx = tf.boolean_mask(other_abbox_yx, neg_agiou_mask)
        neg_abbox_hw = tf.boolean_mask(other_abbox_hw, neg_agiou_mask)
        neg_abbox_y1x1y2x2 = tf.concat([neg_abbox_yx - neg_abbox_hw/2., neg_abbox_yx + neg_abbox_hw/2.], axis=-1)

        neg_shape = tf.shape(neg_pconf)
        num_pos = gshape[0] + pos_shape[0]
        num_neg = neg_shape[0]
        chosen_num_neg = tf.cond(num_neg > 3*num_pos, lambda: 3*num_pos, lambda: num_neg)
        neg_class_id = tf.constant([self.num_classes-1])
        neg_label = tf.tile(neg_class_id, [num_neg])

        total_neg_loss = tf.losses.sparse_softmax_cross_entropy(neg_label, neg_pconf, reduction=tf.losses.Reduction.NONE)
        selected_indices = tf.image.non_max_suppression(
            neg_abbox_y1x1y2x2, total_neg_loss, chosen_num_neg, iou_threshold=0.7
        )
        neg_loss = tf.reduce_mean(tf.gather(total_neg_loss, selected_indices))

        total_pos_pbbox_yx = tf.concat([best_pbbox_yx, pos_pbbox_yx], axis=0)
        total_pos_pbbox_hw = tf.concat([best_pbbox_hw, pos_pbbox_hw], axis=0)
        total_pos_pconf = tf.concat([best_pconf, pos_pconf], axis=0)
        total_pos_label = tf.concat([label, pos_label], axis=0)
        total_pos_gbbox_yx = tf.concat([gbbox_yx, pos_gbbox_yx], axis=0)
        total_pos_gbbox_hw = tf.concat([gbbox_hw, pos_gbbox_hw], axis=0)
        total_pos_abbox_yx = tf.concat([best_abbox_yx, pos_abbox_yx], axis=0)
        total_pos_abbox_hw = tf.concat([best_abbox_hw, pos_abbox_hw], axis=0)

        pos_conf_loss = tf.losses.sparse_softmax_cross_entropy(total_pos_label, total_pos_pconf, reduction=tf.losses.Reduction.MEAN)
        pos_truth_pbbox_yx = (total_pos_gbbox_yx - total_pos_abbox_yx) / total_pos_abbox_hw
        pos_truth_pbbox_hw = tf.log(total_pos_gbbox_hw / total_pos_abbox_hw)
        pos_yx_loss = tf.reduce_sum(self._smooth_l1_loss(total_pos_pbbox_yx - pos_truth_pbbox_yx), axis=-1)
        pos_hw_loss = tf.reduce_sum(self._smooth_l1_loss(total_pos_pbbox_hw - pos_truth_pbbox_hw), axis=-1)
        pos_coord_loss = tf.reduce_mean(pos_yx_loss + pos_hw_loss)

        total_loss = neg_loss + pos_conf_loss + pos_coord_loss
        return total_loss

    def _smooth_l1_loss(self, x):
        return tf.where(tf.abs(x) < 1., 0.5*x*x, tf.abs(x)-0.5)

    def _init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        if self.mode == 'train':
            self.sess.run(self.train_initializer)

    def _create_saver(self):
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()

    def _create_summary(self):
        with tf.variable_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def train_one_epoch(self, lr):
        self.is_training = True
        self.sess.run(self.train_initializer)
        mean_loss = []
        num_iters = self.num_train // self.batch_size
        for i in range(num_iters):
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.lr: lr})
            sys.stdout.write('\r>> ' + 'iters '+str(i)+str('/')+str(num_iters)+' loss '+str(loss))
            sys.stdout.flush()
            mean_loss.append(loss)
        sys.stdout.write('\n')
        mean_loss = np.mean(mean_loss)
        return mean_loss

    def test_one_image(self, images):
        self.is_training = False
        pred = self.sess.run(self.detection_pred, feed_dict={self.images: images})
        return pred

    def save_weight(self, mode, path):
        assert(mode in ['latest', 'best'])
        if mode == 'latest':
            saver = self.saver
        else:
            saver = self.best_saver
        if not tf.gfile.Exists(os.path.dirname(path)):
            tf.gfile.MakeDirs(os.path.dirname(path))
            print(os.path.dirname(path), 'does not exist, create it done')
        saver.save(self.sess, path, global_step=self.global_step)
        print('save', mode, 'model in', path, 'successfully')

    def load_weight(self, path):
        self.saver.restore(self.sess, path)
        print('load weight', path, 'successfully')

    def _bn(self, bottom):
        bn = tf.layers.batch_normalization(
            inputs=bottom,
            axis=3 if self.data_format == 'channels_last' else 1,
            training=self.is_training
        )
        return bn

    def _load_conv_layer(self, bottom, filters, bias, name):
        if self.data_format == 'channels_last':
            data_format = 'NHWC'
        else:
            data_format = 'NCHW'
        conv = tf.nn.conv2d(bottom, filter=filters, strides=[1, 1, 1, 1], name="kernel"+name, padding="SAME", data_format=data_format)
        conv_bias = tf.nn.bias_add(conv, bias=bias, name="bias"+name)
        return tf.nn.relu(conv_bias)

    def _conv_layer(self, bottom, filters, kernel_size, strides, name, dilation_rate=1, activation=None):
        conv = tf.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name=name,
            data_format=self.data_format,
            dilation_rate=dilation_rate,
        )
        bn = self._bn(conv)
        if activation is not None:
            bn = activation(bn)
        return bn

    def _max_pooling(self, bottom, pool_size, strides, name):
        return tf.layers.max_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )

    def _avg_pooling(self, bottom, pool_size, strides, name):
        return tf.layers.average_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )

    def _dropout(self, bottom, name):
        return tf.layers.dropout(
            inputs=bottom,
            rate=self.prob,
            training=self.is_training,
            name=name
        )
