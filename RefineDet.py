from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow as wrap
import sys
import os
import numpy as np


class RefineDet320:
    def __init__(self, config, data_provider):
        assert config['mode'] in ['train', 'test']
        assert config['data_format'] in ['channels_first', 'channels_last']
        self.config = config
        self.data_provider = data_provider
        self.input_size = config['input_size']
        if config['data_format'] == 'channels_last':
            self.data_shape = [self.input_size, self.input_size, 3]
        else:
            self.data_shape = [3, self.input_size, self.input_size]
        self.num_classes = config['num_classes'] + 1
        self.weight_decay = config['weight_decay']
        self.prob = 1. - config['keep_prob']
        self.data_format = config['data_format']
        self.mode = config['mode']
        self.batch_size = config['batch_size'] if config['mode'] == 'train' else 1
        self.anchor_ratios = [0.5, 1.0, 2.0]
        self.num_anchors = len(self.anchor_ratios)
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
            feat1, feat2, feat3, feat4, stride1, stride2, stride3, stride4 = self._feature_extractor(self.images)
            feat1 = tf.nn.l2_normalize(feat1, axis=3 if self.data_format == 'channels_last' else 1)
            feat1_norm_factor = tf.get_variable('feat1_l2_norm', initializer=tf.constant(10.))
            feat1 = feat1_norm_factor * feat1
            feat2 = tf.nn.l2_normalize(feat2, axis=3 if self.data_format == 'channels_last' else 1)
            feat2_norm_factor = tf.get_variable('feat2_l2_norm', initializer=tf.constant(8.))
            feat2 = feat2_norm_factor * feat2
        with tf.variable_scope('ARM'):
            arm1loc, arm1conf = self._arm(feat1, 'arm1')
            arm2loc, arm2conf = self._arm(feat2, 'arm2')
            arm3loc, arm3conf = self._arm(feat3, 'arm3')
            arm4loc, arm4conf = self._arm(feat4, 'arm4')
        with tf.variable_scope('TCB'):
            tcb4 = self._tcb(feat4, 'tcb4')
            tcb3 = self._tcb(feat3, 'tcb3', tcb4)
            tcb2 = self._tcb(feat2, 'tcb2', tcb3)
            tcb1 = self._tcb(feat1, 'tcb1', tcb2)
        with tf.variable_scope('ODM'):
            odm1loc, odm1conf = self._odm(tcb1, 'odm1')
            odm2loc, odm2conf = self._odm(tcb2, 'odm2')
            odm3loc, odm3conf = self._odm(tcb3, 'odm3')
            odm4loc, odm4conf = self._odm(tcb4, 'odm4')
        with tf.variable_scope('inference'):
            if self.data_format == 'channels_first':
                arm1loc = tf.transpose(arm1loc, [0, 2, 3, 1])
                arm1conf = tf.transpose(arm1conf, [0, 2, 3, 1])
                arm2loc = tf.transpose(arm2loc, [0, 2, 3, 1])
                arm2conf = tf.transpose(arm2conf, [0, 2, 3, 1])
                arm3loc = tf.transpose(arm3loc, [0, 2, 3, 1])
                arm3conf = tf.transpose(arm3conf, [0, 2, 3, 1])
                arm4loc = tf.transpose(arm4loc, [0, 2, 3, 1])
                arm4conf = tf.transpose(arm4conf, [0, 2, 3, 1])
                odm1loc = tf.transpose(odm1loc, [0, 2, 3, 1])
                odm1conf = tf.transpose(odm1conf, [0, 2, 3, 1])
                odm2loc = tf.transpose(odm2loc, [0, 2, 3, 1])
                odm2conf = tf.transpose(odm2conf, [0, 2, 3, 1])
                odm3loc = tf.transpose(odm3loc, [0, 2, 3, 1])
                odm3conf = tf.transpose(odm3conf, [0, 2, 3, 1])
                odm4loc = tf.transpose(odm4loc, [0, 2, 3, 1])
                odm4conf = tf.transpose(odm4conf, [0, 2, 3, 1])
            p1shape = tf.shape(arm1loc)
            p2shape = tf.shape(arm2loc)
            p3shape = tf.shape(arm3loc)
            p4shape = tf.shape(arm4loc)
            arm1pbbox_yx, arm1pbbox_hw, arm1pconf = self._get_armpbbox(arm1loc, arm1conf)
            arm2pbbox_yx, arm2pbbox_hw, arm2pconf = self._get_armpbbox(arm2loc, arm2conf)
            arm3pbbox_yx, arm3pbbox_hw, arm3pconf = self._get_armpbbox(arm3loc, arm3conf)
            arm4pbbox_yx, arm4pbbox_hw, arm4pconf = self._get_armpbbox(arm4loc, arm4conf)

            odm1pbbox_yx, odm1pbbox_hw, odm1pconf = self._get_odmpbbox(odm1loc, odm1conf)
            odm2pbbox_yx, odm2pbbox_hw, odm2pconf = self._get_odmpbbox(odm2loc, odm2conf)
            odm3pbbox_yx, odm3pbbox_hw, odm3pconf = self._get_odmpbbox(odm3loc, odm3conf)
            odm4pbbox_yx, odm4pbbox_hw, odm4pconf = self._get_odmpbbox(odm4loc, odm4conf)

            a1bbox_y1x1, a1bbox_y2x2, a1bbox_yx, a1bbox_hw = self._get_abbox(stride1*4, stride1, p1shape)
            a2bbox_y1x1, a2bbox_y2x2, a2bbox_yx, a2bbox_hw = self._get_abbox(stride2*4, stride2, p2shape)
            a3bbox_y1x1, a3bbox_y2x2, a3bbox_yx, a3bbox_hw = self._get_abbox(stride3*4, stride3, p3shape)
            a4bbox_y1x1, a4bbox_y2x2, a4bbox_yx, a4bbox_hw = self._get_abbox(stride4*4, stride4, p4shape)

            armpbbox_yx = tf.concat([arm1pbbox_yx, arm2pbbox_yx, arm3pbbox_yx, arm4pbbox_yx], axis=1)
            armpbbox_hw = tf.concat([arm1pbbox_hw, arm2pbbox_hw, arm3pbbox_hw, arm4pbbox_hw], axis=1)
            armpconf = tf.concat([arm1pconf, arm2pconf, arm3pconf, arm4pconf], axis=1)
            odmpbbox_yx = tf.concat([odm1pbbox_yx, odm2pbbox_yx, odm3pbbox_yx, odm4pbbox_yx], axis=1)
            odmpbbox_hw = tf.concat([odm1pbbox_hw, odm2pbbox_hw, odm3pbbox_hw, odm4pbbox_hw], axis=1)
            odmpconf = tf.concat([odm1pconf, odm2pconf, odm3pconf, odm4pconf], axis=1)
            abbox_y1x1 = tf.concat([a1bbox_y1x1, a2bbox_y1x1, a3bbox_y1x1, a4bbox_y1x1], axis=0)
            abbox_y2x2 = tf.concat([a1bbox_y2x2, a2bbox_y2x2, a3bbox_y2x2, a4bbox_y2x2], axis=0)
            abbox_yx = tf.concat([a1bbox_yx, a2bbox_yx, a3bbox_yx, a4bbox_yx], axis=0)
            abbox_hw = tf.concat([a1bbox_hw, a2bbox_hw, a3bbox_hw, a4bbox_hw], axis=0)
            if self.mode == 'train':
                i = 0.
                loss = 0.
                cond = lambda loss, i: tf.less(i, tf.cast(self.batch_size, tf.float32))
                body = lambda loss, i: (
                    tf.add(loss, self._compute_one_image_loss(
                        tf.squeeze(tf.gather(armpbbox_yx, tf.cast(i, tf.int32))),
                        tf.squeeze(tf.gather(armpbbox_hw, tf.cast(i, tf.int32))),
                        tf.squeeze(tf.gather(armpconf, tf.cast(i, tf.int32))),
                        tf.squeeze(tf.gather(odmpbbox_yx, tf.cast(i, tf.int32))),
                        tf.squeeze(tf.gather(odmpbbox_hw, tf.cast(i, tf.int32))),
                        tf.squeeze(tf.gather(odmpconf, tf.cast(i, tf.int32))),
                        abbox_y1x1,
                        abbox_y2x2,
                        abbox_yx,
                        abbox_hw,
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
                self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            else:
                armconft = tf.nn.softmax(armpconf[0, ...])
                odmconft = tf.nn.softmax(odmpconf[0, ...])
                armmask = armconft[:, 1] < 0.99
                odmmask = tf.argmax(odmconft, axis=-1) < self.num_classes - 1
                mask = (tf.cast(armmask, tf.float32) * tf.cast(odmmask, tf.float32)) > 0.
                armpbbox_yxt = tf.boolean_mask(armpbbox_yx[0, ...], mask)
                armpbbox_hwt = tf.boolean_mask(armpbbox_hw[0, ...], mask)
                odmpbbox_yxt = tf.boolean_mask(odmpbbox_yx[0, ...], mask)
                odmpbbox_hwt = tf.boolean_mask(odmpbbox_hw[0, ...], mask)
                abbox_yxt = tf.boolean_mask(abbox_yx, mask)
                abbox_hwt = tf.boolean_mask(abbox_hw, mask)
                odmconft = tf.boolean_mask(odmconft, mask)
                confidence = odmconft[..., :self.num_classes-1]

                arm_yx = armpbbox_yxt * abbox_hwt + abbox_yxt
                arm_hw = tf.exp(armpbbox_hwt) * abbox_hwt
                odm_yx = odmpbbox_yxt * arm_hw + arm_yx
                odm_hw = tf.exp(odmpbbox_hwt) * arm_hw

                odm_y1x1 = odm_yx - odm_hw / 2.
                odm_y2x2 = odm_yx + odm_hw / 2.
                odm_y1x1y2x2 = tf.concat([odm_y1x1, odm_y2x2], axis=-1)
                filter_mask = tf.greater_equal(confidence, self.nms_score_threshold)
                scores = []
                class_id = []
                bbox = []
                for i in range(self.num_classes-1):
                    scoresi = tf.boolean_mask(confidence[:, i], filter_mask[:, i])
                    bboxi = tf.boolean_mask(odm_y1x1y2x2, filter_mask[:, i])
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
        conv9_1 = self._conv_layer(conv8_2, 256, 1, 1, 'conv9_1', activation=tf.nn.relu)
        conv9_2 = self._conv_layer(conv9_1, 512, 3, 2, 'conv9_2', activation=tf.nn.relu)
        conv10_1 = self._conv_layer(conv9_2, 256, 1, 1, 'conv10_1', activation=tf.nn.relu)
        conv10_2 = self._conv_layer(conv10_1, 256, 3, 1, 'conv10_2', activation=tf.nn.relu)
        stride1 = 8
        stride2 = 16
        stride3 = 32
        stride4 = 64
        return conv4_3, conv5_3, conv8_2, conv10_2, stride1, stride2, stride3, stride4

    def _arm(self, bottom, scope):
        with tf.variable_scope(scope):
            conv1 = self._conv_layer(bottom, 256, 3, 1, activation=tf.nn.relu)
            conv2 = self._conv_layer(conv1, 256, 3, 1, activation=tf.nn.relu)
            conv3 = self._conv_layer(conv2, 256, 3, 1, activation=tf.nn.relu)
            conv4 = self._conv_layer(conv3, 256, 3, 1, activation=tf.nn.relu)
            ploc = self._conv_layer(conv4, 4*self.num_anchors, 3, 1)
            pconf = self._conv_layer(conv4, 2*self.num_anchors, 3, 1)
            return ploc, pconf

    def _tcb(self, bottom, scope, high_level_feat=None):
        with tf.variable_scope(scope):
            conv1 = self._conv_layer(bottom, 256, 3, 1, activation=tf.nn.relu)
            conv2 = self._conv_layer(conv1, 256, 3, 1)
            if high_level_feat is not None:
                dconv = self._dconv_layer(high_level_feat, 256, 4, 2)
                conv2 = tf.nn.relu(conv2 + dconv)
            conv3 = tf.nn.relu(conv2)
            return conv3

    def _odm(self, bottom, scope):
        with tf.variable_scope(scope):
            conv1 = self._conv_layer(bottom, 256, 3, 1, activation=tf.nn.relu)
            conv2 = self._conv_layer(conv1, 256, 3, 1, activation=tf.nn.relu)
            conv3 = self._conv_layer(conv2, 256, 3, 1, activation=tf.nn.relu)
            conv4 = self._conv_layer(conv3, 256, 3, 1, activation=tf.nn.relu)
            ploc = self._conv_layer(conv4, 4*self.num_anchors, 3, 1)
            pconf = self._conv_layer(conv4, self.num_classes*self.num_anchors, 3, 1)
            return ploc, pconf

    def _get_armpbbox(self, ploc, pconf):
        pconf = tf.reshape(pconf, [self.batch_size, -1, 2])
        ploc = tf.reshape(ploc, [self.batch_size, -1, 4])
        pbbox_yx = ploc[..., :2]
        pbbox_hw = ploc[..., 2:]
        return pbbox_yx, pbbox_hw, pconf

    def _get_odmpbbox(self, ploc, pconf):
        pconf = tf.reshape(pconf, [self.batch_size, -1, self.num_classes])
        ploc = tf.reshape(ploc, [self.batch_size, -1, 4])
        pbbox_yx = ploc[..., :2]
        pbbox_hw = ploc[..., 2:]
        return pbbox_yx, pbbox_hw, pconf

    def _get_abbox(self, size, stride, pshape):
        topleft_y = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)
        topleft_x = tf.range(0., tf.cast(pshape[2], tf.float32), dtype=tf.float32)
        topleft_y = tf.reshape(topleft_y, [-1, 1, 1, 1]) + 0.5
        topleft_x = tf.reshape(topleft_x, [1, -1, 1, 1]) + 0.5
        topleft_y = tf.tile(topleft_y, [1, pshape[2], 1, 1]) * stride
        topleft_x = tf.tile(topleft_x, [pshape[1], 1, 1, 1]) * stride
        topleft_yx = tf.concat([topleft_y, topleft_x], -1)
        topleft_yx = tf.tile(topleft_yx, [1, 1, self.num_anchors, 1])

        priors = []
        for ratio in self.anchor_ratios:
            priors.append([size*(ratio**0.5), size/(ratio**0.5)])
        priors = tf.convert_to_tensor(priors, tf.float32)
        priors = tf.reshape(priors, [1, 1, -1, 2])

        abbox_y1x1 = tf.reshape(topleft_yx - priors / 2., [-1, 2])
        abbox_y2x2 = tf.reshape(topleft_yx + priors / 2., [-1, 2])
        abbox_yx = abbox_y1x1 / 2. + abbox_y2x2 / 2.
        abbox_hw = abbox_y2x2 - abbox_y1x1
        return abbox_y1x1, abbox_y2x2, abbox_yx, abbox_hw

    def _compute_one_image_loss(self, armpbbox_yx, armpbbox_hw, armpconf,
                                odmpbbox_yx, odmpbbox_hw, odmpconf,
                                abbox_y1x1, abbox_y2x2,
                                abbox_yx, abbox_hw,  ground_truth):
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
        best_armpbbox_yx = tf.gather(armpbbox_yx, best_raindex)
        best_armpbbox_hw = tf.gather(armpbbox_hw, best_raindex)
        best_armpconf = tf.gather(armpconf, best_raindex)
        best_odmpbbox_yx = tf.gather(odmpbbox_yx, best_raindex)
        best_odmpbbox_hw = tf.gather(odmpbbox_hw, best_raindex)
        best_odmpconf = tf.gather(odmpconf, best_raindex)
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
        other_armpbbox_yx = tf.boolean_mask(armpbbox_yx, othermask)
        other_armpbbox_hw = tf.boolean_mask(armpbbox_hw, othermask)
        other_armpconf = tf.boolean_mask(armpconf, othermask)
        other_odmpbbox_yx = tf.boolean_mask(odmpbbox_yx, othermask)
        other_odmpbbox_hw = tf.boolean_mask(odmpbbox_hw, othermask)
        other_odmpconf = tf.boolean_mask(odmpconf, othermask)
        other_abbox_yx = tf.boolean_mask(abbox_yx, othermask)
        other_abbox_hw = tf.boolean_mask(abbox_hw, othermask)

        agiou_rate = tf.transpose(gaiou_rate)
        other_agiou_rate = tf.boolean_mask(agiou_rate, othermask)
        max_agiou_rate = tf.reduce_max(other_agiou_rate, axis=1)
        pos_agiou_mask = max_agiou_rate > 0.5
        neg_agiou_mask = max_agiou_rate < 0.4
        rgindex = tf.argmax(other_agiou_rate, axis=1)
        pos_rgindex = tf.boolean_mask(rgindex, pos_agiou_mask)
        pos_armppox_yx = tf.boolean_mask(other_armpbbox_yx, pos_agiou_mask)
        pos_armppox_hw = tf.boolean_mask(other_armpbbox_hw, pos_agiou_mask)
        pos_armpconf = tf.boolean_mask(other_armpconf, pos_agiou_mask)
        pos_odmppox_yx = tf.boolean_mask(other_odmpbbox_yx, pos_agiou_mask)
        pos_odmppox_hw = tf.boolean_mask(other_odmpbbox_hw, pos_agiou_mask)
        pos_odmpconf = tf.boolean_mask(other_odmpconf, pos_agiou_mask)
        pos_abbox_yx = tf.boolean_mask(other_abbox_yx, pos_agiou_mask)
        pos_abbox_hw = tf.boolean_mask(other_abbox_hw, pos_agiou_mask)
        pos_odmlabel = tf.gather(label, pos_rgindex)
        pos_gbbox_yx = tf.gather(gbbox_yx, pos_rgindex)
        pos_gbbox_hw = tf.gather(gbbox_hw, pos_rgindex)
        neg_armpconf = tf.boolean_mask(other_armpconf, neg_agiou_mask)
        neg_armabbox_yx = tf.boolean_mask(other_abbox_yx, neg_agiou_mask)
        neg_armabbox_hw = tf.boolean_mask(other_abbox_hw, neg_agiou_mask)
        neg_armabbox_y1x1y2x2 = tf.concat([neg_armabbox_yx - neg_armabbox_hw/2., neg_armabbox_yx + neg_armabbox_hw/2.], axis=-1)
        neg_odmpconf = tf.boolean_mask(other_odmpconf, neg_agiou_mask)

        total_pos_armpbbox_yx = tf.concat([best_armpbbox_yx, pos_armppox_yx], axis=0)
        total_pos_armpbbox_hw = tf.concat([best_armpbbox_hw, pos_armppox_hw], axis=0)
        total_pos_armpconf = tf.concat([best_armpconf, pos_armpconf], axis=0)
        total_pos_odmpbbox_yx = tf.concat([best_odmpbbox_yx, pos_odmppox_yx], axis=0)
        total_pos_odmpbbox_hw = tf.concat([best_odmpbbox_hw, pos_odmppox_hw], axis=0)
        total_pos_odmpconf = tf.concat([best_odmpconf, pos_odmpconf], axis=0)
        total_pos_odmlabel = tf.concat([label, pos_odmlabel], axis=0)
        total_pos_gbbox_yx = tf.concat([gbbox_yx, pos_gbbox_yx], axis=0)
        total_pos_gbbox_hw = tf.concat([gbbox_hw, pos_gbbox_hw], axis=0)
        total_pos_abbox_yx = tf.concat([best_abbox_yx, pos_abbox_yx], axis=0)
        total_pos_abbox_hw = tf.concat([best_abbox_hw, pos_abbox_hw], axis=0)

        num_pos = tf.shape(total_pos_odmlabel)[0]
        num_armneg = tf.shape(neg_armpconf)[0]
        chosen_num_armneg = tf.cond(num_armneg > 3*num_pos, lambda: 3*num_pos, lambda: num_armneg)
        neg_armclass_id = tf.constant([1])
        pos_armclass_id = tf.constant([0])
        neg_armlabel = tf.tile(neg_armclass_id, [num_armneg])
        pos_armlabel = tf.tile(pos_armclass_id, [num_pos])
        total_neg_armloss = tf.losses.sparse_softmax_cross_entropy(neg_armlabel, neg_armpconf, reduction=tf.losses.Reduction.NONE)
        selected_armindices = tf.image.non_max_suppression(
            neg_armabbox_y1x1y2x2, total_neg_armloss, chosen_num_armneg, iou_threshold=0.7
        )
        neg_armloss = tf.reduce_mean(tf.gather(total_neg_armloss, selected_armindices))

        chosen_neg_armpconf = tf.gather(neg_armpconf, selected_armindices)
        chosen_neg_odmpconf = tf.gather(neg_odmpconf, selected_armindices)

        neg_odm_mask = chosen_neg_armpconf[:, 1] < 0.99
        chosen_neg_odmpconf = tf.boolean_mask(chosen_neg_odmpconf, neg_odm_mask)
        chosen_num_odmneg = tf.shape(chosen_neg_odmpconf)[0]
        neg_odmclass_id = tf.constant([self.num_classes-1])
        neg_odmlabel = tf.tile(neg_odmclass_id, [chosen_num_odmneg])
        neg_odmloss = tf.losses.sparse_softmax_cross_entropy(neg_odmlabel, chosen_neg_odmpconf, reduction=tf.losses.Reduction.MEAN)

        pos_armconf_loss = tf.losses.sparse_softmax_cross_entropy(pos_armlabel, total_pos_armpconf, reduction=tf.losses.Reduction.MEAN)
        pos_truth_armpbbox_yx = (total_pos_gbbox_yx - total_pos_abbox_yx) / total_pos_abbox_hw
        pos_truth_armpbbox_hw = tf.log(total_pos_gbbox_hw / total_pos_abbox_hw)
        pos_yx_armloss = tf.reduce_sum(self._smooth_l1_loss(total_pos_armpbbox_yx - pos_truth_armpbbox_yx), axis=-1)
        pos_hw_armloss = tf.reduce_sum(self._smooth_l1_loss(total_pos_armpbbox_hw - pos_truth_armpbbox_hw), axis=-1)
        pos_coord_armloss = tf.reduce_mean(pos_yx_armloss + pos_hw_armloss)

        arm_yx = total_pos_armpbbox_yx * total_pos_abbox_hw + total_pos_abbox_yx
        arm_hw = tf.exp(total_pos_armpbbox_hw) * total_pos_abbox_hw

        pos_odmconf_loss = tf.losses.sparse_softmax_cross_entropy(total_pos_odmlabel, total_pos_odmpconf, reduction=tf.losses.Reduction.MEAN)
        pos_truth_odmpbbox_yx = (total_pos_gbbox_yx - arm_yx) / arm_hw
        pos_truth_odmpbbox_hw = tf.log(total_pos_gbbox_hw / arm_hw)
        pos_yx_odmloss = tf.reduce_sum(self._smooth_l1_loss(total_pos_odmpbbox_yx - pos_truth_odmpbbox_yx), axis=-1)
        pos_hw_odmloss = tf.reduce_sum(self._smooth_l1_loss(total_pos_odmpbbox_hw - pos_truth_odmpbbox_hw), axis=-1)
        pos_coord_odmloss = tf.reduce_mean(pos_yx_odmloss + pos_hw_odmloss)

        armloss = neg_armloss + pos_armconf_loss + pos_coord_armloss
        odmloss = neg_odmloss + pos_odmconf_loss + pos_coord_odmloss
        return armloss + odmloss

    def _smooth_l1_loss(self, x):
        return tf.where(tf.abs(x) < 1., 0.5*x*x, tf.abs(x)-0.5)

    def _init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        if self.mode == 'train':
            self.sess.run(self.train_initializer)

    def _create_saver(self):
        weights = tf.trainable_variables()
        self.saver = tf.train.Saver(weights)
        self.best_saver = tf.train.Saver(weights)

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
            sys.stdout.write('\r>> ' + 'iters '+str(i+1)+str('/')+str(num_iters)+' loss '+str(loss))
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

    def _conv_layer(self, bottom, filters, kernel_size, strides, name=None, dilation_rate=1, activation=None):
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

    def _dconv_layer(self, bottom, filters, kernel_size, strides, name=None, activation=None):
        conv = tf.layers.conv2d_transpose(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name=name,
            data_format=self.data_format,
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
