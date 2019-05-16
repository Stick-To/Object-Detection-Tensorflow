from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import sys
import numpy as np
import math


class RetinaNet:
    def __init__(self, config, data_provider):

        assert len(config['data_shape']) == 3
        assert config['mode'] in ['train', 'test']
        assert config['data_format'] in ['channels_first', 'channels_last']
        self.config = config
        self.data_provider = data_provider

        self.init_conv_kernel_size = 7
        self.init_conv_strides = 2  # must be 2 for construct pyramid
        self.init_pooling_pool_size = 3
        self.init_pooling_strides = 2  # must be 2 for construct pyramid

        self.is_bottleneck = config['is_bottleneck']
        self.block_list = config['residual_block_list']
        self.filters_list = [self.init_conv_kernel_size * (2 ** i) for i in range(len(config['residual_block_list']))]
        self.is_pretraining = config['is_pretraining']
        self.data_shape = config['data_shape']
        self.num_classes = config['num_classes'] + 1
        self.weight_decay = config['weight_decay']
        self.prob = 1. - config['keep_prob']
        self.data_format = config['data_format']
        self.mode = config['mode']
        self.batch_size = config['batch_size'] if config['mode'] == 'train' else 1
        self.gamma = config['gamma']
        self.alpha = config['alpha']

        self.anchors = [32, 64, 128, 256, 512]
        self.aspect_ratios = [1, 1/2, 2]
        self.anchor_size = [2**0, 2**(1/3), 2**(2/3)]
        self.num_anchors = len(self.aspect_ratios) * len(self.anchor_size)
        self.nms_score_threshold = config['nms_score_threshold']
        self.nms_max_boxes = config['nms_max_boxes']
        self.nms_iou_threshold = config['nms_iou_threshold']

        # pi for initialize final conv layer for classifier
        self.pi = 0.01

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

        if self.is_pretraining:
            self._define_pretraining_inputs()
            self._build_pretraining_graph()
            self._create_pretraining_saver()
            self.save_weight = self._save_pretraining_weight
            self.train_one_epoch = self._train_pretraining_epoch
            self.test_one_image = self._test_one_pretraining_image
            if self.mode == 'train':
                self._create_pretraining_summary()
        else:
            self._define_detection_inputs()
            self._build_detection_graph()
            self._create_detection_saver()
            self.save_weight = self._save_detection_weight
            self.train_one_epoch = self._train_detection_epoch
            self.test_one_image = self._test_one_detection_image
            if self.mode == 'train':
                self._create_detection_summary()
        self._init_session()

    def _define_pretraining_inputs(self):
        shape = [self.batch_size]
        shape.extend(self.data_shape)
        mean = tf.convert_to_tensor([123.68, 116.779, 103.979], dtype=tf.float32)
        if self.data_format == 'channels_last':
            mean = tf.reshape(mean, [1, 1, 1, 3])
        else:
            mean = tf.reshape(mean, [1, 3, 1, 1])
        if self.mode == 'train':
            self.images, self.labels = self.train_iterator.get_next()
            self.images.set_shape(shape)
            self.images = self.images - mean
            self.labels = tf.cast(self.labels, tf.int32)
        else:
            self.images = tf.placeholder(tf.float32, shape, name='images')
            self.images = self.images - mean
            self.labels = tf.placeholder(tf.int32, [self.batch_size, 1], name='labels')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

    def _define_detection_inputs(self):
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

    def _build_pretraining_graph(self):
        with tf.variable_scope('feature_extractor'):
            _, _, features = self._feature_extractor(self.images)
        with tf.variable_scope('pretraining'):
            axes = [1, 2] if self.data_format == 'channels_last' else [2, 3]
            global_pool = tf.reduce_mean(features, axis=axes, name='global_pool')
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, global_pool, reduction=tf.losses.Reduction.MEAN)
            self.pred = tf.argmax(global_pool, 1)
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.pred, self.labels), tf.float32), name='accuracy'
            )
            self.loss = loss + self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables('feature_extractor')]
            )
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def _build_detection_graph(self):
        with tf.variable_scope('feature_extractor'):
            feat1, feat2, feat3 = self._feature_extractor(self.images)
            p5 = self._get_pyramid(feat3, 256)
            p4, top_down = self._get_pyramid(feat2, 256, p5)
            p3, _ = self._get_pyramid(feat1, 256, top_down)
            p6 = self._get_pyramid(p5, 256)
            p7 = self._get_pyramid(p6, 256)
        with tf.variable_scope('regressor'):
            pred3c = self._classification_subnet(p3, 256)
            pred3r = self._regression_subnet(p3, 256)
            pred4c = self._classification_subnet(p4, 256)
            pred4r = self._regression_subnet(p4, 256)
            pred5c = self._classification_subnet(p5, 256)
            pred5r = self._regression_subnet(p5, 256)
            pred6c = self._classification_subnet(p6, 256)
            pred6r = self._regression_subnet(p6, 256)
            pred7c = self._classification_subnet(p7, 256)
            pred7r = self._regression_subnet(p7, 256)
            if self.data_format == 'channels_first':
                pred3c = tf.transpose(pred3c, [0, 2, 3, 1])
                pred3r = tf.transpose(pred3r, [0, 2, 3, 1])
                pred4c = tf.transpose(pred4c, [0, 2, 3, 1])
                pred4r = tf.transpose(pred4r, [0, 2, 3, 1])
                pred5c = tf.transpose(pred5c, [0, 2, 3, 1])
                pred5r = tf.transpose(pred5r, [0, 2, 3, 1])
                pred6c = tf.transpose(pred6c, [0, 2, 3, 1])
                pred6r = tf.transpose(pred6r, [0, 2, 3, 1])
                pred7c = tf.transpose(pred7c, [0, 2, 3, 1])
                pred7r = tf.transpose(pred7r, [0, 2, 3, 1])
            p3shape = tf.shape(pred3c)
            p4shape = tf.shape(pred4c)
            p5shape = tf.shape(pred5c)
            p6shape = tf.shape(pred6c)
            p7shape = tf.shape(pred7c)
        with tf.variable_scope('inference'):
            p3bbox_yx, p3bbox_hw, p3conf = self._get_pbbox(pred3c, pred3r)
            p4bbox_yx, p4bbox_hw, p4conf = self._get_pbbox(pred4c, pred4r)
            p5bbox_yx, p5bbox_hw, p5conf = self._get_pbbox(pred5c, pred5r)
            p6bbox_yx, p6bbox_hw, p6conf = self._get_pbbox(pred6c, pred6r)
            p7bbox_yx, p7bbox_hw, p7conf = self._get_pbbox(pred7c, pred7r)

            a3bbox_y1x1, a3bbox_y2x2, a3bbox_yx, a3bbox_hw = self._get_abbox(self.anchors[0], p3shape)
            a4bbox_y1x1, a4bbox_y2x2, a4bbox_yx, a4bbox_hw = self._get_abbox(self.anchors[1], p4shape)
            a5bbox_y1x1, a5bbox_y2x2, a5bbox_yx, a5bbox_hw = self._get_abbox(self.anchors[2], p5shape)
            a6bbox_y1x1, a6bbox_y2x2, a6bbox_yx, a6bbox_hw = self._get_abbox(self.anchors[3], p6shape)
            a7bbox_y1x1, a7bbox_y2x2, a7bbox_yx, a7bbox_hw = self._get_abbox(self.anchors[4], p7shape)

            pbbox_yx = tf.concat([p3bbox_yx, p4bbox_yx, p5bbox_yx, p6bbox_yx, p7bbox_yx], axis=1)
            pbbox_hw = tf.concat([p3bbox_hw, p4bbox_hw, p5bbox_hw, p6bbox_hw, p7bbox_hw], axis=1)
            pconf = tf.concat([p3conf, p4conf, p5conf, p6conf, p7conf], axis=1)
            abbox_y1x1 = tf.concat([a3bbox_y1x1, a4bbox_y1x1, a5bbox_y1x1, a6bbox_y1x1, a7bbox_y1x1], axis=0)
            abbox_y2x2 = tf.concat([a3bbox_y2x2, a4bbox_y2x2, a5bbox_y2x2, a6bbox_y2x2, a7bbox_y2x2], axis=0)
            abbox_yx = tf.concat([a3bbox_yx, a4bbox_yx, a5bbox_yx, a6bbox_yx, a7bbox_yx], axis=0)
            abbox_hw = tf.concat([a3bbox_hw, a4bbox_hw, a5bbox_hw, a6bbox_hw, a7bbox_hw], axis=0)

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
                    [tf.nn.l2_loss(var) for var in tf.trainable_variables('feature_extractor')]
                ) + self.weight_decay * tf.add_n(
                    [tf.nn.l2_loss(var) for var in tf.trainable_variables('regressor')]
                )
                self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
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

    def _feature_extractor(self, image):
        endpoints = []
        conv1_1 = self._conv_bn_activation(
            bottom=image,
            filters=self.config['init_conv_filters'],
            kernel_size=self.init_conv_kernel_size,
            strides=self.init_conv_strides,
            )
        pool1 = self._max_pooling(
            bottom=conv1_1,
            pool_size=self.init_pooling_pool_size,
            strides=self.init_pooling_strides,
            name='pool1'
        )
        if self.is_bottleneck:
            stack_residual_unit_fn = self._residual_bottleneck
        else:
            stack_residual_unit_fn = self._residual_block
        residual_block = pool1
        for i in range(self.block_list[0]):
            residual_block = stack_residual_unit_fn(residual_block, self.filters_list[0], 1, 'block1_unit'+str(i+1))
        endpoints.append(residual_block)
        for i in range(1, len(self.block_list)):
            residual_block = stack_residual_unit_fn(residual_block, self.filters_list[i], 2, 'block'+str(i+1)+'_unit'+str(1))
            for j in range(1, self.block_list[i]):
                residual_block = stack_residual_unit_fn(residual_block, self.filters_list[i], 1, 'block'+str(i+1)+'_unit'+str(j+1))
            endpoints.append(residual_block)
        return endpoints[-3], endpoints[-2], endpoints[-1]

    def _classification_subnet(self, bottom, filters):
        conv1 = self._bn_activation_conv(bottom, filters, 3, 1)
        conv2 = self._bn_activation_conv(conv1, filters, 3, 1)
        conv3 = self._bn_activation_conv(conv2, filters, 3, 1)
        conv4 = self._bn_activation_conv(conv3, filters, 3, 1)
        pred = self._bn_activation_conv(conv4, self.num_classes * self.num_anchors, 3, 1, pi_init=True)
        return pred

    def _regression_subnet(self, bottom, filters):
        conv1 = self._bn_activation_conv(bottom, filters, 3, 1)
        conv2 = self._bn_activation_conv(conv1, filters, 3, 1)
        conv3 = self._bn_activation_conv(conv2, filters, 3, 1)
        conv4 = self._bn_activation_conv(conv3, filters, 3, 1)
        pred = self._bn_activation_conv(conv4, 4 * self.num_anchors, 3, 1)
        return pred

    def _get_pyramid(self, feat, feature_size, top_feat=None):
        if top_feat is None:
            return self._bn_activation_conv(feat, feature_size, 3, 1)
        else:
            if self.data_format == 'channels_last':
                feat = self._bn_activation_conv(feat, feature_size, 1, 1)
                top_feat = tf.image.resize_bilinear(top_feat, [tf.shape(feat)[1], tf.shape(feat)[2]])
                total_feat = feat + top_feat
                return self._bn_activation_conv(total_feat, feature_size, 3, 1), total_feat
            else:
                feat = self._bn_activation_conv(feat, feature_size, 1, 1)
                feat = tf.transpose(feat, [0, 2, 3, 1])
                top_feat = tf.transpose(top_feat, [0, 2, 3, 1])
                top_feat = tf.image.resize_bilinear(top_feat, [tf.shape(feat)[1], tf.shape(feat)[2]])
                total_feat = feat + top_feat
                total_feat = tf.transpose(total_feat, [0, 3, 1, 2])
                return self._bn_activation_conv(total_feat, feature_size, 3, 1), total_feat

    def _get_pbbox(self, predc, predr):
        pconf = tf.reshape(predc, [self.batch_size, -1, self.num_classes])
        pbbox = tf.reshape(predr, [self.batch_size, -1, 4])
        pbbox_yx = pbbox[..., :2]
        pbbox_hw = pbbox[..., 2:]
        return pbbox_yx, pbbox_hw, pconf

    def _get_abbox(self, size, pshape):
        if self.data_format == 'channels_last':
            input_h = self.data_shape[1]
            downsampling_rate = tf.cast(input_h, tf.float32) / tf.cast(pshape[1], tf.float32)
        else:
            input_h = self.data_shape[2]
            downsampling_rate = tf.cast(input_h, tf.float32) / tf.cast(pshape[1], tf.float32)
        topleft_y = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)
        topleft_x = tf.range(0., tf.cast(pshape[2], tf.float32), dtype=tf.float32)
        topleft_y = tf.reshape(topleft_y, [-1, 1, 1, 1]) + 0.5
        topleft_x = tf.reshape(topleft_x, [1, -1, 1, 1]) + 0.5
        topleft_y = tf.tile(topleft_y, [1, pshape[2], 1, 1]) * downsampling_rate
        topleft_x = tf.tile(topleft_x, [pshape[1], 1, 1, 1]) * downsampling_rate
        topleft_yx = tf.concat([topleft_y, topleft_x], -1)
        topleft_yx = tf.tile(topleft_yx, [1, 1, self.num_anchors, 1])

        priors = []
        for r in self.aspect_ratios:
            for s in self.anchor_size:
                priors.append([s*size*(r**0.5), s*size/(r**0.5)])
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
        neg_agiou_mask = best_agiou_rate < 0.4
        rgindex = tf.argmax(other_agiou_rate, axis=1)
        pos_rgindex = tf.boolean_mask(rgindex, pos_agiou_mask)
        pos_ppox_yx = tf.boolean_mask(other_pbbox_yx, pos_agiou_mask)
        pos_ppox_hw = tf.boolean_mask(other_pbbox_hw, pos_agiou_mask)
        pos_pconf = tf.boolean_mask(other_pconf, pos_agiou_mask)
        pos_abbox_yx = tf.boolean_mask(other_abbox_yx, pos_agiou_mask)
        pos_abbox_hw = tf.boolean_mask(other_abbox_hw, pos_agiou_mask)
        pos_label = tf.gather(label, pos_rgindex)
        pos_gbbox_yx = tf.gather(gbbox_yx, pos_rgindex)
        pos_gbbox_hw = tf.gather(gbbox_hw, pos_rgindex)

        neg_pconf = tf.boolean_mask(other_pconf, neg_agiou_mask)
        neg_shape = tf.shape(neg_pconf)
        num_neg = neg_shape[0]
        neg_class_id = tf.constant([self.num_classes-1])
        neg_label = tf.tile(neg_class_id, [num_neg])

        pos_pbbox_yx = tf.concat([best_pbbox_yx, pos_ppox_yx], axis=0)
        pos_pbbox_hw = tf.concat([best_pbbox_hw, pos_ppox_hw], axis=0)
        pos_pconf = tf.concat([best_pconf, pos_pconf], axis=0)
        pos_label = tf.concat([label, pos_label], axis=0)
        pos_gbbox_yx = tf.concat([gbbox_yx, pos_gbbox_yx], axis=0)
        pos_gbbox_hw = tf.concat([gbbox_hw, pos_gbbox_hw], axis=0)
        pos_abbox_yx = tf.concat([best_abbox_yx, pos_abbox_yx], axis=0)
        pos_abbox_hw = tf.concat([best_abbox_hw, pos_abbox_hw], axis=0)
        conf_loss = self._focal_loss(pos_label, pos_pconf, neg_label, neg_pconf)

        pos_truth_pbbox_yx = (pos_gbbox_yx - pos_abbox_yx) / pos_abbox_hw
        pos_truth_pbbox_hw = tf.log(pos_gbbox_hw / pos_abbox_hw)
        pos_yx_loss = tf.reduce_sum(self._smooth_l1_loss(pos_pbbox_yx - pos_truth_pbbox_yx), axis=-1)
        pos_hw_loss = tf.reduce_sum(self._smooth_l1_loss(pos_pbbox_hw - pos_truth_pbbox_hw), axis=-1)
        pos_coord_loss = tf.reduce_mean(pos_yx_loss + pos_hw_loss)

        total_loss = conf_loss + pos_coord_loss
        return total_loss

    def _smooth_l1_loss(self, x):
        return tf.where(tf.abs(x) < 1., 0.5*x*x, tf.abs(x)-0.5)

    def _focal_loss(self, poslabel, posprob, neglabel, negprob):
        posprob = tf.nn.softmax(posprob)
        negprob = tf.nn.softmax(negprob)
        pos_index = tf.concat([
            tf.expand_dims(tf.range(0, tf.shape(posprob)[0], dtype=tf.int32), axis=-1),
            tf.reshape(poslabel, [-1, 1])
        ], axis=-1)
        neg_index = tf.concat([
            tf.expand_dims(tf.range(0, tf.shape(negprob)[0], dtype=tf.int32), axis=-1),
            tf.reshape(neglabel, [-1, 1])
        ], axis=-1)
        posprob = tf.clip_by_value(tf.gather_nd(posprob, pos_index), 1e-8, 1.)
        negprob = tf.clip_by_value(tf.gather_nd(negprob, neg_index), 1e-8, 1.)
        posloss = - self.alpha * tf.pow(1. - posprob, self.gamma) * tf.log(posprob)
        negloss = - self.alpha * tf.pow(1. - negprob, self.gamma) * tf.log(negprob)
        total_loss = tf.concat([posloss, negloss], axis=0)
        loss = tf.reduce_sum(total_loss) / tf.cast(tf.shape(posloss)[0], tf.float32)
        return loss

    def _train_pretraining_epoch(self, lr):
        self.is_training = True
        self.sess.run(self.train_initializer)
        mean_loss = []
        mean_acc = []
        for i in range(self.num_train // self.batch_size):
            _, loss, acc = self.sess.run([self.train_op, self.loss, self.accuracy], feed_dict={self.lr: lr})
            mean_loss.append(loss)
            mean_acc.append(acc)
        mean_loss = np.mean(mean_loss)
        mean_acc = np.mean(mean_acc)
        return mean_loss, mean_acc

    def _train_detection_epoch(self, lr):
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

    def _test_one_pretraining_image(self, images):
        self.is_training = False
        pred = self.sess.run(self.pred, feed_dict={self.images: images})
        return pred

    def _test_one_detection_image(self, images):
        self.is_training = False
        pred = self.sess.run(self.detection_pred, feed_dict={self.images: images})
        return pred

    def _save_pretraining_weight(self, mode, path):
        assert(mode in ['latest', 'best'])
        if mode == 'latest':
            saver = self.saver
        else:
            saver = self.best_saver
        if not tf.gfile.Exists(os.path.dirname(path)):
            tf.gfile.MakeDirs(os.path.dirname(path))
            print(os.path.dirname(path), 'does not exist, create it done')
        saver.save(self.sess, path, global_step=self.global_step)
        print('>> save', mode, 'model in', path, 'successfully')

    def _save_detection_weight(self, mode, path):
        assert(mode in ['latest', 'best'])
        if mode == 'latest':
            saver = self.saver
        else:
            saver = self.best_saver
        if not tf.gfile.Exists(os.path.dirname(path)):
            tf.gfile.MakeDirs(os.path.dirname(path))
            print(os.path.dirname(path), 'does not exist, create it done')
        saver.save(self.sess, path, global_step=self.global_step)
        print('>> save', mode, 'model in', path, 'successfully')

    def load_weight(self, path):
        self.saver.restore(self.sess, path)
        print('>> load weight', path, 'successfully')

    def load_pretraining_weight(self, path):
        self.pretraining_weight_saver.restore(self.sess, path)
        print('>> load pretraining weight', path, 'successfully')

    def _init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        if self.mode == 'train':
            if self.train_initializer is not None:
                self.sess.run(self.train_initializer)

    def _create_pretraining_saver(self):
        weights = tf.trainable_variables(scope='feature_extractor')
        self.saver = tf.train.Saver(weights)
        self.best_saver = tf.train.Saver(weights)

    def _create_detection_saver(self):
        weights = tf.trainable_variables(scope='feature_extractor')
        self.pretraining_weight_saver = tf.train.Saver(weights)
        weights = weights + tf.trainable_variables('regressor')
        self.saver = tf.train.Saver(weights)
        self.best_saver = tf.train.Saver(weights)

    def _create_pretraining_summary(self):
        with tf.variable_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge_all()

    def _create_detection_summary(self):
        with tf.variable_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def _bn(self, bottom):
        bn = tf.layers.batch_normalization(
            inputs=bottom,
            axis=3 if self.data_format == 'channels_last' else 1,
            training=self.is_training
        )
        return bn

    def _conv_bn_activation(self, bottom, filters, kernel_size, strides, activation=tf.nn.relu):
        conv = tf.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
        )
        bn = self._bn(conv)
        if activation is not None:
            return activation(bn)
        else:
            return bn

    def _bn_activation_conv(self, bottom, filters, kernel_size, strides, activation=tf.nn.relu, pi_init=False):
        bn = self._bn(bottom)
        if activation is not None:
            bn = activation(bn)
        if not pi_init:
            conv = tf.layers.conv2d(
                inputs=bn,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                data_format=self.data_format,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
            )
        else:
            conv = tf.layers.conv2d(
                inputs=bn,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                data_format=self.data_format,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                bias_initializer=tf.constant_initializer(-math.log((1 - self.pi) / self.pi))
            )
        return conv

    def _residual_block(self, bottom, filters, strides, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope('conv_branch'):
                conv = self._bn_activation_conv(bottom, filters, 3, strides)
                conv = self._bn_activation_conv(conv, filters, 3, 1)
            with tf.variable_scope('identity_branch'):
                if strides != 1:
                    shutcut = self._bn_activation_conv(bottom, filters, 3, strides)
                else:
                    shutcut = bottom

        return conv + shutcut

    def _residual_bottleneck(self, bottom, filters, strides, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope('conv_branch'):
                conv = self._bn_activation_conv(bottom, filters, 1, 1)
                conv = self._bn_activation_conv(conv, filters, 3, strides)
                conv = self._bn_activation_conv(conv, filters*4, 1, 1)
            with tf.variable_scope('identity_branch'):
                shutcut = self._bn_activation_conv(bottom, filters*4, 3, strides)

        return conv + shutcut

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
