from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import sys
import os
import numpy as np


class LHRCNN:
    def __init__(self, config, data_provider):
        assert config['mode'] in ['train', 'test']
        assert config['data_format'] in ['channels_first', 'channels_last']
        self.config = config
        self.data_provider = data_provider
        self.data_shape = config['data_shape']
        self.num_classes = config['num_classes'] + 1
        self.weight_decay = config['weight_decay']
        self.prob = 1. - config['keep_prob']
        self.data_format = config['data_format']
        self.mode = config['mode']
        self.batch_size = config['batch_size'] if config['mode'] == 'train' else 1
        self.nms_score_threshold = config['nms_score_threshold']
        self.nms_max_boxes = config['nms_max_boxes']
        self.nms_iou_threshold = config['nms_iou_threshold']

        self.rpn_first_step = config['rpn_first_step']
        self.rcnn_first_step = config['rcnn_first_step']
        self.rpn_second_step = config['rpn_second_step']
        self.post_nms_proposal = config['post_nms_proposal']

        self.anchor_scales = [32, 64, 128, 256, 512]
        self.anchor_ratios = [0.5, 1.0, 2.0]
        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)

        if self.data_format == 'channels_first':
            self.h, self.w = float(self.data_shape[1]-1), float(self.data_shape[2]-1)
        else:
            self.h, self.w = float(self.data_shape[0]-1), float(self.data_shape[1]-1)

        if self.mode == 'train':
            self.num_train = data_provider['num_train']
            self.num_val = data_provider['num_val']
            self.train_generator = data_provider['train_generator']
            self.train_initializer, self.train_iterator = self.train_generator
            if data_provider['val_generator'] is not None:
                self.val_generator = data_provider['val_generator']
                self.val_initializer, self.val_iterator = self.val_generator

        self.global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)

        self._define_inputs()
        self._build_graph()
        self._create_saver()
        if self.mode == 'train':
            self._create_summary()
        self._init_session()

    def _define_inputs(self):
        shape = [self.batch_size]
        shape.extend(self.data_shape)
        if self.mode == 'train':
            self.images, self.ground_truth = self.train_iterator.get_next()
            self.images.set_shape(shape)
            self.images = self.images / 127.5 -1.
        else:
            self.images = tf.placeholder(tf.float32, shape, name='images')
            self.images = self.images / 127.5 -1.
            self.ground_truth = tf.placeholder(tf.float32, [self.batch_size, None, 5], name='labels')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

    def _build_graph(self):
        with tf.variable_scope('feature_extractor'):
            c4_feat, stride = self._feature_extractor(self.images)
        with tf.variable_scope('rpn'):
            rpn_conv = self._conv_layer(c4_feat, 256, 3, 1, 'rpn_conv', activation=tf.nn.relu)
            rpn_conf = self._conv_layer(rpn_conv, self.num_anchors*2, 3, 1, 'rpn_conf')
            rpn_pbbox = self._conv_layer(rpn_conv, self.num_anchors*4, 3, 1, 'rpn_pbbox')
            if self.data_format == 'channels_first':
                rpn_conf = tf.transpose(rpn_conf, [0, 2, 3, 1])
                rpn_pbbox = tf.transpose(rpn_pbbox, [0, 2, 3, 1])
            pshape = tf.shape(rpn_conf)
            rpn_pbbox_yx, rpn_pbbox_hw, rpn_pconf = self._get_rpn_pbbox(rpn_conf, rpn_pbbox)
            abbox_y1x1, abbox_y2x2, abbox_yx, abbox_hw = self._get_abbox(pshape, stride)

            min_mask = tf.cast(abbox_y1x1[:, 0] >= 0., tf.float32) * tf.cast(abbox_y1x1[:, 1] >= 0., tf.float32)
            max_mask = tf.cast(abbox_y2x2[:, 0] <= self.h-1, tf.float32) * tf.cast(abbox_y2x2[:, 1] <= self.w-1, tf.float32)
            mask = (min_mask * max_mask) > 0.
            abbox_y1x1 = tf.boolean_mask(abbox_y1x1, mask)
            abbox_y2x2 = tf.boolean_mask(abbox_y2x2, mask)
            abbox_yx = tf.boolean_mask(abbox_yx, mask)
            abbox_hw = tf.boolean_mask(abbox_hw, mask)
            rpn_pbbox_yx = tf.boolean_mask(rpn_pbbox_yx, mask, axis=1)
            rpn_pbbox_hw = tf.boolean_mask(rpn_pbbox_hw, mask, axis=1)
            rpn_pconf = tf.boolean_mask(rpn_pconf, mask, axis=1)
        with tf.variable_scope('rcnn'):
            state5_conv1_1 = self._separable_conv_layer(c4_feat, 256, [1, 15], 1, 'state5_conv1_1', activation=tf.nn.relu)
            state5_conv1_2 = self._separable_conv_layer(state5_conv1_1, 490, [15, 1], 1, 'state5_conv1_2', activation=tf.nn.relu)
            state5_conv2_1 = self._separable_conv_layer(c4_feat, 256, [1, 15], 1, 'state5_conv2_1', activation=tf.nn.relu)
            state5_conv2_2 = self._separable_conv_layer(state5_conv2_1, 490, [15, 1], 1, 'state5_conv2_2', activation=tf.nn.relu)
            rcnn_feat = state5_conv1_2 + state5_conv2_2
            if self.mode == 'train':
                rpn_loss = []
                pos_proposal = []
                pos_rcnn_label = []
                rcnn_truth_pbbox = []
                neg_proposal = []
                pos_box_ind = []
                neg_box_ind = []
                for i in range(self.batch_size):
                    rpn_loss_, pos_proposal_, pos_rcnn_label_, rcnn_truth_pbbox_, neg_proposal_ = self._compute_one_image_loss(
                        rpn_pbbox_yx[i, ...], rpn_pbbox_hw[i, ...],
                        abbox_y1x1, abbox_y2x2, abbox_yx, abbox_hw,
                        rpn_pconf[i, ...], self.ground_truth[i, ...]
                    )
                    pos_box_ind_ = tf.zeros_like(pos_rcnn_label_, dtype=tf.int32) + i
                    neg_box_ind_ = tf.zeros_like(neg_proposal_[:, 0], dtype=tf.int32) + i
                    rpn_loss.append(rpn_loss_)
                    pos_proposal.append(pos_proposal_)
                    pos_rcnn_label.append(pos_rcnn_label_)
                    rcnn_truth_pbbox.append(rcnn_truth_pbbox_)
                    neg_proposal.append(neg_proposal_)
                    pos_box_ind.append(pos_box_ind_)
                    neg_box_ind.append(neg_box_ind_)

                rpn_loss = tf.reduce_mean(rpn_loss)
                pos_proposal = tf.concat(pos_proposal, axis=0)
                pos_rcnn_label = tf.concat(pos_rcnn_label, axis=0)
                rcnn_truth_pbbox = tf.concat(rcnn_truth_pbbox, axis=0)
                neg_proposal = tf.concat(neg_proposal, axis=0)
                pos_box_ind = tf.concat(pos_box_ind, axis=0)
                neg_box_ind = tf.concat(neg_box_ind, axis=0)
            else:
                proposal_yx = rpn_pbbox_yx[0, ...] * abbox_hw + abbox_yx
                proposal_hw = tf.exp(rpn_pbbox_hw[0, ...]) * abbox_hw
                proposal = tf.concat([proposal_yx-proposal_hw/2., proposal_yx+proposal_hw/2.], axis=-1)
                proposal_conf = tf.nn.softmax(rpn_pconf[0, ...])

            if self.mode == 'train':
                pos_proposal = tf.maximum(pos_proposal, [0., 0., 0., 0.])
                pos_proposal = tf.minimum(pos_proposal, [self.h, self.w, self.h, self.w])
                neg_proposal = tf.maximum(neg_proposal, [0., 0., 0., 0.])
                neg_proposal = tf.minimum(neg_proposal, [self.h, self.w, self.h, self.w])
                norm_factor = [self.h, self.w, self.h, self.w]
                pos_roi_feat = tf.image.crop_and_resize(rcnn_feat, pos_proposal/norm_factor, pos_box_ind, [7, 7])
                pos_roi_feat = tf.layers.flatten(pos_roi_feat)
                neg_roi_feat = tf.image.crop_and_resize(rcnn_feat, neg_proposal/norm_factor, neg_box_ind, [7, 7])
                neg_rcnn_label = tf.constant([self.num_classes-1])
                neg_rcnn_label = tf.tile(neg_rcnn_label, [tf.shape(neg_roi_feat)[0]])
                neg_roi_feat = tf.layers.flatten(neg_roi_feat)
                roi_feat = tf.concat([pos_roi_feat, neg_roi_feat], axis=0)
                rcnn_label = tf.concat([pos_rcnn_label, neg_rcnn_label], axis=0)
                num_pos = tf.shape(pos_rcnn_label)[0]
            else:
                proposal = tf.maximum(proposal, [0., 0., 0., 0.])
                proposal = tf.minimum(proposal, [self.h, self.w, self.h, self.w])
                selected_indices = tf.image.non_max_suppression(
                    proposal, proposal_conf[:, 0], self.post_nms_proposal, iou_threshold=0.7
                )
                proposal = tf.gather(proposal, selected_indices)
                proposal_yx = proposal[..., 0:2] / 2. + proposal[..., 2:4] / 2.
                proposal_hw = proposal[..., 2:4] - proposal[..., 0:2]
                box_ind = tf.zeros_like(selected_indices, dtype=tf.int32)
                norm_factor = [self.h, self.w, self.h, self.w]
                roi_feat = tf.image.crop_and_resize(rcnn_feat, proposal/norm_factor, box_ind, [7, 7])
                roi_feat = tf.layers.flatten(roi_feat)

            roi_feat = tf.layers.dense(roi_feat, 2048, name='roi_feat_dense', activation=tf.nn.relu)
            rcnn_pconf = tf.layers.dense(roi_feat, self.num_classes, name='rcnn_pconf')
            rcnn_pbbox = tf.layers.dense(roi_feat, 4, name='rcnn_pbbox')

            if self.mode == 'train':
                rcnn_conf_loss = tf.losses.sparse_softmax_cross_entropy(rcnn_label, rcnn_pconf, reduction=tf.losses.Reduction.MEAN)
                pos_rcnn_pbbox_loss = self._smooth_l1_loss(tf.gather(rcnn_pbbox, tf.range(num_pos, dtype=tf.int32)) - rcnn_truth_pbbox)
                pos_rcnn_pbbox_loss = tf.reduce_mean(tf.reduce_sum(pos_rcnn_pbbox_loss, axis=-1))
                rcnn_loss = rcnn_conf_loss + pos_rcnn_pbbox_loss
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=.9)
                rpn_loss = rpn_loss + self.weight_decay * tf.add_n(
                    [tf.nn.l2_loss(var) for var in tf.trainable_variables('feature_extractor')]
                ) + self.weight_decay * tf.add_n(
                    [tf.nn.l2_loss(var) for var in tf.trainable_variables('rpn')]
                )
                rcnn_loss = rcnn_loss + + self.weight_decay * tf.add_n(
                    [tf.nn.l2_loss(var) for var in tf.trainable_variables('rcnn')]
                )
                rpn_vars = tf.trainable_variables('feature_extractor') + tf.trainable_variables('rpn')
                rpn_grads_and_vars = optimizer.compute_gradients(rpn_loss, rpn_vars)
                train_rpn_op = optimizer.apply_gradients(rpn_grads_and_vars)
                rcnn_vars = tf.trainable_variables('rcnn')
                rcnn_grads_and_vars = optimizer.compute_gradients(rcnn_loss, rcnn_vars)
                train_rcnn_op = optimizer.apply_gradients(rcnn_grads_and_vars, global_step=self.global_step)

                train_op = tf.case([(tf.less(self.global_step, self.rpn_first_step), lambda: train_rpn_op),
                                    (tf.less(self.global_step, self.rcnn_first_step), lambda: train_rcnn_op),
                                    (tf.less(self.global_step, self.rpn_second_step), lambda: train_rpn_op)],
                                         default=lambda: train_rcnn_op, exclusive=False)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self.train_op = tf.group([update_ops, train_op])
                self.loss = tf.case([(tf.less(self.global_step, self.rpn_first_step), lambda: rpn_loss),
                                     (tf.less(self.global_step, self.rcnn_first_step), lambda: rcnn_loss),
                                     (tf.less(self.global_step, self.rpn_second_step), lambda: rpn_loss)],
                                     default=lambda: rcnn_loss, exclusive=False)

            else:
                rcnn_pbbox_yxt = rcnn_pbbox[..., 0:2]
                rcnn_pbbox_hwt = rcnn_pbbox[..., 2:4]
                proposal_yxt = proposal_yx
                proposal_hwt = proposal_hw
                confidence = tf.nn.softmax(rcnn_pconf)
                class_id = tf.argmax(confidence, axis=-1)
                conf_mask = tf.less(class_id, self.num_classes-1)
                rcnn_pbbox_yxt = tf.boolean_mask(rcnn_pbbox_yxt, conf_mask)
                rcnn_pbbox_hwt = tf.boolean_mask(rcnn_pbbox_hwt, conf_mask)
                confidence = tf.boolean_mask(confidence, conf_mask)
                proposal_yxt = tf.boolean_mask(proposal_yxt, conf_mask)
                proposal_hwt = tf.boolean_mask(proposal_hwt, conf_mask)
                dpbbox_yxt = rcnn_pbbox_yxt * proposal_hwt + proposal_yxt
                dpbbox_hwt = proposal_hwt * tf.exp(rcnn_pbbox_hwt)
                dpbbox_y1x1 = dpbbox_yxt - dpbbox_hwt / 2.
                dpbbox_y2x2 = dpbbox_yxt + dpbbox_hwt / 2.
                dpbbox_y1x1y2x2 = tf.concat([dpbbox_y1x1, dpbbox_y2x2], axis=-1)
                filter_mask = tf.greater_equal(confidence, self.nms_score_threshold)
                scores = []
                class_id = []
                bbox = []
                for i in range(self.num_classes-1):
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
        with tf.variable_scope('stage1'):
            conv1 = self._conv_layer(images, 24, 3, 2, 'conv1', activation=tf.nn.relu)
            pool1 = self._max_pooling(conv1, 3, 2, 'pool1')
        with tf.variable_scope('stage2'):
            stage2_sconv1 = self._conv_layer(pool1, 144, 3, 2, 'stage2_sconv1', activation=tf.nn.relu)
            stage2_sconv2 = self._separable_conv_layer(stage2_sconv1, 144, 3, 1, 'stage2_sconv2', activation=tf.nn.relu)
            stage2_sconv3 = self._separable_conv_layer(stage2_sconv2, 144, 3, 1, 'stage2_sconv3', activation=tf.nn.relu)
            stage2_sconv4 = self._separable_conv_layer(stage2_sconv3, 144, 3, 1, 'stage2_sconv4', activation=tf.nn.relu)
        with tf.variable_scope('stage3'):
            stage3_sconv1 = self._conv_layer(stage2_sconv4, 288, 3, 2, 'stage3_sconv1', activation=tf.nn.relu)
            stage3_sconv2 = self._separable_conv_layer(stage3_sconv1, 288, 3, 1, 'stage3_sconv2', activation=tf.nn.relu)
            stage3_sconv3 = self._separable_conv_layer(stage3_sconv2, 288, 3, 1, 'stage3_sconv3', activation=tf.nn.relu)
            stage3_sconv4 = self._separable_conv_layer(stage3_sconv3, 288, 3, 1, 'stage3_sconv4', activation=tf.nn.relu)
            stage3_sconv5 = self._separable_conv_layer(stage3_sconv4, 288, 3, 1, 'stage3_sconv5', activation=tf.nn.relu)
            stage3_sconv6 = self._separable_conv_layer(stage3_sconv5, 288, 3, 1, 'stage3_sconv6', activation=tf.nn.relu)
            stage3_sconv7 = self._separable_conv_layer(stage3_sconv6, 288, 3, 1, 'stage3_sconv7', activation=tf.nn.relu)
            stage3_sconv8 = self._separable_conv_layer(stage3_sconv7, 288, 3, 1, 'stage3_sconv8', activation=tf.nn.relu)
        with tf.variable_scope('stage4'):
            stage4_sconv1 = self._conv_layer(stage3_sconv8, 576, 3, 2, 'stage4_sconv1', activation=tf.nn.relu)
            stage4_sconv2 = self._separable_conv_layer(stage4_sconv1, 576, 3, 1, 'stage4_sconv2', activation=tf.nn.relu)
            stage4_sconv3 = self._separable_conv_layer(stage4_sconv2, 576, 3, 1, 'stage4_sconv3', activation=tf.nn.relu)
            stage4_sconv4 = self._separable_conv_layer(stage4_sconv3, 576, 3, 1, 'stage4_sconv4', activation=tf.nn.relu)

        downsampling_rate = 32.
        return stage4_sconv4, downsampling_rate

    def _get_rpn_pbbox(self, rpn_conf, rpn_bbox):
        rpn_conf = tf.reshape(rpn_conf, [self.batch_size, -1, 2])
        rpn_bbox = tf.reshape(rpn_bbox, [self.batch_size, -1, 4])
        rpn_pbbox_yx = rpn_bbox[..., :2]
        rpn_pbbox_hw = rpn_bbox[..., 2:]
        return rpn_pbbox_yx, rpn_pbbox_hw, rpn_conf

    def _get_abbox(self, pshape, stride):
        topleft_y = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)
        topleft_x = tf.range(0., tf.cast(pshape[2], tf.float32), dtype=tf.float32)
        topleft_y = tf.reshape(topleft_y, [-1, 1, 1, 1]) + 0.5
        topleft_x = tf.reshape(topleft_x, [1, -1, 1, 1]) + 0.5
        topleft_y = tf.tile(topleft_y, [1, pshape[2], 1, 1])
        topleft_x = tf.tile(topleft_x, [pshape[1], 1, 1, 1])
        topleft_yx = tf.concat([topleft_y, topleft_x], -1)
        topleft_yx = tf.tile(topleft_yx, [1, 1, self.num_anchors, 1]) * stride

        priors = []
        for size in self.anchor_scales:
            for ratio in self.anchor_ratios:
                priors.append([size*(ratio**0.5), size/(ratio**0.5)])
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
        rcnn_label = tf.cast(ground_truth[..., 4], dtype=tf.int32)

        abbox_hwti = tf.reshape(abbox_hw, [1, -1, 2])
        abbox_y1x1ti = tf.reshape(abbox_y1x1, [1, -1, 2])
        abbox_y2x2ti = tf.reshape(abbox_y2x2, [1, -1, 2])
        gbbox_hwti = tf.reshape(gbbox_hw, [-1, 1, 2])
        gbbox_y1x1ti = tf.reshape(gbbox_y1x1, [-1, 1, 2])
        gbbox_y2x2ti = tf.reshape(gbbox_y2x2, [-1, 1, 2])
        num_a = tf.shape(abbox_hwti)[1]
        num_g = tf.shape(gbbox_hwti)[0]
        abbox_hwti = tf.tile(abbox_hwti, [num_g, 1, 1])
        abbox_y1x1ti = tf.tile(abbox_y1x1ti, [num_g, 1, 1])
        abbox_y2x2ti = tf.tile(abbox_y2x2ti, [num_g, 1, 1])
        gbbox_hwti = tf.tile(gbbox_hwti, [1, num_a, 1])
        gbbox_y1x1ti = tf.tile(gbbox_y1x1ti, [1, num_a, 1])
        gbbox_y2x2ti = tf.tile(gbbox_y2x2ti, [1, num_a, 1])

        gaiou_y1x1ti = tf.maximum(abbox_y1x1ti, gbbox_y1x1ti)
        gaiou_y2x2ti = tf.minimum(abbox_y2x2ti, gbbox_y2x2ti)
        gaiou_area = tf.reduce_prod(tf.maximum(gaiou_y2x2ti - gaiou_y1x1ti, 0), axis=-1)
        aarea = tf.reduce_prod(abbox_hwti, axis=-1)
        garea = tf.reduce_prod(gbbox_hwti, axis=-1)
        gaiou_rate = gaiou_area / (aarea + garea - gaiou_area + 1e-8)

        best_raindex = tf.argmax(gaiou_rate, axis=1)
        best_pbbox_yx = tf.gather(pbbox_yx, best_raindex)
        best_pbbox_hw = tf.gather(pbbox_hw, best_raindex)
        best_pconf = tf.gather(pconf, best_raindex)
        best_abbox_yx = tf.gather(abbox_yx, best_raindex)
        best_abbox_hw = tf.gather(abbox_hw, best_raindex)
        best_rcnn_label = tf.gather(rcnn_label, best_raindex)

        bestmask, _ = tf.unique(best_raindex)
        bestmask = tf.contrib.framework.sort(bestmask)
        bestmask = tf.reshape(bestmask, [-1, 1])
        bestmask = tf.sparse.SparseTensor(tf.concat([bestmask, tf.zeros_like(bestmask)], axis=-1),
                                          tf.squeeze(tf.ones_like(bestmask)), dense_shape=[num_a, 1])
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
        max_agiou_rate = tf.reduce_max(other_agiou_rate, axis=1)
        pos_agiou_mask = max_agiou_rate > 0.5
        neg_agiou_mask = max_agiou_rate < 0.3
        rgindex = tf.argmax(other_agiou_rate, axis=1)
        pos_rgindex = tf.boolean_mask(rgindex, pos_agiou_mask)
        pos_pbbox_yx = tf.boolean_mask(other_pbbox_yx, pos_agiou_mask)
        pos_pbbox_hw = tf.boolean_mask(other_pbbox_hw, pos_agiou_mask)
        pos_pconf = tf.boolean_mask(other_pconf, pos_agiou_mask)
        pos_abbox_yx = tf.boolean_mask(other_abbox_yx, pos_agiou_mask)
        pos_abbox_hw = tf.boolean_mask(other_abbox_hw, pos_agiou_mask)
        pos_gbbox_yx = tf.gather(gbbox_yx, pos_rgindex)
        pos_gbbox_hw = tf.gather(gbbox_hw, pos_rgindex)
        pos_rcnn_label = tf.gather(rcnn_label, pos_rgindex)

        pos_pbbox_yx = tf.concat([best_pbbox_yx, pos_pbbox_yx], axis=0)
        pos_pbbox_hw = tf.concat([best_pbbox_hw, pos_pbbox_hw], axis=0)
        pos_pconf = tf.concat([best_pconf, pos_pconf], axis=0)
        pos_gbbox_yx = tf.concat([gbbox_yx, pos_gbbox_yx], axis=0)
        pos_gbbox_hw = tf.concat([gbbox_hw, pos_gbbox_hw], axis=0)
        pos_rcnn_label = tf.concat([best_rcnn_label, pos_rcnn_label], axis=0)
        pos_abbox_yx = tf.concat([best_abbox_yx, pos_abbox_yx], axis=0)
        pos_abbox_hw = tf.concat([best_abbox_hw, pos_abbox_hw], axis=0)
        pos_abbox_y1x1y2x2 = tf.concat([pos_abbox_yx-pos_abbox_hw/2., pos_abbox_yx+pos_abbox_hw/2.], axis=-1)

        neg_pconf = tf.boolean_mask(other_pconf, neg_agiou_mask)
        neg_abbox_yx = tf.boolean_mask(other_abbox_yx, neg_agiou_mask)
        neg_abbox_hw = tf.boolean_mask(other_abbox_hw, neg_agiou_mask)
        neg_pbbox_yx = tf.boolean_mask(other_pbbox_yx, neg_agiou_mask)
        neg_pbbox_hw = tf.boolean_mask(other_pbbox_hw, neg_agiou_mask)
        neg_abbox_y1x1y2x2 = tf.concat([neg_abbox_yx-neg_abbox_hw/2., neg_abbox_yx+neg_abbox_hw/2.], axis=-1)

        num_pos = tf.shape(pos_pconf)[0]
        num_neg = tf.shape(neg_pconf)[0]
        pos_label = tf.constant([0])
        pos_label = tf.tile(pos_label, [num_pos])
        neg_label = tf.constant([1])
        neg_label = tf.tile(neg_label, [num_neg])
        chosen_num_pos = tf.cond(num_pos > 128, lambda: 128, lambda: num_pos)
        chosen_num_neg = tf.cond(num_neg > 256-chosen_num_pos, lambda: 256-chosen_num_pos, lambda: num_neg)
        pos_conf_loss = tf.losses.sparse_softmax_cross_entropy(pos_label, pos_pconf, reduction=tf.losses.Reduction.NONE)
        selected_posindices = tf.image.non_max_suppression(
            pos_abbox_y1x1y2x2, tf.nn.softmax(pos_pconf)[:, 0], chosen_num_pos, iou_threshold=0.7
        )
        pos_conf_loss = tf.reduce_mean(tf.gather(pos_conf_loss, selected_posindices))

        neg_loss = tf.losses.sparse_softmax_cross_entropy(neg_label, neg_pconf, reduction=tf.losses.Reduction.NONE)
        selected_negindices = tf.image.non_max_suppression(
            neg_abbox_y1x1y2x2, neg_loss, chosen_num_neg, iou_threshold=0.7
        )
        neg_loss = tf.reduce_mean(tf.gather(neg_loss, selected_negindices))

        pos_abbox_yx = tf.gather(pos_abbox_yx, selected_posindices)
        pos_abbox_hw = tf.gather(pos_abbox_hw, selected_posindices)
        pos_pbbox_yx = tf.gather(pos_pbbox_yx, selected_posindices)
        pos_pbbox_hw = tf.gather(pos_pbbox_hw, selected_posindices)
        pos_gbbox_yx = tf.gather(pos_gbbox_yx, selected_posindices)
        pos_gbbox_hw = tf.gather(pos_gbbox_hw, selected_posindices)
        pos_rcnn_label = tf.gather(pos_rcnn_label, selected_posindices)

        neg_abbox_yx = tf.gather(neg_abbox_yx, selected_negindices)
        neg_abbox_hw = tf.gather(neg_abbox_hw, selected_negindices)
        neg_pbbox_yx = tf.gather(neg_pbbox_yx, selected_negindices)
        neg_pbbox_hw = tf.gather(neg_pbbox_hw, selected_negindices)

        pos_truth_pbbox_yx = (pos_gbbox_yx - pos_abbox_yx) / pos_abbox_hw
        pos_truth_pbbox_hw = tf.log(pos_gbbox_hw / pos_abbox_hw)
        pos_yx_loss = tf.reduce_sum(self._smooth_l1_loss(pos_pbbox_yx - pos_truth_pbbox_yx), axis=-1)
        pos_hw_loss = tf.reduce_sum(self._smooth_l1_loss(pos_pbbox_hw - pos_truth_pbbox_hw), axis=-1)
        pos_coord_loss = tf.reduce_mean(pos_yx_loss + pos_hw_loss)

        total_loss = neg_loss + pos_conf_loss + 10.*pos_coord_loss

        pos_proposal_yx = pos_abbox_hw * pos_pbbox_yx + pos_abbox_yx
        pos_proposal_hw = tf.exp(pos_pbbox_hw) * pos_abbox_hw
        rcnn_truth_pbbox_yx = (pos_gbbox_yx - pos_proposal_yx) / pos_proposal_yx
        rcnn_truth_pbbox_hw = tf.log(pos_gbbox_hw / pos_proposal_hw)
        rcnn_truth_pbbox = tf.concat([rcnn_truth_pbbox_yx, rcnn_truth_pbbox_hw], axis=-1)
        neg_proposal_yx = neg_abbox_hw * neg_pbbox_yx + neg_abbox_yx
        neg_proposal_hw = tf.exp(neg_pbbox_hw) * neg_abbox_hw
        pos_proposal_y1x1 = pos_proposal_yx - pos_proposal_hw / 2.
        pos_proposal_y2x2 = pos_proposal_yx + pos_proposal_hw / 2.
        pos_proposal = tf.concat([pos_proposal_y1x1, pos_proposal_y2x2], axis=-1)
        neg_proposal_y1x1 = neg_proposal_yx - neg_proposal_hw / 2.
        neg_proposal_y2x2 = neg_proposal_yx + neg_proposal_hw / 2.
        neg_proposal = tf.concat([neg_proposal_y1x1, neg_proposal_y2x2], axis=-1)

        return total_loss, pos_proposal, pos_rcnn_label, rcnn_truth_pbbox, neg_proposal

    def _smooth_l1_loss(self, x):
        return tf.where(tf.abs(x) < 1., 0.5*x*x, tf.abs(x)-0.5)

    def _init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        if self.mode == 'train':
            self.sess.run(self.train_initializer)

    def _create_saver(self):
        weights = tf.trainable_variables(scope='feature_extractor')
        self.pretraining_weight_saver = tf.train.Saver(weights)
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()

    def _create_summary(self):
        with tf.variable_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def train_one_epoch(self, lr):
        self.sess.run(self.train_initializer)
        mean_loss = []
        num_iters = self.num_train // self.batch_size
        for i in range(num_iters):
            _, loss, global_step = self.sess.run([self.train_op, self.loss, self.global_step], feed_dict={self.lr: lr, self.is_training:True})
            # sys.stdout.write('\r>> ' + 'iters '+str(i+1)+str('/')+str(num_iters)+' loss '+str(loss))
            if global_step < self.rpn_first_step:
                loss_name = 'rpn_loss'
            elif global_step < self.rcnn_first_step:
                loss_name = 'rcnn_loss'
            elif global_step < self.rpn_second_step:
                loss_name = 'rpn_loss'
            else:
                loss_name = 'rcnn_loss'
            print('iters ',str(i+1)+str('/')+str(num_iters), loss_name, loss, 'global_step', global_step)
            # sys.stdout.flush()
            mean_loss.append(loss)
        # sys.stdout.write('\n')
        mean_loss = np.mean(mean_loss)
        return mean_loss

    def test_one_image(self, images):
        pred = self.sess.run(self.detection_pred, feed_dict={self.images: images, self.is_training:False})
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

    def load_rpn_weight(self, path):
        self.rpn_saver.restore(self.sess, path)
        print('load rpn weight', path, 'successfully')

    def load_pretraining_weight(self, path):
        self.pretraining_weight_saver.restore(self.sess, path)
        print('>> load pretraining weight', path, 'successfully')

    def _bn(self, bottom):
        bn = tf.layers.batch_normalization(
            inputs=bottom,
            axis=3 if self.data_format == 'channels_last' else 1,
            training=self.is_training
        )
        return bn

    def _conv_layer(self, bottom, filters, kernel_size, strides, name=None, dilation_rate=1, activation=None, padding='same'):
        conv = tf.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            name=name,
            data_format=self.data_format,
            dilation_rate=dilation_rate,
        )
        bn = self._bn(conv)
        if activation is not None:
            bn = activation(bn)
        return bn

    def _separable_conv_layer(self, bottom, filters, kernel_size, strides, name=None, dilation_rate=1, activation=None):
        conv = tf.layers.separable_conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name=name,
            data_format=self.data_format,
            use_bias=False,
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
