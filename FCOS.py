from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import os
import math


class FCOS:
    def __init__(self, config, data_provider):

        assert config['mode'] in ['train', 'test']
        assert config['data_format'] in ['channels_first', 'channels_last']
        self.config = config
        self.data_provider = data_provider
        self.data_shape = config['data_shape']
        self.num_classes = config['num_classes']
        self.weight_decay = config['weight_decay']
        self.prob = 1. - config['keep_prob']
        self.data_format = config['data_format']
        self.mode = config['mode']
        self.batch_size = config['batch_size'] if config['mode'] == 'train' else 1
        self.nms_score_threshold = config['nms_score_threshold']
        self.nms_max_boxes = config['nms_max_boxes']
        self.nms_iou_threshold = config['nms_iou_threshold']

        self.is_bottleneck = True,
        self.block_list = [3, 4, 6, 3]     # must len 4
        self.filters_list = [16 * (2 ** i) for i in range(len(self.block_list))]

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
        with tf.variable_scope('backone'):
            endpoints = []
            conv1_1 = self._conv_bn_activation(
                bottom=self.images,
                filters=16,
                kernel_size=7,
                strides=2,
            )
            pool1 = self._max_pooling(
                bottom=conv1_1,
                pool_size=3,
                strides=2,
                name='pool1'
            )
            if self.is_bottleneck:
                stack_residual_unit_fn = self._residual_bottleneck
            else:
                stack_residual_unit_fn = self._residual_block
            residual_block = pool1
            for i in range(self.block_list[0]):
                residual_block = stack_residual_unit_fn(residual_block, self.filters_list[0], 1, 'block1_unit' + str(i + 1))
            endpoints.append(residual_block)
            for i in range(1, len(self.block_list)):
                residual_block = stack_residual_unit_fn(residual_block, self.filters_list[i], 2, 'block' + str(i + 1) + '_unit' + str(1))
                for j in range(1, self.block_list[i]):
                    residual_block = stack_residual_unit_fn(residual_block, self.filters_list[i], 1, 'block' + str(i + 1) + '_unit' + str(j + 1))
                endpoints.append(residual_block)

        with tf.variable_scope('pyramid'):
            c3 = self._bn_activation_conv(endpoints[-3], 256, 1, 1)
            c4 = self._bn_activation_conv(endpoints[-2], 256, 1, 1)
            c5 = self._bn_activation_conv(endpoints[-1], 256, 1, 1)
            p5 = self._get_pyramid(c5, 256)
            p4, top_down = self._get_pyramid(c4, 256, p5)
            p3, _ = self._get_pyramid(c3, 256, top_down)
            p6 = self._bn_activation_conv(p5, 256, 3, 2)
            p7 = self._bn_activation_conv(p6, 256, 3, 2)
        with tf.variable_scope('head'):
            p3conf, p3reg, p3center = self._detect_head(p3)
            p4conf, p4reg, p4center = self._detect_head(p4)
            p5conf, p5reg, p5center = self._detect_head(p5)
            p6conf, p6reg, p6center = self._detect_head(p6)
            p7conf, p7reg, p7center = self._detect_head(p7)
            if self.data_format == 'channels_first':
                p3conf = tf.transpose(p3conf, [0, 2, 3, 1])
                p3reg = tf.transpose(p3reg, [0, 2, 3, 1])
                p3center = tf.transpose(p3center, [0, 2, 3, 1])
                p4conf = tf.transpose(p4conf, [0, 2, 3, 1])
                p4reg = tf.transpose(p4reg, [0, 2, 3, 1])
                p4center = tf.transpose(p4center, [0, 2, 3, 1])
                p5conf = tf.transpose(p5conf, [0, 2, 3, 1])
                p5reg = tf.transpose(p5reg, [0, 2, 3, 1])
                p5center = tf.transpose(p5center, [0, 2, 3, 1])
                p6conf = tf.transpose(p6conf, [0, 2, 3, 1])
                p6reg = tf.transpose(p6reg, [0, 2, 3, 1])
                p6center = tf.transpose(p6center, [0, 2, 3, 1])
                p7conf = tf.transpose(p7conf, [0, 2, 3, 1])
                p7reg = tf.transpose(p7reg, [0, 2, 3, 1])
                p7center = tf.transpose(p7center, [0, 2, 3, 1])
            s3, s4, s5, s6, s7 = 8, 16, 32, 64, 128
            p3shape = [tf.shape(p3center)[1], tf.shape(p3center)[2]]
            p4shape = [tf.shape(p4center)[1], tf.shape(p4center)[2]]
            p5shape = [tf.shape(p5center)[1], tf.shape(p5center)[2]]
            p6shape = [tf.shape(p6center)[1], tf.shape(p6center)[2]]
            p7shape = [tf.shape(p7center)[1], tf.shape(p7center)[2]]
            h3 = tf.range(0., tf.cast(p3shape[0], tf.float32), dtype=tf.float32)
            w3 = tf.range(0., tf.cast(p3shape[1], tf.float32), dtype=tf.float32)
            h4 = tf.range(0., tf.cast(p4shape[0], tf.float32), dtype=tf.float32)
            w4 = tf.range(0., tf.cast(p4shape[1], tf.float32), dtype=tf.float32)
            h5 = tf.range(0., tf.cast(p5shape[0], tf.float32), dtype=tf.float32)
            w5 = tf.range(0., tf.cast(p5shape[1], tf.float32), dtype=tf.float32)
            h6 = tf.range(0., tf.cast(p6shape[0], tf.float32), dtype=tf.float32)
            w6 = tf.range(0., tf.cast(p6shape[1], tf.float32), dtype=tf.float32)
            h7 = tf.range(0., tf.cast(p7shape[0], tf.float32), dtype=tf.float32)
            w7 = tf.range(0., tf.cast(p7shape[1], tf.float32), dtype=tf.float32)
            [grid_x3, grid_y3] = tf.meshgrid(w3, h3)
            [grid_x4, grid_y4] = tf.meshgrid(w4, h4)
            [grid_x5, grid_y5] = tf.meshgrid(w5, h5)
            [grid_x6, grid_y6] = tf.meshgrid(w6, h6)
            [grid_x7, grid_y7] = tf.meshgrid(w7, h7)

            if self.mode == 'train':
                total_loss = []
                for i in range(self.batch_size):
                    gt_i = self.ground_truth[i, ...]
                    slice_index = tf.argmin(gt_i, axis=0)[0]
                    gt_i = tf.gather(gt_i, tf.range(0, slice_index, dtype=tf.int64))
                    gt_size = tf.sqrt(gt_i[..., 2] * gt_i[..., 3])
                    g3 = tf.boolean_mask(gt_i, gt_size <= 64.)
                    g4 = tf.boolean_mask(gt_i, tf.cast(gt_size >= 64., tf.float32)*tf.cast(gt_size <= 128., tf.float32) > 0.)
                    g5 = tf.boolean_mask(gt_i, tf.cast(gt_size >= 128., tf.float32)*tf.cast(gt_size <= 256., tf.float32) > 0.)
                    g6 = tf.boolean_mask(gt_i, tf.cast(gt_size >= 256., tf.float32)*tf.cast(gt_size <= 512., tf.float32) > 0.)
                    g7 = tf.boolean_mask(gt_i, gt_size >= 512.)
                    loss3 = tf.cond(
                        tf.shape(g3)[0] > 0,
                        lambda: self._compute_one_image_loss(p3conf[i, ...], p3reg[i, ...], p3center[i, ...], g3, grid_y3, grid_x3, s3, p3shape),
                        lambda: 0.
                    )
                    loss4 = tf.cond(
                        tf.shape(g4)[0] > 0,
                        lambda: self._compute_one_image_loss(p4conf[i, ...], p4reg[i, ...], p4center[i, ...], g4, grid_y4, grid_x4, s4, p4shape),
                        lambda: 0.
                    )
                    loss5 = tf.cond(
                        tf.shape(g5)[0] > 0,
                        lambda: self._compute_one_image_loss(p5conf[i, ...], p5reg[i, ...], p5center[i, ...], g5, grid_y5, grid_x5, s5, p5shape),
                        lambda: 0.
                    )
                    loss6 = tf.cond(
                        tf.shape(g6)[0] > 0,
                        lambda: self._compute_one_image_loss(p6conf[i, ...], p6reg[i, ...], p6center[i, ...], g6, grid_y6, grid_x6, s6, p6shape),
                        lambda: 0.
                    )
                    loss7 = tf.cond(
                        tf.shape(g7)[0] > 0,
                        lambda: self._compute_one_image_loss(p7conf[i, ...], p7reg[i, ...], p7center[i, ...], g7, grid_y7, grid_x7, s7, p7shape),
                        lambda: 0.
                    )
                    total_loss.append(loss3 + loss4 + loss5 + loss6 + loss7)
                self.loss = tf.reduce_mean(total_loss) + self.weight_decay * tf.add_n(
                            [tf.nn.l2_loss(var) for var in tf.trainable_variables()])
                optimizer = tf.train.MomentumOptimizer(self.lr, momentum=0.9)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                train_op = optimizer.minimize(self.loss, global_step=self.global_step)
                self.train_op = tf.group([update_ops, train_op])
            else:
                p3conf = tf.reshape(tf.sigmoid(p3conf[0, ...])*tf.sigmoid(p3center[0, ...]), [-1, self.num_classes])
                p4conf = tf.reshape(tf.sigmoid(p4conf[0, ...])*tf.sigmoid(p4center[0, ...]), [-1, self.num_classes])
                p5conf = tf.reshape(tf.sigmoid(p5conf[0, ...])*tf.sigmoid(p5center[0, ...]), [-1, self.num_classes])
                p6conf = tf.reshape(tf.sigmoid(p6conf[0, ...])*tf.sigmoid(p6center[0, ...]), [-1, self.num_classes])
                p7conf = tf.reshape(tf.sigmoid(p7conf[0, ...])*tf.sigmoid(p7center[0, ...]), [-1, self.num_classes])
                pconf = tf.concat([p3conf, p4conf, p5conf, p6conf, p7conf], axis=0)

                p3reg = p3reg[0, ...]
                p4reg = p4reg[0, ...]
                p5reg = p5reg[0, ...]
                p6reg = p6reg[0, ...]
                p7reg = p7reg[0, ...]
                grid_y3 = tf.expand_dims(grid_y3, axis=-1)
                grid_x3 = tf.expand_dims(grid_x3, axis=-1)
                grid_y4 = tf.expand_dims(grid_y4, axis=-1)
                grid_x4 = tf.expand_dims(grid_x4, axis=-1)
                grid_y5 = tf.expand_dims(grid_y5, axis=-1)
                grid_x5 = tf.expand_dims(grid_x5, axis=-1)
                grid_y6 = tf.expand_dims(grid_y6, axis=-1)
                grid_x6 = tf.expand_dims(grid_x6, axis=-1)
                grid_y7 = tf.expand_dims(grid_y7, axis=-1)
                grid_x7 = tf.expand_dims(grid_x7, axis=-1)

                p3_y1 = grid_y3 - p3reg[..., 2:3]
                p3_y2 = grid_y3 + p3reg[..., 3:4]
                p3_x1 = grid_x3 - p3reg[..., 0:1]
                p3_x2 = grid_x3 + p3reg[..., 1:2]
                p4_y1 = grid_y4 - p4reg[..., 2:3]
                p4_y2 = grid_y4 + p4reg[..., 3:4]
                p4_x1 = grid_x4 - p4reg[..., 0:1]
                p4_x2 = grid_x4 + p4reg[..., 1:2]
                p5_y1 = grid_y5 - p5reg[..., 2:3]
                p5_y2 = grid_y5 + p5reg[..., 3:4]
                p5_x1 = grid_x5 - p5reg[..., 0:1]
                p5_x2 = grid_x5 + p5reg[..., 1:2]
                p6_y1 = grid_y6 - p6reg[..., 2:3]
                p6_y2 = grid_y6 + p6reg[..., 3:4]
                p6_x1 = grid_x6 - p6reg[..., 0:1]
                p6_x2 = grid_x6 + p6reg[..., 1:2]
                p7_y1 = grid_y7 - p7reg[..., 2:3]
                p7_y2 = grid_y7 + p7reg[..., 3:4]
                p7_x1 = grid_x7 - p7reg[..., 0:1]
                p7_x2 = grid_x7 + p7reg[..., 1:2]

                p3bbox = tf.reshape(tf.concat([p3_y1, p3_x1, p3_y2, p3_x2], axis=-1), [-1, 4]) * s3
                p4bbox = tf.reshape(tf.concat([p4_y1, p4_x1, p4_y2, p4_x2], axis=-1), [-1, 4]) * s4
                p5bbox = tf.reshape(tf.concat([p5_y1, p5_x1, p5_y2, p5_x2], axis=-1), [-1, 4]) * s5
                p6bbox = tf.reshape(tf.concat([p6_y1, p6_x1, p6_y2, p6_x2], axis=-1), [-1, 4]) * s6
                p7bbox = tf.reshape(tf.concat([p7_y1, p7_x1, p7_y2, p7_x2], axis=-1), [-1, 4]) * s7
                pbbox = tf.concat([p3bbox, p4bbox, p5bbox, p6bbox, p7bbox], axis=0)

                filter_mask = tf.greater_equal(pconf, self.nms_score_threshold)
                scores = []
                class_id = []
                bbox = []
                for i in range(self.num_classes - 1):
                    scoresi = tf.boolean_mask(pconf[:, i], filter_mask[:, i])
                    bboxi = tf.boolean_mask(pbbox, filter_mask[:, i])
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

    def _compute_one_image_loss(self, heatmap_pred, dist_pred, center_pred, ground_truth, grid_y, grid_x,
                                stride, pshape):
        gbbox_y = ground_truth[..., 0] / stride
        gbbox_x = ground_truth[..., 1] / stride
        gbbox_h = ground_truth[..., 2] / stride
        gbbox_w = ground_truth[..., 3] / stride
        class_id = tf.cast(ground_truth[..., 4], dtype=tf.int32)
        gbbox_y1 = gbbox_y - gbbox_h/2.
        gbbox_y2 = gbbox_y + gbbox_h/2.
        gbbox_x1 = gbbox_x - gbbox_w/2.
        gbbox_x2 = gbbox_x + gbbox_w/2.

        gbbox_y1 = tf.reshape(gbbox_y1, [1, 1, -1])
        gbbox_x1 = tf.reshape(gbbox_x1, [1, 1, -1])
        gbbox_y2 = tf.reshape(gbbox_y2, [1, 1, -1])
        gbbox_x2 = tf.reshape(gbbox_x2, [1, 1, -1])
        num_g = tf.shape(gbbox_y1)[-1]
        grid_y = tf.expand_dims(grid_y, -1)
        grid_x = tf.expand_dims(grid_x, -1)
        grid_y = tf.tile(grid_y, [1, 1, num_g])
        grid_x = tf.tile(grid_x, [1, 1, num_g])
        dist_l = grid_x - gbbox_x1
        dist_r = gbbox_x2 - grid_x
        dist_t = grid_y - gbbox_y1
        dist_b = gbbox_y2 - grid_y
        grid_y_mask = tf.cast(dist_t > 0., tf.float32) * tf.cast(dist_b > 0., tf.float32)
        grid_x_mask = tf.cast(dist_l > 0., tf.float32) * tf.cast(dist_r > 0., tf.float32)
        heatmask = grid_y_mask * grid_x_mask
        dist_l *= heatmask
        dist_r *= heatmask
        dist_t *= heatmask
        dist_b *= heatmask
        loc = tf.reduce_max(heatmask, axis=-1)
        dist_area = (dist_l + dist_r) * (dist_t + dist_b)
        dist_area_ = dist_area + (1. - heatmask) * 1e8
        dist_area_min = tf.reduce_min(dist_area_, axis=-1, keepdims=True)
        dist_mask = tf.cast(tf.equal(dist_area, dist_area_min), tf.float32) * tf.expand_dims(loc, axis=-1)
        dist_l *= dist_mask
        dist_r *= dist_mask
        dist_t *= dist_mask
        dist_b *= dist_mask
        dist_l = tf.reduce_max(dist_l, axis=-1)
        dist_r = tf.reduce_max(dist_r, axis=-1)
        dist_t = tf.reduce_max(dist_t, axis=-1)
        dist_b = tf.reduce_max(dist_b, axis=-1)
        dist_pred_l = dist_pred[..., 0]
        dist_pred_r = dist_pred[..., 1]
        dist_pred_t = dist_pred[..., 2]
        dist_pred_b = dist_pred[..., 3]
        inter_width = tf.minimum(dist_l, dist_pred_l) + tf.minimum(dist_r, dist_pred_r)
        inter_height = tf.minimum(dist_t, dist_pred_t) + tf.minimum(dist_b, dist_pred_b)
        inter_area = inter_width * inter_height
        union_area = (dist_l+dist_r)*(dist_t+dist_b) + (dist_pred_l+dist_pred_r)*(dist_pred_t+dist_pred_b) - inter_area
        iou = inter_area / (union_area+1e-12)
        iou_loss = tf.reduce_sum(-tf.log(iou+1e-12)*loc)

        lr_min = tf.minimum(dist_l, dist_r)
        tb_min = tf.minimum(dist_t, dist_b)
        lr_max = tf.maximum(dist_l, dist_r)
        tb_max = tf.maximum(dist_t, dist_b)
        center_pred = tf.squeeze(center_pred)
        center_gt = tf.sqrt(lr_min*tb_min/(lr_max*tb_max+1e-12))
        # center_loss = tf.square(center_pred - center_gt)
        center_loss = tf.keras.backend.binary_crossentropy(output=center_pred, target=center_gt, from_logits=True)
        center_loss = tf.reduce_sum(center_loss)

        zero_like_heat = tf.expand_dims(tf.zeros(pshape, dtype=tf.float32), axis=-1)
        heatmap_gt = []
        for i in range(self.num_classes):
            exist_i = tf.equal(class_id, i)
            heatmask_i = tf.boolean_mask(heatmask, exist_i, axis=2)
            heatmap_i = tf.cond(
                tf.equal(tf.shape(heatmask_i)[-1], 0),
                lambda: zero_like_heat,
                lambda: tf.reduce_max(heatmask_i, axis=2, keepdims=True)
            )
            heatmap_gt.append(heatmap_i)
        heatmap_gt = tf.concat(heatmap_gt, axis=-1)
        heatmap_pos_loss = -.25 * tf.pow(1.-tf.sigmoid(heatmap_pred), 2.) * tf.log_sigmoid(heatmap_pred) * heatmap_gt
        heatmap_neg_loss = -.25 * tf.pow(tf.sigmoid(heatmap_pred), 2.) * (-heatmap_pred+tf.log_sigmoid(heatmap_pred)) * (1.-heatmap_gt)
        heatmap_loss = tf.reduce_sum(heatmap_pos_loss) + tf.reduce_sum(heatmap_neg_loss)
        total_loss = (iou_loss + heatmap_loss + center_loss) / tf.reduce_sum(heatmap_gt)
        return total_loss

    def _detect_head(self, bottom):
        with tf.variable_scope('classifier_head', reuse=tf.AUTO_REUSE):
            conv1 = self._bn_activation_conv(bottom, 256, 3, 1)
            conv2 = self._bn_activation_conv(conv1, 256, 3, 1)
            conv3 = self._bn_activation_conv(conv2, 256, 3, 1)
            conv4 = self._bn_activation_conv(conv3, 256, 3, 1)
            pconf = self._bn_activation_conv(conv4, self.num_classes, 3, 1, pi_init=True)
            pcenterness = self._bn_activation_conv(conv4, 1, 3, 1, pi_init=True)
        with tf.variable_scope('regress_head', reuse=tf.AUTO_REUSE):
            conva = self._bn_activation_conv(bottom, 256, 3, 1)
            convb = self._bn_activation_conv(conva, 256, 3, 1)
            convc = self._bn_activation_conv(convb, 256, 3, 1)
            convd = self._bn_activation_conv(convc, 256, 3, 1)
            preg = tf.exp(self._bn_activation_conv(convd, 4, 3, 1))
        return pconf, preg, pcenterness

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

    def _init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        if self.mode == 'train':
            self.sess.run(self.train_initializer)

    def _create_saver(self):
        weights = tf.trainable_variables('backone')
        self.pretrained_saver = tf.train.Saver(weights)
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
        assert (mode in ['latest', 'best'])
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

    def load_pretrained_weight(self, path):
        self.pretrained_saver.restore(self.sess, path)
        print('load pretrained weight', path, 'successfully')

    def _gn(self, bottom):
        gn = tf.contrib.layers.group_norm(
            inputs=bottom,
            groups=8,
            channels_axis=3 if self.data_format == 'channels_last' else 1,
            reduction_axes=(1, 2) if self.data_format == 'channels_last' else (2, 3),
            trainable=self.is_training
        )
        return gn

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
        bn = self._gn(conv)
        if activation is not None:
            return activation(bn)
        else:
            return bn

    def _bn_activation_conv(self, bottom, filters, kernel_size, strides, activation=tf.nn.relu, pi_init=False):
        bn = self._gn(bottom)
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
                bias_initializer=tf.constant_initializer(-math.log((1 - 0.01) / 0.01))
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
                conv = self._bn_activation_conv(conv, filters * 4, 1, 1)
            with tf.variable_scope('identity_branch'):
                shutcut = self._bn_activation_conv(bottom, filters * 4, 3, strides)

        return conv + shutcut

    def _max_pooling(self, bottom, pool_size, strides, name=None):
        return tf.layers.max_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )

    def _avg_pooling(self, bottom, pool_size, strides, name=None):
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
