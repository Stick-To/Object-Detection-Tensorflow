from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import sys
import datetime


class YOLOv3:
    def __init__(self, config, data_provider):

        assert len(config['data_shape']) == 3
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

        self.coord_sacle = config['coord_scale']
        self.noobj_scale = config['noobj_scale']
        self.obj_scale = config['obj_scale']
        self.class_scale = config['class_scale']
        self.num_priors = config['num_priors']

        self.nms_score_threshold = config['nms_score_threshold']
        self.nms_max_boxes = config['nms_max_boxes']
        self.nms_iou_threshold = config['nms_iou_threshold']

        priors = config['priors']
        self.stride = [8., 16., 32.]
        self.priors = []
        for i in range(len(priors)):
            self.priors.append(tf.reshape(tf.constant(priors[i], dtype=tf.float32)/self.stride[i], [1, 1, -1, 2]))
        self.final_units = (self.num_classes + 5) * self.num_priors

        if self.mode == 'train':
            self.num_train = data_provider['num_train']
            self.train_generator = data_provider['train_generator']
            self.train_initializer, self.train_iterator = self.train_generator
            if data_provider['val_generator'] is not None:
                self.num_val = data_provider['num_val']
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
            pyd1, pyd2, pyd3= self._feature_extractor(self.images)
        with tf.variable_scope('head'):
            pred1, top_down = self._yolo3_header(pyd1, 1024, 'pyd1', )
            pred2, top_down = self._yolo3_header(pyd2, 256, 'pyd2', top_down)
            pred3, _ = self._yolo3_header(pyd3, 128, 'pyd3', top_down)
            if self.data_format != 'channels_last':
                pred1 = tf.transpose(pred1, [0, 2, 3, 1])
                pred2 = tf.transpose(pred2, [0, 2, 3, 1])
                pred3 = tf.transpose(pred3, [0, 2, 3, 1])
            p1shape = tf.shape(pred1)
            p2shape = tf.shape(pred2)
            p3shape = tf.shape(pred3)
            pred1 = tf.reshape(pred1, [p1shape[0], p1shape[1], p1shape[2], self.num_priors, -1])
            pred2 = tf.reshape(pred2, [p2shape[0], p2shape[1], p2shape[2], self.num_priors, -1])
            pred3 = tf.reshape(pred3, [p3shape[0], p3shape[1], p3shape[2], self.num_priors, -1])

            p1class = pred1[..., :self.num_classes]
            p1bbox_yx = pred1[..., self.num_classes:self.num_classes+2]
            p1bbox_hw = pred1[..., self.num_classes+2:self.num_classes+4]
            p1obj = pred1[..., self.num_classes+4:]
            p2class = pred2[..., :self.num_classes]
            p2bbox_yx = pred2[..., self.num_classes:self.num_classes + 2]
            p2bbox_hw = pred2[..., self.num_classes + 2:self.num_classes + 4]
            p2obj = pred2[..., self.num_classes + 4:]
            p3class = pred3[..., :self.num_classes]
            p3bbox_yx = pred3[..., self.num_classes:self.num_classes + 2]
            p3bbox_hw = pred3[..., self.num_classes + 2:self.num_classes + 4]
            p3obj = pred3[..., self.num_classes + 4:]
            a1bbox_yx, a1bbox_hw, a1bbox_y1x1, a1bbox_y2x2 = self._get_priors(p1shape, self.priors[0])
            a2bbox_yx, a2bbox_hw, a2bbox_y1x1, a2bbox_y2x2 = self._get_priors(p2shape, self.priors[1])
            a3bbox_yx, a3bbox_hw, a3bbox_y1x1, a3bbox_y2x2 = self._get_priors(p3shape, self.priors[2])

        if self.mode == 'train':
            total_loss = []
            for i in range(self.batch_size):
                gn1_yxi, gn1_hwi, gn1_labeli = self._get_normlized_gn(self.stride[-1], i)
                gn2_yxi, gn2_hwi, gn2_labeli = self._get_normlized_gn(self.stride[-2], i)
                gn3_yxi, gn3_hwi, gn3_labeli = self._get_normlized_gn(self.stride[-3], i)
                num_g = tf.shape(gn2_labeli)[0]
                num_gf = tf.cast(num_g, tf.float32)
                gn1_floor_ = tf.cast(tf.floor(gn1_yxi), tf.int64)
                gn2_floor_ = tf.cast(tf.floor(gn2_yxi), tf.int64)
                gn3_floor_ = tf.cast(tf.floor(gn3_yxi), tf.int64)
                nogn1_mask = tf.sparse.SparseTensor(gn1_floor_, tf.ones_like(gn1_floor_[..., 0]), dense_shape=[p1shape[1], p1shape[2]])
                nogn1_mask = (1 - tf.sparse.to_dense(nogn1_mask, validate_indices=False)) > 0
                nogn2_mask = tf.sparse.SparseTensor(gn2_floor_, tf.ones_like(gn2_floor_[..., 0]), dense_shape=[p2shape[1], p2shape[2]])
                nogn2_mask = (1 - tf.sparse.to_dense(nogn2_mask, validate_indices=False)) > 0
                nogn3_mask = tf.sparse.SparseTensor(gn3_floor_, tf.ones_like(gn3_floor_[..., 0]), dense_shape=[p3shape[1], p3shape[2]])
                nogn3_mask = (1 - tf.sparse.to_dense(nogn3_mask, validate_indices=False)) > 0
                rp1bbox_yx = tf.gather_nd(p1bbox_yx[i, ...], gn1_floor_)
                rp2bbox_yx = tf.gather_nd(p2bbox_yx[i, ...], gn2_floor_)
                rp3bbox_yx = tf.gather_nd(p3bbox_yx[i, ...], gn3_floor_)
                rp1bbox_hw = tf.gather_nd(p1bbox_hw[i, ...], gn1_floor_)
                rp2bbox_hw = tf.gather_nd(p2bbox_hw[i, ...], gn2_floor_)
                rp3bbox_hw = tf.gather_nd(p3bbox_hw[i, ...], gn3_floor_)
                ra1bbox_hw = tf.gather_nd(a1bbox_hw, gn1_floor_)
                ra2bbox_hw = tf.gather_nd(a2bbox_hw, gn2_floor_)
                ra3bbox_hw = tf.gather_nd(a3bbox_hw, gn3_floor_)
                rp1class = tf.gather_nd(p1class[i, ...], gn1_floor_)
                rp2class = tf.gather_nd(p2class[i, ...], gn2_floor_)
                rp3class = tf.gather_nd(p3class[i, ...], gn3_floor_)
                rp1obj = tf.gather_nd(p1obj[i, ...], gn1_floor_)
                rp2obj = tf.gather_nd(p2obj[i, ...], gn2_floor_)
                rp3obj = tf.gather_nd(p3obj[i, ...], gn3_floor_)
                ra1bbox_y1x1 = tf.gather_nd(a1bbox_y1x1, gn1_floor_)
                ra2bbox_y1x1 = tf.gather_nd(a2bbox_y1x1, gn2_floor_)
                ra3bbox_y1x1 = tf.gather_nd(a3bbox_y1x1, gn3_floor_)
                ra1bbox_y2x2 = tf.gather_nd(a1bbox_y2x2, gn1_floor_)
                ra2bbox_y2x2 = tf.gather_nd(a2bbox_y2x2, gn2_floor_)
                ra3bbox_y2x2 = tf.gather_nd(a3bbox_y2x2, gn3_floor_)
                gn1_y1x1i = tf.expand_dims(gn1_yxi - gn1_hwi / 2., axis=1)
                gn1_y2x2i = tf.expand_dims(gn1_yxi + gn1_hwi / 2., axis=1)
                gn2_y1x1i = tf.expand_dims(gn2_yxi - gn2_hwi / 2., axis=1)
                gn2_y2x2i = tf.expand_dims(gn2_yxi + gn2_hwi / 2., axis=1)
                gn3_y1x1i = tf.expand_dims(gn3_yxi - gn3_hwi / 2., axis=1)
                gn3_y2x2i = tf.expand_dims(gn3_yxi + gn3_hwi / 2., axis=1)
                rga1iou_y1x1 = tf.maximum(gn1_y1x1i, ra1bbox_y1x1)
                rga2iou_y1x1 = tf.maximum(gn2_y1x1i, ra2bbox_y1x1)
                rga3iou_y1x1 = tf.maximum(gn3_y1x1i, ra3bbox_y1x1)
                rga1iou_y2x2 = tf.minimum(gn1_y2x2i, ra1bbox_y2x2)
                rga2iou_y2x2 = tf.minimum(gn2_y2x2i, ra2bbox_y2x2)
                rga3iou_y2x2 = tf.minimum(gn3_y2x2i, ra3bbox_y2x2)
                rga1iou_area = tf.reduce_prod(rga1iou_y2x2 - rga1iou_y1x1, axis=-1)
                rga2iou_area = tf.reduce_prod(rga2iou_y2x2 - rga2iou_y1x1, axis=-1)
                rga3iou_area = tf.reduce_prod(rga3iou_y2x2 - rga3iou_y1x1, axis=-1)

                g1area = tf.reduce_prod(gn1_y2x2i - gn1_y1x1i, axis=-1)
                g2area = tf.reduce_prod(gn2_y2x2i - gn2_y1x1i, axis=-1)
                g3area = tf.reduce_prod(gn3_y2x2i - gn3_y1x1i, axis=-1)
                a1area = tf.reduce_prod(ra1bbox_hw, axis=-1)
                a2area = tf.reduce_prod(ra2bbox_hw, axis=-1)
                a3area = tf.reduce_prod(ra3bbox_hw, axis=-1)
                rga1iou = rga1iou_area / (a1area + g1area - rga1iou_area)
                rga2iou = rga2iou_area / (a2area + g2area - rga2iou_area)
                rga3iou = rga3iou_area / (a3area + g3area - rga3iou_area)

                rga1index = tf.expand_dims(tf.cast(tf.argmax(rga1iou, axis=-1), tf.int32), -1)
                rga2index = tf.expand_dims(tf.cast(tf.argmax(rga2iou, axis=-1), tf.int32), -1)
                rga3index = tf.expand_dims(tf.cast(tf.argmax(rga3iou, axis=-1), tf.int32), -1)
                rga1iou_max = tf.reduce_max(rga1iou, axis=-1)
                rga2iou_max = tf.reduce_max(rga2iou, axis=-1)
                rga3iou_max = tf.reduce_max(rga3iou, axis=-1)
                rga1mask = tf.cast(rga1iou_max > rga2iou_max, tf.float32) * tf.cast(rga1iou_max > rga3iou_max, tf.float32)
                rga2mask = tf.cast(rga2iou_max > rga1iou_max, tf.float32) * tf.cast(rga2iou_max > rga3iou_max, tf.float32)
                rga3mask = (1. - tf.cast(rga1mask + rga2mask, tf.float32)) > 0.
                rga2mask = rga2mask > 0.
                rga1mask = rga1mask > 0.
                rga1index = tf.boolean_mask(rga1index, rga1mask)
                rga2index = tf.boolean_mask(rga2index, rga2mask)
                rga3index = tf.boolean_mask(rga3index, rga3mask)
                rga1index = tf.concat([tf.expand_dims(tf.range(tf.shape(rga1index)[0]), -1), rga1index], axis=-1)
                rga2index = tf.concat([tf.expand_dims(tf.range(tf.shape(rga2index)[0]), -1), rga2index], axis=-1)
                rga3index = tf.concat([tf.expand_dims(tf.range(tf.shape(rga3index)[0]), -1), rga3index], axis=-1)

                rp1bbox_yx = tf.reshape(tf.gather_nd(tf.boolean_mask(rp1bbox_yx, rga1mask), rga1index), [-1, 2])
                rp2bbox_yx = tf.reshape(tf.gather_nd(tf.boolean_mask(rp2bbox_yx, rga2mask), rga2index), [-1, 2])
                rp3bbox_yx = tf.reshape(tf.gather_nd(tf.boolean_mask(rp3bbox_yx, rga3mask), rga3index), [-1, 2])
                rp1bbox_hw = tf.reshape(tf.gather_nd(tf.boolean_mask(rp1bbox_hw, rga1mask), rga1index), [-1, 2])
                rp2bbox_hw = tf.reshape(tf.gather_nd(tf.boolean_mask(rp2bbox_hw, rga2mask), rga2index), [-1, 2])
                rp3bbox_hw = tf.reshape(tf.gather_nd(tf.boolean_mask(rp3bbox_hw, rga3mask), rga3index), [-1, 2])
                rp1class = tf.reshape(tf.gather_nd(tf.boolean_mask(rp1class, rga1mask), rga1index), [-1, self.num_classes])
                rp2class = tf.reshape(tf.gather_nd(tf.boolean_mask(rp2class, rga2mask), rga2index), [-1, self.num_classes])
                rp3class = tf.reshape(tf.gather_nd(tf.boolean_mask(rp3class, rga3mask), rga3index), [-1, self.num_classes])
                rp1obj = tf.reshape(tf.gather_nd(tf.boolean_mask(rp1obj, rga1mask), rga1index), [-1, 1])
                rp2obj = tf.reshape(tf.gather_nd(tf.boolean_mask(rp2obj, rga2mask), rga2index), [-1, 1])
                rp3obj = tf.reshape(tf.gather_nd(tf.boolean_mask(rp3obj, rga3mask), rga3index), [-1, 1])
                ra1bbox_hw = tf.reshape(tf.gather_nd(tf.boolean_mask(ra1bbox_hw, rga1mask), rga1index), [-1, 2])
                ra2bbox_hw = tf.reshape(tf.gather_nd(tf.boolean_mask(ra2bbox_hw, rga2mask), rga2index), [-1, 2])
                ra3bbox_hw = tf.reshape(tf.gather_nd(tf.boolean_mask(ra3bbox_hw, rga3mask), rga3index), [-1, 2])
                gn1_yxi = tf.boolean_mask(gn1_yxi, rga1mask)
                gn1_hwi = tf.boolean_mask(gn1_hwi, rga1mask)
                gn2_yxi = tf.boolean_mask(gn2_yxi, rga2mask)
                gn2_hwi = tf.boolean_mask(gn2_hwi, rga2mask)
                gn3_yxi = tf.boolean_mask(gn3_yxi, rga3mask)
                gn3_hwi = tf.boolean_mask(gn3_hwi, rga3mask)
                gn1_labeli = tf.one_hot(tf.boolean_mask(gn1_labeli, rga1mask), self.num_classes)
                gn2_labeli = tf.one_hot(tf.boolean_mask(gn2_labeli, rga2mask), self.num_classes)
                gn3_labeli = tf.one_hot(tf.boolean_mask(gn3_labeli, rga3mask), self.num_classes)
                rp1bbox_yx_target = gn1_yxi - tf.floor(gn1_yxi)
                rp2bbox_yx_target = gn2_yxi - tf.floor(gn2_yxi)
                rp3bbox_yx_target = gn3_yxi - tf.floor(gn3_yxi)
                rp1bbox_hw_target = tf.log(gn1_hwi/ra1bbox_hw)
                rp2bbox_hw_target = tf.log(gn2_hwi/ra2bbox_hw)
                rp3bbox_hw_target = tf.log(gn3_hwi/ra3bbox_hw)
                yx_loss1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=rp1bbox_yx_target, logits=rp1bbox_yx))
                yx_loss2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=rp2bbox_yx_target, logits=rp2bbox_yx))
                yx_loss3 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=rp3bbox_yx_target, logits=rp3bbox_yx))
                hw_loss1 = 0.5 * tf.reduce_sum(tf.square(rp1bbox_hw - rp1bbox_hw_target))
                hw_loss2 = 0.5 * tf.reduce_sum(tf.square(rp2bbox_hw - rp2bbox_hw_target))
                hw_loss3 = 0.5 * tf.reduce_sum(tf.square(rp3bbox_hw - rp3bbox_hw_target))
                coord_loss = (yx_loss1 + yx_loss2 + yx_loss3 + hw_loss1 + hw_loss2 + hw_loss3)
                class_loss1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=gn1_labeli, logits=rp1class))
                class_loss2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=gn2_labeli, logits=rp2class))
                class_loss3 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=gn3_labeli, logits=rp3class))
                class_loss = (class_loss1 + class_loss2 + class_loss3)
                obj_loss1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(rp1obj), logits=rp1obj))
                obj_loss2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(rp2obj), logits=rp2obj))
                obj_loss3 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(rp3obj), logits=rp3obj))
                obj_loss = (obj_loss1 + obj_loss2 + obj_loss3)
                nogn1_mask = tf.reshape(nogn1_mask, [-1])
                nogn2_mask = tf.reshape(nogn2_mask, [-1])
                nogn3_mask = tf.reshape(nogn3_mask, [-1])

                a1bbox_yx_nobest = tf.expand_dims(tf.boolean_mask(tf.reshape(a1bbox_yx-a1bbox_hw/2., [-1, self.num_priors, 2]), nogn1_mask), 1)
                a1bbox_hw_nobest = tf.expand_dims(tf.boolean_mask(tf.reshape(a1bbox_yx+a1bbox_hw/2., [-1, self.num_priors, 2]), nogn1_mask), 1)
                a2bbox_yx_nobest = tf.expand_dims(tf.boolean_mask(tf.reshape(a2bbox_yx-a2bbox_hw/2., [-1, self.num_priors, 2]), nogn2_mask), 1)
                a2bbox_hw_nobest = tf.expand_dims(tf.boolean_mask(tf.reshape(a2bbox_yx+a2bbox_hw/2., [-1, self.num_priors, 2]), nogn2_mask), 1)
                a3bbox_yx_nobest = tf.expand_dims(tf.boolean_mask(tf.reshape(a3bbox_yx-a3bbox_hw/2., [-1, self.num_priors, 2]), nogn3_mask), 1)
                a3bbox_hw_nobest = tf.expand_dims(tf.boolean_mask(tf.reshape(a3bbox_yx+a3bbox_hw/2., [-1, self.num_priors, 2]), nogn3_mask), 1)
                a1bbox_y1x1_nobest = a1bbox_yx_nobest - a1bbox_hw_nobest/2.
                a1bbox_y2x2_nobest = a1bbox_yx_nobest + a1bbox_hw_nobest/2.
                a2bbox_y1x1_nobest = a2bbox_yx_nobest - a2bbox_hw_nobest/2.
                a2bbox_y2x2_nobest = a2bbox_yx_nobest + a2bbox_hw_nobest/2.
                a3bbox_y1x1_nobest = a3bbox_yx_nobest - a3bbox_hw_nobest/2.
                a3bbox_y2x2_nobest = a3bbox_yx_nobest + a3bbox_hw_nobest/2.

                p1obj_nobest = tf.boolean_mask(tf.reshape(p1obj[i, ...], [-1, self.num_priors]), nogn1_mask)
                p2obj_nobest = tf.boolean_mask(tf.reshape(p2obj[i, ...], [-1, self.num_priors]), nogn2_mask)
                p3obj_nobest = tf.boolean_mask(tf.reshape(p3obj[i, ...], [-1, self.num_priors]), nogn3_mask)
                num_g1 = tf.shape(gn1_y1x1i)[0]
                num_g2 = tf.shape(gn2_y1x1i)[0]
                num_g3 = tf.shape(gn3_y1x1i)[0]
                num_a1 = tf.shape(a1bbox_y1x1_nobest)[0]
                num_a2 = tf.shape(a2bbox_y1x1_nobest)[0]
                num_a3 = tf.shape(a3bbox_y1x1_nobest)[0]
                gn1_y1x1i = tf.tile(tf.expand_dims(gn1_y1x1i, 0), [num_a1, 1, 1, 1])
                gn1_y2x2i = tf.tile(tf.expand_dims(gn1_y2x2i, 0), [num_a1, 1, 1, 1])
                gn2_y1x1i = tf.tile(tf.expand_dims(gn2_y1x1i, 0), [num_a2, 1, 1, 1])
                gn2_y2x2i = tf.tile(tf.expand_dims(gn2_y2x2i, 0), [num_a2, 1, 1, 1])
                gn3_y1x1i = tf.tile(tf.expand_dims(gn3_y1x1i, 0), [num_a3, 1, 1, 1])
                gn3_y2x2i = tf.tile(tf.expand_dims(gn3_y2x2i, 0), [num_a3, 1, 1, 1])
                a1bbox_y1x1_nobest = tf.tile(a1bbox_y1x1_nobest, [1, num_g1, 1, 1])
                a2bbox_y1x1_nobest = tf.tile(a2bbox_y1x1_nobest, [1, num_g2, 1, 1])
                a3bbox_y1x1_nobest = tf.tile(a3bbox_y1x1_nobest, [1, num_g3, 1, 1])
                a1bbox_y2x2_nobest = tf.tile(a1bbox_y2x2_nobest, [1, num_g1, 1, 1])
                a2bbox_y2x2_nobest = tf.tile(a2bbox_y2x2_nobest, [1, num_g2, 1, 1])
                a3bbox_y2x2_nobest = tf.tile(a3bbox_y2x2_nobest, [1, num_g3, 1, 1])
                ag1iou_y1x1 = tf.maximum(gn1_y1x1i, a1bbox_y1x1_nobest)
                ag1iou_y2x2 = tf.minimum(gn1_y2x2i, a1bbox_y2x2_nobest)
                ag2iou_y1x1 = tf.maximum(gn2_y1x1i, a2bbox_y1x1_nobest)
                ag2iou_y2x2 = tf.minimum(gn2_y2x2i, a2bbox_y2x2_nobest)
                ag3iou_y1x1 = tf.maximum(gn3_y1x1i, a3bbox_y1x1_nobest)
                ag3iou_y2x2 = tf.minimum(gn3_y2x2i, a3bbox_y2x2_nobest)
                ag1iou_area = tf.reduce_prod(ag1iou_y2x2 - ag1iou_y1x1, axis=-1)
                ag2iou_area = tf.reduce_prod(ag2iou_y2x2 - ag2iou_y1x1, axis=-1)
                ag3iou_area = tf.reduce_prod(ag3iou_y2x2 - ag3iou_y1x1, axis=-1)
                a1area = tf.reduce_prod(a1bbox_y2x2_nobest-a1bbox_y1x1_nobest, axis=-1)
                g1area = tf.reduce_prod(gn1_y2x2i - gn1_y1x1i, axis=-1)
                a2area = tf.reduce_prod(a2bbox_y2x2_nobest - a2bbox_y1x1_nobest, axis=-1)
                g2area = tf.reduce_prod(gn2_y2x2i - gn2_y1x1i, axis=-1)
                a3area = tf.reduce_prod(a3bbox_y2x2_nobest - a3bbox_y1x1_nobest, axis=-1)
                g3area = tf.reduce_prod(gn3_y2x2i - gn3_y1x1i, axis=-1)
                ag1iou = ag1iou_area / (a1area + g1area - ag1iou_area)
                ag2iou = ag2iou_area / (a2area + g2area - ag2iou_area)
                ag3iou = ag3iou_area / (a3area + g3area - ag3iou_area)
                ag1iou = tf.reduce_max(ag1iou, axis=1)
                ag2iou = tf.reduce_max(ag2iou, axis=1)
                ag3iou = tf.reduce_max(ag3iou, axis=1)
                ag1iou_noobj_mask = tf.cast(ag1iou <= 0.5, tf.float32)
                ag2iou_noobj_mask = tf.cast(ag2iou <= 0.5, tf.float32)
                ag3iou_noobj_mask = tf.cast(ag3iou <= 0.5, tf.float32)
                noonj_loss1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(p1obj_nobest), logits=p1obj_nobest) * ag1iou_noobj_mask)
                noonj_loss2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(p2obj_nobest), logits=p2obj_nobest) * ag2iou_noobj_mask)
                noonj_loss3 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(p3obj_nobest), logits=p3obj_nobest) * ag3iou_noobj_mask)
                noonj_loss = noonj_loss1 + noonj_loss2 + noonj_loss3
                pos_loss = (self.coord_sacle*coord_loss+self.class_scale*class_loss+self.obj_scale*obj_loss) / num_gf
                neg_loss = self.noobj_scale*noonj_loss / num_gf
                total_loss.append(pos_loss + neg_loss)
            total_loss = tf.reduce_mean(total_loss)
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
            self.loss = .5 * total_loss + self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables()]
            )
            train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.group([update_ops, train_op])
        else:
            p1class = tf.reshape(p1class[0, ...], [-1, self.num_classes])
            p2class = tf.reshape(p2class[0, ...], [-1, self.num_classes])
            p3class = tf.reshape(p3class[0, ...], [-1, self.num_classes])
            p1obj = tf.reshape(p1obj[0, ...], [-1, 1])
            p2obj = tf.reshape(p2obj[0, ...], [-1, 1])
            p3obj = tf.reshape(p3obj[0, ...], [-1, 1])
            p1bbox_yx = tf.reshape(p1bbox_yx[0, ...], [-1, 2])
            p2bbox_yx = tf.reshape(p2bbox_yx[0, ...], [-1, 2])
            p3bbox_yx = tf.reshape(p3bbox_yx[0, ...], [-1, 2])
            p1bbox_hw = tf.reshape(p1bbox_hw[0, ...], [-1, 2])
            p2bbox_hw = tf.reshape(p2bbox_hw[0, ...], [-1, 2])
            p3bbox_hw = tf.reshape(p3bbox_hw[0, ...], [-1, 2])
            a1bbox_yx = tf.reshape(a1bbox_yx, [-1, 2])
            a2bbox_yx = tf.reshape(a2bbox_yx, [-1, 2])
            a3bbox_yx = tf.reshape(a3bbox_yx, [-1, 2])
            a1bbox_hw = tf.reshape(a1bbox_hw, [-1, 2])
            a2bbox_hw = tf.reshape(a2bbox_hw, [-1, 2])
            a3bbox_hw = tf.reshape(a3bbox_hw, [-1, 2])
            pclasst = tf.sigmoid(tf.concat([p1class, p2class, p3class], axis=0))
            pobjt = tf.sigmoid(tf.concat([p1obj, p2obj, p3obj], axis=0))
            bbox_yx1 = a1bbox_yx + tf.sigmoid(p1bbox_yx)
            bbox_hw1 = a1bbox_hw + tf.exp(p1bbox_hw)
            bbox_yx2 = a2bbox_yx + tf.sigmoid(p2bbox_yx)
            bbox_hw2 = a2bbox_hw + tf.exp(p2bbox_hw)
            bbox_yx3 = a3bbox_yx + tf.sigmoid(p3bbox_yx)
            bbox_hw3 = a3bbox_hw + tf.exp(p3bbox_hw)
            bbox1_y1x1y2x2 = tf.concat([bbox_yx1 - bbox_hw1 / 2., bbox_yx1 + bbox_hw1 / 2.], axis=-1) * self.stride[-1]
            bbox2_y1x1y2x2 = tf.concat([bbox_yx2 - bbox_hw2 / 2., bbox_yx2 + bbox_hw2 / 2.], axis=-1) * self.stride[-1]
            bbox3_y1x1y2x2 = tf.concat([bbox_yx3 - bbox_hw3 / 2., bbox_yx3 + bbox_hw3 / 2.], axis=-1) * self.stride[-2]
            bbox_y1x1y2x2 = tf.concat([bbox1_y1x1y2x2, bbox2_y1x1y2x2, bbox3_y1x1y2x2], axis=0)
            confidence = pclasst * pobjt
            filter_mask = tf.greater_equal(confidence, self.nms_score_threshold)
            scores = []
            class_id = []
            bbox = []
            for i in range(self.num_classes):
                scoresi = tf.boolean_mask(confidence[:, i], filter_mask[:, i])
                bboxi = tf.boolean_mask(bbox_y1x1y2x2, filter_mask[:, i])
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

    def _init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        if self.mode == 'train':
            self.sess.run(self.train_initializer)

    def _create_saver(self):
        weights = tf.trainable_variables(scope='backone')
        self.pretraining_weight_saver = tf.train.Saver(weights)
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()

    def _create_summary(self):
        with tf.variable_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def _feature_extractor(self, image):
        init_conv = self._conv_layer(image, 32, 3, 1)
        block1 = self._darknet_block(init_conv, 64, 1, 'block1')
        block2 = self._darknet_block(block1, 128, 2, 'block2')
        block3 = self._darknet_block(block2, 256, 8, 'block3')
        block4 = self._darknet_block(block3, 512, 8, 'block4')
        block5 = self._darknet_block(block4, 1024, 4, 'block5')
        return block5, block4, block3

    def _yolo3_header(self, bottom, filters, scope, pyramid=None):
        with tf.variable_scope(scope):
            if pyramid is not None:
                if self.data_format == 'channels_last':
                    axes = 3
                    shape = [int(bottom.get_shape()[1]), int(bottom.get_shape()[2])]
                else:
                    axes = 1
                    shape = [int(bottom.get_shape()[2]), int(bottom.get_shape()[3])]
                conv = self._conv_layer(pyramid, filters, 1, 1, False)
                conv = tf.image.resize_nearest_neighbor(conv, shape)
                conv = tf.concat([bottom, conv], axis=axes)
            else:
                conv = bottom
            conv1 = self._conv_layer(conv, filters/2, 1, 1)
            conv2 = self._conv_layer(conv1, filters, 3, 1)
            conv3 = self._conv_layer(conv2, filters/2, 1, 1)
            conv4 = self._conv_layer(conv3, filters, 3, 1)
            conv5 = self._conv_layer(conv4, filters/2, 1, 1)
            conv6 = self._conv_layer(conv5, filters, 3, 1)
            pred = self._conv_layer(conv6, self.final_units, 1, 1)
            return pred, conv5

    def _get_priors(self, pshape, priors):
        tl_y = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)
        tl_x = tf.range(0., tf.cast(pshape[2], tf.float32), dtype=tf.float32)
        tl_y_ = tf.reshape(tl_y, [-1, 1, 1, 1])
        tl_x_ = tf.reshape(tl_x, [1, -1, 1, 1])
        tl_y_ = tf.tile(tl_y_, [1, pshape[2], 1, 1])
        tl_x_ = tf.tile(tl_x_, [pshape[1], 1, 1, 1])
        tl = tf.concat([tl_y_, tl_x_], -1)
        abbox_yx = tl + 0.5
        abbox_yx = tf.tile(abbox_yx, [1, 1, self.num_priors, 1])
        abbox_hw = priors
        abbox_hw = tf.tile(abbox_hw, [pshape[1], pshape[2], 1, 1])
        abbox_y1x1 = abbox_yx - abbox_hw / 2
        abbox_y2x2 = abbox_yx + abbox_hw / 2
        return abbox_yx, abbox_hw, abbox_y1x1, abbox_y2x2

    def _get_normlized_gn(self, downsampling_rate, i):

        slice_index = tf.argmin(self.ground_truth[i, ...], axis=0)[0]
        ground_truth = tf.gather(self.ground_truth[i, ...], tf.range(0, slice_index, dtype=tf.int64))
        scale = tf.constant([downsampling_rate, downsampling_rate, downsampling_rate, downsampling_rate, 1], dtype=tf.float32)
        scale = tf.reshape(scale, [1, 5])
        gn = ground_truth / scale
        return gn[..., :2], gn[..., 2:4], tf.cast(gn[..., 4], tf.int32)

    def train_one_epoch(self, lr):
        self.is_training = True
        self.sess.run(self.train_initializer)
        mean_loss = []
        num_iters = self.num_train // self.batch_size
        for i in range(num_iters):
            _, loss, _ = self.sess.run([self.train_op, self.loss, self.summary_op],
                                       feed_dict={self.lr: lr})
            sys.stdout.write('\r>> ' + 'iters '+str(i)+str('/')+str(num_iters)+' loss '+str(loss))
            sys.stdout.flush()
            mean_loss.append(loss)
        sys.stdout.write('\n')
        mean_loss = np.mean(mean_loss)
        return mean_loss

    def test_one_image(self, images):
        self.is_training = False
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))

        pred = self.sess.run(self.detection_pred, feed_dict={self.images: images})
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
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

    def load_pretraining_weight(self, path):
        self.pretraining_weight_saver.restore(self.sess, path)
        print('load pretraining weight', path, 'successfully')

    def _darknet_block(self, bottom, filters, blocks, scope):
        with tf.variable_scope(scope):
            conv = self._conv_layer(bottom, filters, 3, 2)
            for i in range(1, blocks+1):
                conv1 = self._conv_layer(conv, filters/2, 1, 1)
                conv2 = self._conv_layer(conv1, filters, 3, 1)
                conv = conv + conv2
            return conv

    def _conv_layer(self, bottom, filters, kernel_size, strides, is_activation=True):
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
        if is_activation:
            bn = tf.nn.leaky_relu(bn, 0.1)
        return bn

    def _bn(self, bottom):
        bn = tf.layers.batch_normalization(
            inputs=bottom,
            axis=3 if self.data_format == 'channels_last' else 1,
            training=self.is_training
        )
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
