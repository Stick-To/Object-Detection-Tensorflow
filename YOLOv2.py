from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import sys
import numpy as np


class YOLOv2:
    def __init__(self, config, data_provider):

        assert len(config['data_shape']) == 3
        assert config['mode'] in ['train', 'test']
        assert config['data_format'] in ['channels_first', 'channels_last']
        self.config = config
        self.data_provider = data_provider
        self.is_pretraining = config['is_pretraining']
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
        self.nms_score_threshold = config['nms_score_threshold']
        self.nms_max_boxes = config['nms_max_boxes']
        self.nms_iou_threshold = config['nms_iou_threshold']
        self.rescore_confidence = config['rescore_confidence']
        self.num_priors = len(config['priors'])
        priors = tf.convert_to_tensor(config['priors'], dtype=tf.float32)
        self.priors = tf.reshape(priors, [1, 1, 1, self.num_priors, 2])
        self.final_units = (self.num_classes + 5) * self.num_priors

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
            features, _, _ = self._feature_extractor(self.images)
        with tf.variable_scope('pretraining'):
            conv = self._conv_layer(features, self.num_classes, 1, 1, 'conv1')
            axes = [1, 2] if self.data_format == 'channels_last' else [2, 3]
            global_pool = tf.reduce_mean(conv, axis=axes, name='global_pool')
            labels = tf.one_hot(self.labels, self.num_classes)
            labels = tf.reshape(labels, [self.batch_size, self.num_classes])
            loss = tf.losses.softmax_cross_entropy(labels, global_pool, reduction=tf.losses.Reduction.MEAN)
            self.pred = tf.argmax(global_pool, 1)
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.pred, tf.argmax(labels, 1)), tf.float32), name='accuracy'
            )
            self.loss = loss + self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables('feature_extractor')]
            ) + self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables('pretraining')]
            )
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def _build_detection_graph(self):
        with tf.variable_scope('feature_extractor'):
            features, passthrough, downsampling_rate = self._feature_extractor(self.images)
        with tf.variable_scope('regressor'):
            conv1 = self._conv_layer(features, 1024, 3, 1, 'conv1')
            lrelu1 = tf.nn.leaky_relu(conv1, 0.1, 'lrelu1')
            conv2 = self._conv_layer(lrelu1, 512, 1, 1, 'conv2')
            lrelu2 = tf.nn.leaky_relu(conv2, 0.1, 'lrelu2')
            conv3 = self._conv_layer(lrelu2, 1024, 3, 1, 'conv3')
            lrelu3 = tf.nn.leaky_relu(conv3, 0.1, 'lrelu3')
            conv4 = self._conv_layer(lrelu3, 512, 1, 1, 'conv4')
            lrelu4 = tf.nn.leaky_relu(conv4, 0.1, 'lrelu4')
            conv5 = self._conv_layer(lrelu4, 1024, 3, 1, 'conv5')
            lrelu5 = tf.nn.leaky_relu(conv5, 0.1, 'lrelu5')
            axes = 3 if self.data_format == 'channels_last' else 1
            lrelu5 = tf.concat([passthrough, lrelu5], axis=axes)
            pred = self._conv_layer(lrelu5, self.final_units, 1, 1, 'predictions')
            if self.data_format == 'channels_first':
                pred = tf.transpose(pred, [0, 2, 3, 1])
            pshape = tf.shape(pred)

        with tf.variable_scope('inference'):
            topleft_y = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)
            topleft_x = tf.range(0., tf.cast(pshape[2], tf.float32), dtype=tf.float32)
            topleft_y = tf.reshape(topleft_y, [1, -1, 1, 1, 1])
            topleft_x = tf.reshape(topleft_x, [1, 1, -1, 1, 1])
            topleft_y = tf.tile(topleft_y, [1, 1, pshape[2], 1, 1])
            topleft_x = tf.tile(topleft_x, [1, pshape[1], 1, 1, 1])
            topleft = tf.concat([topleft_y, topleft_x], -1)

            pclass = pred[..., :self.num_classes*self.num_priors]
            pbbox_yx = pred[..., self.num_classes*self.num_priors:self.num_classes*self.num_priors+self.num_priors*2]
            pbbox_hw = pred[..., self.num_classes*self.num_priors+self.num_priors*2:self.num_classes*self.num_priors+self.num_priors*4]
            pconf = pred[..., self.num_classes*self.num_priors+self.num_priors*4:]

            pclasst = tf.nn.softmax(tf.reshape(pclass, [self.batch_size, -1, self.num_classes]))
            pbbox_yx = tf.nn.sigmoid(tf.reshape(pbbox_yx, [self.batch_size, pshape[1], pshape[2], self.num_priors, 2]))
            pbbox_hw = tf.reshape(pbbox_hw, [self.batch_size, pshape[1], pshape[2], self.num_priors, 2])
            pbbox_loss = tf.concat([pbbox_yx, pbbox_hw], axis=-1)
            pbbox_loss = tf.reshape(pbbox_loss, [self.batch_size, -1, 4])

            pconft = tf.nn.sigmoid(tf.reshape(pconf, [self.batch_size, -1]))
            npbbox_yx = pbbox_yx + topleft
            npbbox_hw = tf.exp(pbbox_hw) * self.priors
            npbbox_y1x1 = npbbox_yx - npbbox_hw / 2
            npbbox_y2x2 = npbbox_yx + npbbox_hw / 2
            npbbox_y1x1 = tf.reshape(npbbox_y1x1, [self.batch_size, -1, 2])
            npbbox_y2x2 = tf.reshape(npbbox_y2x2, [self.batch_size, -1, 2])
            npbbox_y1x1y2x2t = tf.concat([npbbox_y1x1, npbbox_y2x2], axis=-1)
            npbbox_y1x1y2x2t = tf.reshape(npbbox_y1x1y2x2t, [self.batch_size, -1, 4]) * downsampling_rate
            npbbox_hw = tf.reshape(npbbox_hw, [self.batch_size, -1, 2])

        if self.mode == 'train':
            total_loss = []
            scale = tf.constant([downsampling_rate, downsampling_rate, downsampling_rate, downsampling_rate, 1], dtype=tf.float32)
            scale = tf.reshape(scale, [1, 1, 5])
            ground_truth = self.ground_truth / scale
            abbox_yx = topleft + 0.5
            abbox_yx = tf.tile(abbox_yx, [1, 1, 1, self.num_priors, 1])
            abbox_hw = self.priors
            abbox_hw = tf.tile(abbox_hw, [1, pshape[1], pshape[2], 1, 1])
            abbox_y1x1 = abbox_yx - abbox_hw / 2
            abbox_y2x2 = abbox_yx + abbox_hw / 2
            abbox_hw = tf.reshape(abbox_hw, [1, -1, 2])
            abbox_y1x1 = tf.reshape(abbox_y1x1, [1, -1, 2])
            abbox_y2x2 = tf.reshape(abbox_y2x2, [1, -1, 2])
            k = 0.
            lossi = 0.
            cond = lambda loss, i: tf.less(i, tf.cast(self.batch_size, tf.float32))
            body = lambda loss, i: (
                tf.add(loss, self._compute_one_image_loss(
                    tf.squeeze(tf.gather(pclasst, tf.cast(i, tf.int32))),
                    tf.squeeze(tf.gather(pbbox_loss, tf.cast(i, tf.int32))),
                    tf.squeeze(tf.gather(npbbox_hw, tf.cast(i, tf.int32))),
                    tf.squeeze(tf.gather(npbbox_y1x1, tf.cast(i, tf.int32))),
                    tf.squeeze(tf.gather(npbbox_y2x2, tf.cast(i, tf.int32))),
                    tf.squeeze(tf.gather(pconft, tf.cast(i, tf.int32))),
                    abbox_hw,
                    abbox_y1x1,
                    abbox_y2x2,
                    tf.squeeze(tf.gather(ground_truth, tf.cast(i, tf.int32))),
                )),
                tf.add(i, 1.)
            )
            init_state = (lossi, k)
            state = tf.while_loop(cond, body, init_state, parallel_iterations=self.batch_size)
            total_loss, _ = state
            total_loss = total_loss / self.batch_size
            # for i in range(self.batch_size):
            #     loss = self._compute_one_image_loss(pclasst[i, ...], pbbox_loss[i, ...], npbbox_hw[i, ...], npbbox_y1x1[i, ...],
            #                                         npbbox_y2x2[i, ...], pconft[i, ...], abbox_hw, abbox_y1x1, abbox_y2x2,
            #                                         ground_truth[i, ...])
            #     total_loss.append(loss)
            # total_loss = tf.reduce_mean(total_loss)
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=.9)
            self.loss = .5 * total_loss + self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables('feature_extractor')]
            ) + self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables('regressor')]
            )
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            confidence = pclasst[0, ...] * tf.expand_dims(pconft[0, ...], axis=-1)
            npbbox_y1x1y2x2t = npbbox_y1x1y2x2t[0, ...]
            class_id = tf.argmax(confidence, axis=-1)

            scores = tf.reduce_max(confidence, axis=-1)
            pred_mask = scores >= self.nms_score_threshold
            scores = tf.boolean_mask(scores, pred_mask)
            class_id = tf.boolean_mask(class_id, pred_mask)
            npbbox = tf.boolean_mask(npbbox_y1x1y2x2t, pred_mask)
            selected_index = tf.image.non_max_suppression(
                npbbox, scores, iou_threshold=self.nms_score_threshold, max_output_size=self.nms_max_boxes
            )
            bbox = tf.gather(npbbox, selected_index)
            class_id = tf.gather(class_id, selected_index)
            scores = tf.gather(scores, selected_index)
            self.detection_pred = [scores, bbox, class_id]

            # per class nms
            # filter_mask = tf.greater_equal(confidence, self.nms_score_threshold)
            # scores = []
            # class_id = []
            # bbox = []
            # for i in range(self.num_classes):
            #     scoresi = tf.boolean_mask(scores[:, i], filter_mask[:, i])
            #     bboxi = tf.boolean_mask(npbbox_y1x1y2x2t, filter_mask[:, i])
            #     selected_indices = tf.image.non_max_suppression(
            #
            #         bboxi, scoresi, self.nms_max_boxes, self.nms_iou_threshold,
            #     )
            #     scores.append(tf.gather(scoresi, selected_indices))
            #     bbox.append(tf.gather(bboxi, selected_indices))
            #     class_id.append(tf.ones_like(tf.gather(scoresi, selected_indices), tf.int32) * i)
            # bbox = tf.concat(bbox, axis=0)
            # scores = tf.concat(scores, axis=0)
            # class_id = tf.concat(class_id, axis=0)
            # self.detection_pred = [scores, bbox, class_id]

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

    def _feature_extractor(self, image):
        conv1 = self._conv_layer(image, 32, 3, 1, 'conv1')
        lrelu1 = tf.nn.leaky_relu(conv1, 0.1, 'lrelu1')
        pool1 = self._max_pooling(lrelu1, 2, 2, 'pool1')

        conv2 = self._conv_layer(pool1, 64, 3, 1, 'conv2')
        lrelu2 = tf.nn.leaky_relu(conv2, 0.1, 'lrelu2')
        pool2 = self._max_pooling(lrelu2, 2, 2, 'pool2')

        conv3 = self._conv_layer(pool2, 128, 3, 1, 'conv3')
        lrelu3 = tf.nn.leaky_relu(conv3, 0.1, 'lrelu3')
        conv4 = self._conv_layer(lrelu3, 64, 1, 1, 'conv4')
        lrelu4 = tf.nn.leaky_relu(conv4, 0.1, 'lrelu4')
        conv5 = self._conv_layer(lrelu4, 128, 3, 1, 'conv5')
        lrelu5 = tf.nn.leaky_relu(conv5, 0.1, 'lrelu5')
        pool3 = self._max_pooling(lrelu5, 2, 2, 'pool3')

        conv6 = self._conv_layer(pool3, 256, 3, 1, 'conv6')
        lrelu6 = tf.nn.leaky_relu(conv6, 0.1, 'lrelu6')
        conv7 = self._conv_layer(lrelu6, 128, 1, 1, 'conv7')
        lrelu7 = tf.nn.leaky_relu(conv7, 0.1, 'lrelu7')
        conv8 = self._conv_layer(lrelu7, 256, 3, 1, 'conv8')
        lrelu8 = tf.nn.leaky_relu(conv8, 0.1, 'lrelu8')
        pool4 = self._max_pooling(lrelu8, 2, 2, 'pool4')

        conv9 = self._conv_layer(pool4, 512, 3, 1, 'conv9')
        lrelu9 = tf.nn.leaky_relu(conv9, 0.1, 'lrelu9')
        conv10 = self._conv_layer(lrelu9, 256, 1, 1, 'conv10')
        lrelu10 = tf.nn.leaky_relu(conv10, 0.1, 'lrelu10')
        conv11 = self._conv_layer(lrelu10, 512, 3, 1, 'conv11')
        lrelu11 = tf.nn.leaky_relu(conv11, 0.1, 'lrelu11')
        conv12 = self._conv_layer(lrelu11, 256, 1, 1, 'conv12')
        lrelu12 = tf.nn.leaky_relu(conv12, 0.1, 'lrelu12')
        conv13 = self._conv_layer(lrelu12, 512, 3, 1, 'conv13')
        lrelu13 = tf.nn.leaky_relu(conv13, 0.1, 'lrelu13')
        pool5 = self._max_pooling(lrelu13, 2, 2, 'pool5')

        conv14 = self._conv_layer(pool5, 1024, 3, 1, 'conv14')
        lrelu14 = tf.nn.leaky_relu(conv14, 0.1, 'lrelu14')
        conv15 = self._conv_layer(lrelu14, 512, 1, 1, 'conv15')
        lrelu15 = tf.nn.leaky_relu(conv15, 0.1, 'lrelu15')
        conv16 = self._conv_layer(lrelu15, 1024, 3, 1, 'conv16')
        lrelu16 = tf.nn.leaky_relu(conv16, 0.1, 'lrelu16')
        conv17 = self._conv_layer(lrelu16, 512, 1, 1, 'conv17')
        lrelu17 = tf.nn.leaky_relu(conv17, 0.1, 'lrelu17')
        conv18 = self._conv_layer(lrelu17, 1024, 3, 1, 'conv18')
        lrelu18 = tf.nn.leaky_relu(conv18, 0.1, 'lrelu18')
        downsampling_rate = 32.0
        return lrelu18, lrelu17, downsampling_rate

    def _compute_one_image_loss(self, pclass, npbbox_loss, npbbox_hw, npbbox_y1x1, npbbox_y2x2, pconf,
                                abbox_hw, abbox_y1x1, abbox_y2x2, nground_truth):
        slice_index = tf.argmin(nground_truth, axis=0)[0]
        nground_truth = tf.gather(nground_truth, tf.range(0, slice_index, dtype=tf.int64))
        ngbbox_yx = nground_truth[..., 0:2]
        ngbbox_hw = nground_truth[..., 2:4]
        gclass_id = tf.cast(nground_truth[..., 4], tf.int32)
        gbbox_yx_loss = ngbbox_yx - tf.floor(ngbbox_yx)
        gbbox_hw_loss = ngbbox_hw
        ngbbox_yx = tf.reshape(ngbbox_yx, [-1, 1, 2])
        ngbbox_hw = tf.reshape(ngbbox_hw, [-1, 1, 2])
        gshape = tf.shape(ngbbox_yx)
        ngbbox_y1x1 = ngbbox_yx - ngbbox_hw / 2
        ngbbox_y2x2 = ngbbox_yx + ngbbox_hw / 2

        npbbox_hwti = tf.reshape(npbbox_hw, [1, -1, 2])
        npbbox_y1x1ti = tf.reshape(npbbox_y1x1, [1, -1, 2])
        npbbox_y2x2ti = tf.reshape(npbbox_y2x2, [1, -1, 2])
        pshape = tf.shape(npbbox_hwti)
        npbbox_hwti = tf.tile(npbbox_hwti, [gshape[0], 1, 1])
        npbbox_y1x1ti = tf.tile(npbbox_y1x1ti, [gshape[0], 1, 1])
        npbbox_y2x2ti = tf.tile(npbbox_y2x2ti, [gshape[0], 1, 1])
        ngbbox_hwti = tf.tile(ngbbox_hw, [1, pshape[1], 1])
        ngbbox_y1x1ti = tf.tile(ngbbox_y1x1, [1, pshape[1], 1])
        ngbbox_y2x2ti = tf.tile(ngbbox_y2x2, [1, pshape[1], 1])
        abbox_hwti = tf.tile(abbox_hw, [gshape[0], 1, 1])
        abbox_y1x1ti = tf.tile(abbox_y1x1, [gshape[0], 1, 1])
        abbox_y2x2ti = tf.tile(abbox_y2x2, [gshape[0], 1, 1])

        agiou_y1x1ti = tf.maximum(abbox_y1x1ti, ngbbox_y1x1ti)
        agiou_y2x2ti = tf.minimum(abbox_y2x2ti, ngbbox_y2x2ti)
        agiou_area = tf.reduce_prod(tf.maximum(agiou_y2x2ti - agiou_y1x1ti, 0), axis=-1)
        aarea = tf.reduce_prod(abbox_hwti, axis=-1)
        ngarea = tf.reduce_prod(ngbbox_hwti, axis=-1)
        agiou_rate = agiou_area / (aarea + ngarea - agiou_area)
        rpriors_index = tf.argmax(agiou_rate, axis=-1)
        rpriors = tf.gather(tf.squeeze(abbox_hw), rpriors_index)

        gbbox_hw_loss = tf.log(gbbox_hw_loss / rpriors)
        gbbox_loss = tf.concat([gbbox_yx_loss, gbbox_hw_loss], axis=-1)
        rnpbbox_loss = tf.gather(npbbox_loss, rpriors_index)

        npgiou_y1x1ti = tf.maximum(npbbox_y1x1ti, ngbbox_y1x1ti)
        npgiou_y2x2ti = tf.minimum(npbbox_y2x2ti, ngbbox_y2x2ti)
        npgiou_area = tf.reduce_prod(tf.maximum(npgiou_y2x2ti - npgiou_y1x1ti, 0), axis=-1)
        nparea = tf.reduce_prod(npbbox_hwti, axis=-1)
        ngarea = tf.reduce_prod(ngbbox_hwti, axis=-1)
        npgiou_rate = npgiou_area / (nparea + ngarea - npgiou_area)

        rnpgiou_index = tf.concat([tf.expand_dims(tf.range(0, gshape[0]), -1),
                                   tf.expand_dims(tf.cast(rpriors_index, tf.int32), -1)], axis=-1)
        rnpgiou_rate = tf.gather_nd(npgiou_rate, rnpgiou_index)
        rpconf = tf.gather(pconf, rpriors_index)
        rpclass = tf.gather(pclass, rpriors_index)

        nobj_mask = tf.reduce_min(npgiou_rate, axis=0)
        nobj_mask = tf.cast(nobj_mask <= 0.6, tf.float32)
        detectmask, _ = tf.unique(rpriors_index)
        detectmask = tf.contrib.framework.sort(detectmask)
        detectmask = tf.reshape(detectmask, [-1, 1])
        detectmask = tf.sparse.SparseTensor(tf.concat([detectmask, tf.zeros_like(detectmask)], axis=-1),
                                            tf.squeeze(tf.ones_like(detectmask)), dense_shape=[pshape[1], 1])
        detectmask = tf.reshape(tf.cast(tf.sparse.to_dense(detectmask), tf.float32), [-1])
        noobj_loss = self.noobj_scale * tf.reduce_sum((1. - detectmask) * nobj_mask * tf.square(pconf))
        if self.rescore_confidence:
            obj_loss = self.obj_scale * tf.reduce_sum(tf.square(rnpgiou_rate - rpconf))
        else:
            obj_loss = self.obj_scale * tf.reduce_sum(tf.square(1. - rpconf))
        obj_loss = noobj_loss + obj_loss

        coord_loss = self.coord_sacle * tf.reduce_sum(tf.square(gbbox_loss - rnpbbox_loss))

        gclass = tf.one_hot(gclass_id, self.num_classes)
        class_loss = self.class_scale * tf.reduce_sum(tf.square(gclass - rpclass))
        loss = obj_loss + coord_loss + class_loss
        return loss

    def _train_pretraining_epoch(self, lr, writer=None, data_provider=None):
        self.is_training = True
        if data_provider is not None:
            self.num_train = data_provider['num_train']
            self.num_val = data_provider['num_val']
            self.train_generator = data_provider['train_generator']
            self.train_initializer, self.train_iterator = self.train_generator
            self.data_shape = data_provider['data_shape']
            shape = [self.batch_size].extend(data_provider['data_shape'])
            self.images.set_shape(shape)
        self.sess.run(self.train_initializer)
        mean_loss = []
        mean_acc = []
        for i in range(self.num_train // self.batch_size):
            _, loss, acc, summaries = self.sess.run([self.train_op, self.loss, self.accuracy, self.summary_op],
                                                    feed_dict={self.lr: lr})
            mean_loss.append(loss)
            mean_acc.append(acc)
            if writer is not None:
                writer.add_summary(summaries, global_step=self.global_step)
        mean_loss = np.mean(mean_loss)
        mean_acc = np.mean(mean_acc)
        return mean_loss, mean_acc

    def _train_detection_epoch(self, lr, writer=None, data_provider=None):
        self.is_training = True
        if data_provider is not None:
            self.num_train = data_provider['num_train']
            self.num_val = data_provider['num_val']
            self.train_generator = data_provider['train_generator']
            self.train_initializer, self.train_iterator = self.train_generator
            self.data_shape = data_provider['data_shape']
            shape = [self.batch_size].extend(data_provider['data_shape'])
            self.images.set_shape(shape)
        self.sess.run(self.train_initializer)
        mean_loss = []
        num_iters = self.num_train // self.batch_size
        for i in range(num_iters):
            _, loss, summaries = self.sess.run([self.train_op, self.loss, self.summary_op],
                                               feed_dict={self.lr: lr})
            sys.stdout.write('\r>> ' + 'iters '+str(i+1)+str('/')+str(num_iters)+' loss '+str(loss))
            sys.stdout.flush()
            mean_loss.append(loss)
            if writer is not None:
                writer.add_summary(summaries, global_step=self.global_step)
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

    def _bn(self, bottom):
        bn = tf.layers.batch_normalization(
            inputs=bottom,
            axis=3 if self.data_format == 'channels_last' else 1,
            training=self.is_training
        )
        return bn

    def _conv_layer(self, bottom, filters, kernel_size, strides, name):
        conv = tf.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name=name,
            data_format=self.data_format,
        )
        bn = self._bn(conv)
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
