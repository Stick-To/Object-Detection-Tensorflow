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
        self.priors = tf.reshape(priors, [1, 1, self.num_priors, 2])
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
            features, passthrough, downsampling_rate = self._feature_extractor(self.images)
        with tf.variable_scope('head'):
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

            pred = tf.reshape(pred, [pshape[0], pshape[1], pshape[2], self.num_priors, -1])
            pclass = pred[..., :self.num_classes]
            pbbox_yx = pred[..., self.num_classes:self.num_classes + 2]
            pbbox_hw = pred[..., self.num_classes + 2:self.num_classes + 4]
            pobj = pred[..., self.num_classes + 4:]
            abbox_yx, abbox_hw, abbox_y1x1, abbox_y2x2 = self._get_priors(pshape, self.priors)

        if self.mode == 'train':
            total_loss = []
            for i in range(self.batch_size):
                gn_yxi, gn_hwi, gn_labeli = self._get_normlized_gn(downsampling_rate, i)
                gn_floor_ = tf.cast(tf.floor(gn_yxi), tf.int64)
                nogn_mask = tf.sparse.SparseTensor(gn_floor_, tf.ones_like(gn_floor_[..., 0]), dense_shape=[pshape[1], pshape[2]])
                nogn_mask = (1 - tf.sparse.to_dense(nogn_mask, validate_indices=False)) > 0
                rpbbox_yx = tf.gather_nd(pbbox_yx[i, ...], gn_floor_)
                rpbbox_hw = tf.gather_nd(pbbox_hw[i, ...], gn_floor_)
                rabbox_hw = tf.gather_nd(abbox_hw, gn_floor_)
                rpclass = tf.gather_nd(pclass[i, ...], gn_floor_)
                rpobj = tf.gather_nd(pobj[i, ...], gn_floor_)
                rabbox_y1x1 = tf.gather_nd(abbox_y1x1, gn_floor_)
                rabbox_y2x2 = tf.gather_nd(abbox_y2x2, gn_floor_)
                gn_y1x1i = tf.expand_dims(gn_yxi - gn_hwi / 2., axis=1)
                gn_y2x2i = tf.expand_dims(gn_yxi + gn_hwi / 2., axis=1)
                rgaiou_y1x1 = tf.maximum(gn_y1x1i, rabbox_y1x1)
                rgaiou_y2x2 = tf.minimum(gn_y2x2i, rabbox_y2x2)

                rgaiou_area = tf.reduce_prod(rgaiou_y2x2 - rgaiou_y1x1, axis=-1)

                garea = tf.reduce_prod(gn_y2x2i - gn_y1x1i, axis=-1)
                aarea = tf.reduce_prod(rabbox_hw, axis=-1)
                rgaiou = rgaiou_area / (aarea + garea - rgaiou_area)

                rgaindex = tf.expand_dims(tf.cast(tf.argmax(rgaiou, axis=-1), tf.int32), -1)
                rgaindex = tf.concat([tf.expand_dims(tf.range(tf.shape(rgaindex)[0]), -1), rgaindex], axis=-1)

                rpbbox_yx = tf.reshape(tf.gather_nd(rpbbox_yx, rgaindex), [-1, 2])
                rpbbox_hw = tf.reshape(tf.gather_nd(rpbbox_hw, rgaindex), [-1, 2])

                rpclass = tf.reshape(tf.gather_nd(rpclass, rgaindex), [-1, self.num_classes])
                rpobj = tf.reshape(tf.gather_nd(rpobj, rgaindex), [-1, 1])
                rabbox_hw = tf.reshape(tf.gather_nd(rabbox_hw, rgaindex), [-1, 2])

                gn_labeli = tf.one_hot(gn_labeli, self.num_classes)
                rpbbox_yx_target = gn_yxi - tf.floor(gn_yxi)
                rpbbox_hw_target = tf.log(gn_hwi / rabbox_hw)
                yx_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=rpbbox_yx_target, logits=rpbbox_yx))
                hw_loss = 0.5 * tf.reduce_sum(tf.square(rpbbox_hw - rpbbox_hw_target))
                coord_loss = yx_loss + hw_loss
                class_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=gn_labeli, logits=rpclass))
                obj_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(rpobj), logits=rpobj))
                nogn_mask = tf.reshape(nogn_mask, [-1])

                abbox_yx_nobest = tf.expand_dims(tf.boolean_mask(tf.reshape(abbox_yx-abbox_hw/2., [-1, self.num_priors, 2]), nogn_mask), 1)
                abbox_hw_nobest = tf.expand_dims(tf.boolean_mask(tf.reshape(abbox_yx+abbox_hw/2., [-1, self.num_priors, 2]), nogn_mask), 1)
                abbox_y1x1_nobest = abbox_yx_nobest - abbox_hw_nobest/2.
                abbox_y2x2_nobest = abbox_yx_nobest + abbox_hw_nobest/2.
                pobj_nobest = tf.boolean_mask(tf.reshape(pobj[i, ...], [-1, self.num_priors]), nogn_mask)


                num_g = tf.shape(gn_y1x1i)[0]
                num_p = tf.shape(abbox_y1x1_nobest)[0]
                gn_y1x1i = tf.tile(tf.expand_dims(gn_y1x1i, 0), [num_p, 1, 1, 1])
                gn_y2x2i = tf.tile(tf.expand_dims(gn_y2x2i, 0), [num_p, 1, 1, 1])

                abbox_y1x1_nobest = tf.tile(abbox_y1x1_nobest, [1, num_g, 1, 1])
                abbox_y2x2_nobest = tf.tile(abbox_y2x2_nobest, [1, num_g, 1, 1])
                agiou_y1x1 = tf.maximum(gn_y1x1i, abbox_y1x1_nobest)
                agiou_y2x2 = tf.minimum(gn_y2x2i, abbox_y2x2_nobest)

                agiou_area = tf.reduce_prod(agiou_y2x2 - agiou_y1x1, axis=-1)
                aarea = tf.reduce_prod(abbox_y2x2_nobest - abbox_y1x1_nobest, axis=-1)
                garea = tf.reduce_prod(gn_y2x2i - gn_y1x1i, axis=-1)
                agiou = agiou_area / (aarea + garea - agiou_area)
                agiou = tf.reduce_max(agiou, axis=1)

                noobj_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(pobj_nobest), logits=pobj_nobest)*tf.cast(agiou <= 0.6, tf.float32))
                loss = self.coord_sacle * coord_loss + self.class_scale * class_loss + self.obj_scale * obj_loss + self.noobj_scale * noobj_loss
                total_loss.append(loss)
            total_loss = tf.reduce_mean(total_loss)
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
            self.loss = total_loss + self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables()]
            )
            train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.group([update_ops, train_op])
        else:
            pclasst = tf.sigmoid(tf.reshape(pclass[0, ...], [-1, self.num_classes]))
            pobjt = tf.sigmoid(tf.reshape(pobj[0, ...], [-1, 1]))
            pbbox_yx = tf.reshape(pbbox_yx[0, ...], [-1, 2])
            pbbox_hw = tf.reshape(pbbox_hw[0, ...], [-1, 2])
            abbox_yx = tf.reshape(abbox_yx, [-1, 2])
            abbox_hw = tf.reshape(abbox_hw, [-1, 2])
            bbox_yx = abbox_yx + tf.sigmoid(pbbox_yx)
            bbox_hw = abbox_hw + tf.exp(pbbox_hw)
            bbox_y1x1y2x2 = tf.concat([bbox_yx - bbox_hw / 2., bbox_yx + bbox_hw / 2.], axis=-1) * downsampling_rate
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
            if self.train_initializer is not None:
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

    def train_one_epoch(self, lr, writer=None):
        self.is_training = True
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
