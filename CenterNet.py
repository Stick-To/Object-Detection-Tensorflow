from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import os


class CenterNet:
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
        self.num_classes = config['num_classes']
        self.weight_decay = config['weight_decay']
        self.prob = 1. - config['keep_prob']
        self.data_format = config['data_format']
        self.mode = config['mode']
        self.batch_size = config['batch_size'] if config['mode'] == 'train' else 1

        if self.mode == 'train':
            self.num_train = data_provider['num_train']
            self.num_val = data_provider['num_val']
            self.train_generator = data_provider['train_generator']
            self.train_initializer, self.train_iterator = self.train_generator
            if data_provider['val_generator'] is not None:
                self.val_generator = data_provider['val_generator']
                self.val_initializer, self.val_iterator = self.val_generator
        else:
            self.score_threshold = config['score_threshold']
            self.top_k_results_output = config['top_k_results_output']

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
        mean = tf.convert_to_tensor([0.485, 0.456, 0.406], dtype=tf.float32)
        std = tf.convert_to_tensor([0.229, 0.224, 0.225], dtype=tf.float32)
        if self.data_format == 'channels_last':
            mean = tf.reshape(mean, [1, 1, 1, 3])
            std = tf.reshape(std, [1, 1, 1, 3])
        else:
            mean = tf.reshape(mean, [1, 3, 1, 1])
            std = tf.reshape(std, [1, 3, 1, 1])
        if self.mode == 'train':
            self.images, self.ground_truth = self.train_iterator.get_next()
            self.images.set_shape(shape)
            self.images = (self.images / 255. - mean) / std
        else:
            self.images = tf.placeholder(tf.float32, shape, name='images')
            self.images = (self.images / 255. - mean) / std
            self.ground_truth = tf.placeholder(tf.float32, [self.batch_size, None, 5], name='labels')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

    def _build_graph(self):
        with tf.variable_scope('backone'):
            conv = self._conv_bn_activation(
                bottom=self.images,
                filters=16,
                kernel_size=7,
                strides=1,
            )
            conv = self._conv_bn_activation(
                bottom=conv,
                filters=16,
                kernel_size=3,
                strides=1,
            )
            conv = self._conv_bn_activation(
                bottom=conv,
                filters=32,
                kernel_size=3,
                strides=2,
            )
            dla_stage3 = self._dla_generator(conv, 64, 1, self._basic_block)
            dla_stage3 = self._max_pooling(dla_stage3, 2, 2)

            dla_stage4 = self._dla_generator(dla_stage3, 128, 2, self._basic_block)
            residual = self._conv_bn_activation(dla_stage3, 128, 1, 1)
            residual = self._avg_pooling(residual, 2, 2)
            dla_stage4 = self._max_pooling(dla_stage4, 2, 2)
            dla_stage4 = dla_stage4 + residual

            dla_stage5 = self._dla_generator(dla_stage4, 256, 2, self._basic_block)
            residual = self._conv_bn_activation(dla_stage4, 256, 1, 1)
            residual = self._avg_pooling(residual, 2, 2)
            dla_stage5 = self._max_pooling(dla_stage5, 2, 2)
            dla_stage5 = dla_stage5 + residual

            dla_stage6 = self._dla_generator(dla_stage5, 512, 1, self._basic_block)
            residual = self._conv_bn_activation(dla_stage5, 512, 1, 1)
            residual = self._avg_pooling(residual, 2, 2)
            dla_stage6 = self._max_pooling(dla_stage6, 2, 2)
            dla_stage6 = dla_stage6 + residual
        with tf.variable_scope('upsampling'):
            dla_stage6 = self._conv_bn_activation(dla_stage6, 256, 1, 1)
            dla_stage6_5 = self._dconv_bn_activation(dla_stage6, 256, 4, 2)
            dla_stage6_4 = self._dconv_bn_activation(dla_stage6_5, 256, 4, 2)
            dla_stage6_3 = self._dconv_bn_activation(dla_stage6_4, 256, 4, 2)

            dla_stage5 = self._conv_bn_activation(dla_stage5, 256, 1, 1)
            dla_stage5_4 = self._conv_bn_activation(dla_stage5+dla_stage6_5, 256, 3, 1)
            dla_stage5_4 = self._dconv_bn_activation(dla_stage5_4, 256, 4, 2)
            dla_stage5_3 = self._dconv_bn_activation(dla_stage5_4, 256, 4, 2)

            dla_stage4 = self._conv_bn_activation(dla_stage4, 256, 1, 1)
            dla_stage4_3 = self._conv_bn_activation(dla_stage4+dla_stage5_4+dla_stage6_4, 256, 3, 1)
            dla_stage4_3 = self._dconv_bn_activation(dla_stage4_3, 256, 4, 2)

            features = self._conv_bn_activation(dla_stage6_3+dla_stage5_3+dla_stage4_3, 256, 3, 1)
            features = self._conv_bn_activation(features, 256, 1, 1)
            stride = 4.0

        with tf.variable_scope('center_detector'):
            keypoints = self._conv_bn_activation(features, self.num_classes, 3, 1, None)
            offset = self._conv_bn_activation(features, 2, 3, 1, None)
            size = self._conv_bn_activation(features, 2, 3, 1, None)
            if self.data_format == 'channels_first':
                keypoints = tf.transpose(keypoints, [0, 2, 3, 1])
                offset = tf.transpose(offset, [0, 2, 3, 1])
                size = tf.transpose(size, [0, 2, 3, 1])
            pshape = [tf.shape(offset)[1], tf.shape(offset)[2]]

            h = tf.range(0., tf.cast(pshape[0], tf.float32), dtype=tf.float32)
            w = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)
            [meshgrid_x, meshgrid_y] = tf.meshgrid(w, h)
            if self.mode == 'train':
                total_loss = []
                for i in range(self.batch_size):
                    loss = self._compute_one_image_loss(keypoints[i, ...], offset[i, ...], size[i, ...],
                                                        self.ground_truth[i, ...], meshgrid_y, meshgrid_x,
                                                        stride, pshape)
                    total_loss.append(loss)

                self.loss = tf.reduce_mean(total_loss) + self.weight_decay * tf.add_n(
                    [tf.nn.l2_loss(var) for var in tf.trainable_variables()])
                optimizer = tf.train.AdamOptimizer(self.lr)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                train_op = optimizer.minimize(self.loss, global_step=self.global_step)
                self.train_op = tf.group([update_ops, train_op])
            else:
                keypoints = tf.sigmoid(keypoints)
                meshgrid_y = tf.expand_dims(meshgrid_y, axis=-1)
                meshgrid_x = tf.expand_dims(meshgrid_x, axis=-1)
                center = tf.concat([meshgrid_y, meshgrid_x], axis=-1)
                category = tf.expand_dims(tf.squeeze(tf.argmax(keypoints, axis=-1, output_type=tf.int32)), axis=-1)
                meshgrid_xyz = tf.concat([tf.zeros_like(category), tf.cast(center, tf.int32), category], axis=-1)
                keypoints = tf.gather_nd(keypoints, meshgrid_xyz)
                keypoints = tf.expand_dims(keypoints, axis=0)
                keypoints = tf.expand_dims(keypoints, axis=-1)
                keypoints_peak = self._max_pooling(keypoints, 3, 1)
                keypoints_mask = tf.cast(tf.equal(keypoints, keypoints_peak), tf.float32)
                keypoints = keypoints * keypoints_mask
                scores = tf.reshape(keypoints, [-1])
                class_id = tf.reshape(category, [-1])
                bbox_yx = tf.reshape(center+offset, [-1, 2])
                bbox_hw = tf.reshape(size, [-1, 2])
                score_mask = scores > self.score_threshold
                scores = tf.boolean_mask(scores, score_mask)
                class_id = tf.boolean_mask(class_id, score_mask)
                bbox_yx = tf.boolean_mask(bbox_yx, score_mask)
                bbox_hw = tf.boolean_mask(bbox_hw, score_mask)
                bbox = tf.concat([bbox_yx-bbox_hw/2., bbox_yx+bbox_hw/2.], axis=-1) * stride
                num_select = tf.cond(tf.shape(scores)[0] > self.top_k_results_output, lambda: self.top_k_results_output, lambda: tf.shape(scores)[0])
                select_scores, select_indices = tf.nn.top_k(scores, num_select)
                select_class_id = tf.gather(class_id, select_indices)
                select_bbox = tf.gather(bbox, select_indices)
                self.detection_pred = [select_scores, select_bbox, select_class_id]

    def _compute_one_image_loss(self, keypoints, offset, size, ground_truth, meshgrid_y, meshgrid_x,
                                stride, pshape):
        slice_index = tf.argmin(ground_truth, axis=0)[0]
        ground_truth = tf.gather(ground_truth, tf.range(0, slice_index, dtype=tf.int64))
        ngbbox_y = ground_truth[..., 0] / stride
        ngbbox_x = ground_truth[..., 1] / stride
        ngbbox_h = ground_truth[..., 2] / stride
        ngbbox_w = ground_truth[..., 3] / stride
        class_id = tf.cast(ground_truth[..., 4], dtype=tf.int32)
        ngbbox_yx = ground_truth[..., 0:2] / stride
        ngbbox_yx_round = tf.floor(ngbbox_yx)
        offset_gt = ngbbox_yx - ngbbox_yx_round
        size_gt = ground_truth[..., 2:4] / stride
        ngbbox_yx_round_int = tf.cast(ngbbox_yx_round, tf.int64)
        keypoints_loss = self._keypoints_loss(keypoints, ngbbox_yx_round_int, ngbbox_y, ngbbox_x, ngbbox_h,
                                              ngbbox_w, class_id, meshgrid_y, meshgrid_x, pshape)

        offset = tf.gather_nd(offset, ngbbox_yx_round_int)
        size = tf.gather_nd(size, ngbbox_yx_round_int)
        offset_loss = tf.reduce_mean(tf.abs(offset_gt - offset))
        size_loss = tf.reduce_mean(tf.abs(size_gt - size))
        total_loss = keypoints_loss + 0.1*size_loss + offset_loss
        return total_loss

    def _keypoints_loss(self, keypoints, gbbox_yx, gbbox_y, gbbox_x, gbbox_h, gbbox_w,
                        classid, meshgrid_y, meshgrid_x, pshape):
        sigma = self._gaussian_radius(gbbox_h, gbbox_w, 0.7)
        gbbox_y = tf.reshape(gbbox_y, [-1, 1, 1])
        gbbox_x = tf.reshape(gbbox_x, [-1, 1, 1])
        sigma = tf.reshape(sigma, [-1, 1, 1])

        num_g = tf.shape(gbbox_y)[0]
        meshgrid_y = tf.expand_dims(meshgrid_y, 0)
        meshgrid_y = tf.tile(meshgrid_y, [num_g, 1, 1])
        meshgrid_x = tf.expand_dims(meshgrid_x, 0)
        meshgrid_x = tf.tile(meshgrid_x, [num_g, 1, 1])

        keyp_penalty_reduce = tf.exp(-((gbbox_y-meshgrid_y)**2 + (gbbox_x-meshgrid_x)**2)/(2*sigma**2))
        zero_like_keyp = tf.expand_dims(tf.zeros(pshape, dtype=tf.float32), axis=-1)
        reduction = []
        gt_keypoints = []
        for i in range(self.num_classes):
            exist_i = tf.equal(classid, i)
            reduce_i = tf.boolean_mask(keyp_penalty_reduce, exist_i, axis=0)
            reduce_i = tf.cond(
                tf.equal(tf.shape(reduce_i)[0], 0),
                lambda: zero_like_keyp,
                lambda: tf.expand_dims(tf.reduce_max(reduce_i, axis=0), axis=-1)
            )
            reduction.append(reduce_i)

            gbbox_yx_i = tf.boolean_mask(gbbox_yx, exist_i)
            gt_keypoints_i = tf.cond(
                tf.equal(tf.shape(gbbox_yx_i)[0], 0),
                lambda: zero_like_keyp,
                lambda: tf.expand_dims(tf.sparse.to_dense(tf.sparse.SparseTensor(gbbox_yx_i, tf.ones_like(gbbox_yx_i[..., 0], tf.float32), dense_shape=pshape), validate_indices=False),
                                       axis=-1)
            )
            gt_keypoints.append(gt_keypoints_i)
        reduction = tf.concat(reduction, axis=-1)
        gt_keypoints = tf.concat(gt_keypoints, axis=-1)
        keypoints_pos_loss = -tf.pow(1.-tf.sigmoid(keypoints), 2.) * tf.log_sigmoid(keypoints) * gt_keypoints
        keypoints_neg_loss = -tf.pow(1.-reduction, 4) * tf.pow(tf.sigmoid(keypoints), 2.) * (-keypoints+tf.log_sigmoid(keypoints)) * (1.-gt_keypoints)
        keypoints_loss = tf.reduce_sum(keypoints_pos_loss) / tf.cast(num_g, tf.float32) + tf.reduce_sum(keypoints_neg_loss) / tf.cast(num_g, tf.float32)
        return keypoints_loss

    # from cornernet
    def _gaussian_radius(self, height, width, min_overlap=0.7):
        a1 = 1.
        b1 = (height + width)
        c1 = width * height * (1. - min_overlap) / (1. + min_overlap)
        sq1 = tf.sqrt(b1 ** 2. - 4. * a1 * c1)
        r1 = (b1 + sq1) / 2.
        a2 = 4.
        b2 = 2. * (height + width)
        c2 = (1. - min_overlap) * width * height
        sq2 = tf.sqrt(b2 ** 2. - 4. * a2 * c2)
        r2 = (b2 + sq2) / 2.
        a3 = 4. * min_overlap
        b3 = -2. * min_overlap * (height + width)
        c3 = (min_overlap - 1.) * width * height
        sq3 = tf.sqrt(b3 ** 2. - 4. * a3 * c3)
        r3 = (b3 + sq3) / 2.
        return tf.reduce_min([r1, r2, r3])

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
        self.sess.run(self.train_initializer)
        mean_loss = []
        num_iters = self.num_train // self.batch_size
        for i in range(num_iters):
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.lr: lr, self.is_training:True})
            sys.stdout.write('\r>> ' + 'iters '+str(i+1)+str('/')+str(num_iters)+' loss '+str(loss))
            sys.stdout.flush()
            mean_loss.append(loss)
        sys.stdout.write('\n')
        mean_loss = np.mean(mean_loss)
        return mean_loss

    def test_one_image(self, images):
        pred = self.sess.run(self.detection_pred, feed_dict={self.images: images, self.is_training:False})
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
            data_format=self.data_format
        )
        bn = self._bn(conv)
        if activation is not None:
            return activation(bn)
        else:
            return bn

    def _dconv_bn_activation(self, bottom, filters, kernel_size, strides, activation=tf.nn.relu):
        conv = tf.layers.conv2d_transpose(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
        )
        bn = self._bn(conv)
        if activation is not None:
            bn = activation(bn)
        return bn

    def _separable_conv_layer(self, bottom, filters, kernel_size, strides, activation=tf.nn.relu):
        conv = tf.layers.separable_conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            use_bias=False,
        )
        bn = self._bn(conv)
        if activation is not None:
            bn = activation(bn)
        return bn

    def _basic_block(self, bottom, filters):
        conv = self._conv_bn_activation(bottom, filters, 3, 1)
        conv = self._conv_bn_activation(conv, filters, 3, 1)
        axis = 3 if self.data_format == 'channels_last' else 1
        input_channels = tf.shape(bottom)[axis]
        shutcut = tf.cond(
            tf.equal(input_channels, filters),
            lambda: bottom,
            lambda: self._conv_bn_activation(bottom, filters, 1, 1)
        )
        return conv + shutcut

    def _dla_generator(self, bottom, filters, levels, stack_block_fn):
        if levels == 1:
            block1 = stack_block_fn(bottom, filters)
            block2 = stack_block_fn(block1, filters)
            aggregation = block1 + block2
            aggregation = self._conv_bn_activation(aggregation, filters, 3, 1)
        else:
            block1 = self._dla_generator(bottom, filters, levels-1, stack_block_fn)
            block2 = self._dla_generator(block1, filters, levels-1, stack_block_fn)
            aggregation = block1 + block2
            aggregation = self._conv_bn_activation(aggregation, filters, 3, 1)
        return aggregation

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
