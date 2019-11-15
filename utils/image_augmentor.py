from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def image_augmentor(image, input_shape, data_format, output_shape, zoom_size=None,
                    crop_method=None, flip_prob=None, fill_mode='BILINEAR', keep_aspect_ratios=False,
                    constant_values=0., color_jitter_prob=None, rotate=None, ground_truth=None, pad_truth_to=None):

    """
    :param image: HWC or CHW
    :param input_shape: [h, w]
    :param data_format: 'channels_first', 'channels_last'
    :param output_shape: [h, w]
    :param zoom_size: [h, w]
    :param crop_method: 'random', 'center'
    :param flip_prob: [flip_top_down_prob, flip_left_right_prob]
    :param fill_mode: 'CONSTANT', 'NEAREST_NEIGHBOR', 'BILINEAR', 'BICUBIC'
    :param keep_aspect_ratios: True, False
    :param constant_values:
    :param color_jitter_prob: prob of color_jitter
    :param rotate: [prob, min_angle, max_angle]
    :param ground_truth: [ymin, ymax, xmin, xmax, classid]
    :param pad_truth_to: pad ground_truth to size [pad_truth_to, 5] with -1
    :return image: output_shape
    :return ground_truth: [pad_truth_to, 5] [ycenter, xcenter, h, w, class_id]
    """
    if data_format not in ['channels_first', 'channels_last']:
        raise Exception("data_format must in ['channels_first', 'channels_last']!")
    if fill_mode not in ['CONSTANT', 'NEAREST_NEIGHBOR', 'BILINEAR', 'BICUBIC']:
        raise Exception("fill_mode must in ['CONSTANT', 'NEAREST_NEIGHBOR', 'BILINEAR', 'BICUBIC']!")
    if fill_mode == 'CONSTANT' and zoom_size is not None:
        raise Exception("if fill_mode is 'CONSTANT', zoom_size can't be None!")
    if zoom_size is not None:
        if keep_aspect_ratios:
            if constant_values is None:
                raise Exception('please provide constant_values!')
        if not zoom_size[0] >= output_shape[0] and zoom_size[1] >= output_shape[1]:
            raise Exception("output_shape can't greater that zoom_size!")
        if crop_method not in ['random', 'center']:
            raise Exception("crop_method must in ['random', 'center']!")
        if fill_mode is 'CONSTANT' and constant_values is None:
            raise Exception("please provide constant_values!")
    if color_jitter_prob is not None:
        if not 0. <= color_jitter_prob <= 1.:
            raise Exception("color_jitter_prob can't less that 0.0, and can't grater that 1.0")
    if flip_prob is not None:
        if not 0. <= flip_prob[0] <= 1. and 0. <= flip_prob[1] <= 1.:
            raise Exception("flip_prob can't less than 0.0, and can't grater than 1.0")
    if rotate is not None:
        if len(rotate) != 3:
            raise Exception('please provide "rotate" parameter as [rotate_prob, min_angle, max_angle]!')
        if not 0. <= rotate[0] <= 1.:
            raise Exception("rotate prob can't less that 0.0, and can't grater that 1.0")
        if ground_truth is not None:
            if not -5. <= rotate[1] <= 5. and -5. <= rotate[2] <= 5.:
                raise Exception('rotate range must be -5 to 5, otherwise coordinate mapping become imprecise!')
        if not rotate[1] <= rotate[2]:
            raise Exception("rotate[1] can't  grater than rotate[2]")

    image_copy = image

    input_h, input_w, input_c = input_shape[0], input_shape[1], input_shape[2]
    input_h_f, input_w_f, input_c_f = tf.cast(input_h, tf.float32), tf.cast(input_w, tf.float32), \
                                      tf.cast(input_c, tf.float32)
    output_h_f, output_w_f = float(output_shape[0]), float(output_shape[1])

    if fill_mode == 'CONSTANT':
        keep_aspect_ratios = True
    fill_mode_project = {
        'NEAREST_NEIGHBOR': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        'BILINEAR': tf.image.ResizeMethod.BILINEAR,
        'BICUBIC': tf.image.ResizeMethod.BICUBIC
    }
    if ground_truth is not None:
        ymin = ground_truth[:, 0:1]
        ymax = ground_truth[:, 1:2]
        xmin = ground_truth[:, 2:3]
        xmax = ground_truth[:, 3:4]
        class_id = ground_truth[:, 4:5]
        ground_truth_copy = tf.concat([ymin/2.+ymax/2., xmin/2.+xmax/2., ymax-ymin, xmax-xmin, class_id], axis=-1)

    if data_format == 'channels_first':
        image = tf.transpose(image, [1, 2, 0])
    input_h, input_w, input_c = input_shape[0], input_shape[1], input_shape[2]
    output_h, output_w = output_shape
    if zoom_size is not None:
        zoom_or_output_h, zoom_or_output_w = zoom_size
    else:
        zoom_or_output_h, zoom_or_output_w = output_shape
    if keep_aspect_ratios:
        if fill_mode in ['NEAREST_NEIGHBOR', 'BILINEAR', 'BICUBIC']:
            zoom_ratio = tf.cond(
                tf.less(zoom_or_output_h / input_h, zoom_or_output_w / input_w),
                lambda: tf.cast(zoom_or_output_h / input_h, tf.float32),
                lambda: tf.cast(zoom_or_output_w / input_w, tf.float32)
            )
            resize_h, resize_w = tf.cond(
                tf.less(zoom_or_output_h / input_h, zoom_or_output_w / input_w),
                lambda: (zoom_or_output_h,tf.cast(tf.cast(input_w, tf.float32) * zoom_ratio, tf.int32)),
                lambda: (tf.cast(tf.cast(input_h, tf.float32)*zoom_ratio, tf.int32), zoom_or_output_w)
            )
            image = tf.image.resize_images(
                image, [resize_h, resize_w], fill_mode_project[fill_mode],
                align_corners=True,
            )
            if ground_truth is not None:
                ymin, ymax = ymin * zoom_ratio, ymax * zoom_ratio
                xmin, xmax = xmin * zoom_ratio, xmax * zoom_ratio
            image = tf.pad(
                image, [[0, zoom_or_output_h-resize_h], [0, zoom_or_output_w-resize_w], [0, 0]],
                mode='CONSTANT', constant_values=constant_values
            )
        else:
            image = tf.pad(
                image, [[0, zoom_or_output_h-input_h], [0, zoom_or_output_w-input_w], [0, 0]],
                mode='CONSTANT', constant_values=constant_values
            )
    else:
        image = tf.image.resize_images(
            image, [zoom_or_output_h, zoom_or_output_w], fill_mode_project[fill_mode],
            align_corners=True, preserve_aspect_ratio=False
        )
        if ground_truth is not None:
            zoom_ratio_y = tf.cast(zoom_or_output_h / input_h, tf.float32)
            zoom_ratio_x = tf.cast(zoom_or_output_w / input_w, tf.float32)
            ymin, ymax = ymin * zoom_ratio_y, ymax * zoom_ratio_y
            xmin, xmax = xmin * zoom_ratio_x, xmax * zoom_ratio_x

    if zoom_size is not None:
        if crop_method == 'random':
            random_h = zoom_or_output_h - output_h
            random_w = zoom_or_output_w - output_w
            crop_h = tf.random_uniform([], 0, random_h, tf.int32)
            crop_w = tf.random_uniform([], 0, random_w, tf.int32)
        else:
            crop_h = (zoom_or_output_h - output_h) // 2
            crop_w = (zoom_or_output_w - output_w) // 2
        image = tf.slice(
            image, [crop_h, crop_w, 0], [output_h, output_w, input_c]
        )
        if ground_truth is not None:
            ymin, ymax = ymin - tf.cast(crop_h, tf.float32), ymax - tf.cast(crop_h, tf.float32)
            xmin, xmax = xmin - tf.cast(crop_w, tf.float32), xmax - tf.cast(crop_w, tf.float32)

    if flip_prob is not None:
        flip_td_prob = tf.random_uniform([], 0., 1.)
        flip_lr_prob = tf.random_uniform([], 0., 1.)
        image = tf.cond(
            tf.less(flip_td_prob, flip_prob[0]),
            lambda: tf.reverse(image, [0]),
            lambda: image
        )
        image = tf.cond(
            tf.less(flip_lr_prob, flip_prob[1]),
            lambda: tf.reverse(image, [1]),
            lambda: image
        )
        if ground_truth is not None:
            ymax, ymin = tf.cond(
                tf.less(flip_td_prob, flip_prob[0]),
                lambda: (output_h - ymin -1., output_h - ymax -1.),
                lambda: (ymax, ymin)
            )
            xmax, xmin = tf.cond(
                tf.less(flip_lr_prob, flip_prob[1]),
                lambda: (output_w - xmin -1., output_w - xmax - 1.),
                lambda: (xmax, xmin)
            )
    if color_jitter_prob is not None:
        bcs = tf.random_uniform([3], 0., 1.)
        image = tf.cond(bcs[0] < color_jitter_prob,
                        lambda: tf.image.adjust_brightness(image, tf.random_uniform([], 0., 0.3)),
                        lambda: image
                )
        image = tf.cond(bcs[1] < color_jitter_prob,
                        lambda: tf.image.adjust_contrast(image, tf.random_uniform([], 0.8, 1.2)),
                        lambda: image
                )
        image = tf.cond(bcs[2] < color_jitter_prob,
                        lambda: tf.image.adjust_hue(image, tf.random_uniform([], -0.1, 0.1)),
                        lambda: image
                )

    if rotate is not None:
        rotate_prob, angmin, angmax = rotate[0], rotate[1], rotate[2]
        rp = tf.random.uniform([], 0., 1.)
        image, ymin, xmin, ymax, xmax = tf.cond(
            rp < rotate_prob,
            lambda: rotate_helper(image, angmin, angmax, ymin, xmin, ymax, xmax, output_h_f, output_w_f),
            lambda: (image, ymin, xmin, ymax, xmax)
        )

    if data_format == 'channels_first':
        image = tf.transpose(image, [2, 0, 1])
    if ground_truth is not None:
        ymin = tf.where(ymin < 0., ymin-ymin, ymin)
        xmin = tf.where(xmin < 0., xmin-xmin, xmin)
        ymax = tf.where(ymax < 0., ymax-ymax, ymax)
        xmax = tf.where(xmax < 0., xmax-xmax, xmax)
        ymin = tf.where(ymin > output_h_f - 1., ymin-ymin+output_h_f - 1., ymin)
        xmin = tf.where(xmin > output_w_f - 1., xmin-xmin+output_w_f - 1., xmin)
        ymax = tf.where(ymax > output_h_f - 1., ymax-ymax+output_h_f - 1., ymax)
        xmax = tf.where(xmax > output_w_f - 1., xmax-xmax+output_w_f - 1., xmax)
        y_center = (ymin + ymax) / 2.
        x_center = (xmin + xmax) / 2.
        y_mask = tf.cast(y_center > 0., tf.float32) * tf.cast(y_center < output_h_f - 1., tf.float32)
        x_mask = tf.cast(x_center > 0., tf.float32) * tf.cast(x_center < output_w_f - 1., tf.float32)
        mask = tf.reshape((x_mask * y_mask) > 0., [-1])
        ymin = tf.boolean_mask(ymin, mask)
        xmin = tf.boolean_mask(xmin, mask)
        ymax = tf.boolean_mask(ymax, mask)
        xmax = tf.boolean_mask(xmax, mask)
        class_id = tf.boolean_mask(class_id, mask)

        ground_truth = tf.concat([y_center, x_center, ymax-ymin, xmax-xmin, class_id], axis=-1)

        image, ground_truth = tf.cond(
            tf.shape(ymin)[0] <= 0,
            lambda: gt_checker_helper(image_copy, ground_truth_copy, output_shape, output_h_f / input_h_f,
                                      output_w_f / input_w_f),
            lambda: (image, ground_truth)
        )

    if pad_truth_to is not None:
        ground_truth = tf.pad(
                        ground_truth, [[0, pad_truth_to-tf.shape(ground_truth)[0]], [0, 0]],
                        constant_values=-1.0
                    )
        return image_copy, ground_truth
    else:
        return image


def rotate_helper(img, angmn, angmx, ymn, xmn, ymx, xmx, outh, outw):
    ang = tf.random.uniform([], angmn, angmx) * 3.1415926 / 180.
    img = tf.contrib.image.rotate(img, ang, 'BILINEAR')
    ang = -ang
    rotate_center_x = (outw - 1.) / 2.
    rotate_center_y = (outh - 1.) / 2.
    offset_x = rotate_center_x * (1 - tf.cos(ang)) + rotate_center_y * tf.sin(ang)
    offset_y = rotate_center_y * (1 - tf.cos(ang)) - rotate_center_x * tf.sin(ang)
    xmnymn_x = xmn * tf.cos(ang) - ymn * tf.sin(ang) + offset_x
    xmnymn_y = xmn * tf.sin(ang) + ymn * tf.cos(ang) + offset_y
    xmxymx_x = xmx * tf.cos(ang) - ymx * tf.sin(ang) + offset_x
    xmxymx_y = xmx * tf.sin(ang) + ymx * tf.cos(ang) + offset_y
    xmnymx_x = xmn * tf.cos(ang) - ymx * tf.sin(ang) + offset_x
    xmnymx_y = xmn * tf.sin(ang) + ymx * tf.cos(ang) + offset_y
    xmxymn_x = xmx * tf.cos(ang) - ymn * tf.sin(ang) + offset_x
    xmxymn_y = xmx * tf.sin(ang) + ymn * tf.cos(ang) + offset_y
    xmn = tf.reduce_min(tf.concat([xmnymn_x, xmxymx_x, xmnymx_x, xmxymn_x], axis=-1), axis=-1,
                        keepdims=True)
    ymn = tf.reduce_min(tf.concat([xmnymn_y, xmxymx_y, xmnymx_y, xmxymn_y], axis=-1), axis=-1,
                        keepdims=True)
    xmx = tf.reduce_max(tf.concat([xmnymn_x, xmxymx_x, xmnymx_x, xmxymn_x], axis=-1), axis=-1,
                        keepdims=True)
    ymx = tf.reduce_max(tf.concat([xmnymn_y, xmxymx_y, xmnymx_y, xmxymn_y], axis=-1), axis=-1,
                        keepdims=True)
    return img, ymn, xmn, ymx, xmx


def gt_checker_helper(image, ground_truth, size, h_ratio, w_ratio):
    image = tf.image.resize(image, size)
    fact = tf.reshape([h_ratio, w_ratio, h_ratio, w_ratio, 1.], [1, 5])
    ground_truth = ground_truth * fact
    return image, ground_truth
