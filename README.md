# Object-Detection-API-Tensorflow

# models

Yolo2

Yolo3

SSD

RetinaNet

RefineDet

Light Head Rcnn

PFPNet

CenterNet

# TFRecord generation

1) voc format dataset

2) fill in utils.voc_classname_encoder.py

3) run utils.test_voc_utils.py

# config online image augmentor

fill in dict 'image_augmentor_config' in test-model.py

see utils.image_augmentor.py for details

# config model

fill in dict 'config' in test-model.py


# Train
run test-model.py

# Test
run annotated code in test-model.py


# Experimental Environment
python3.7 tensorflow1.13


# ImageNet pretraining
see utils.tfrecord_imagenet_utils.py
