# Object-Detection-API-Tensorflow

# Features
## Every model is implemented in only one file!
# Models

Yolo2

Yolo3

SSD

RetinaNet

RefineDet

Light Head Rcnn

PFPNet

CenterNet

FCOS

# Train your own data
# Train your own data
## 1. TFRecord generation

1) voc format dataset

2) fill in utils.voc_classname_encoder.py

3) run utils.test_voc_utils.py

## 2. config online image augmentor

fill in dict 'image_augmentor_config' in test-model.py

see utils.image_augmentor.py for details

see https://github.com/Stick-To/Online_Image_Augmentor_tensorflow for details
## 3. config model

fill in dict 'config' in test-model.py


## 4. Train
run test-model.py

The pre-trained vgg_16.ckpt could be downloaded from http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz

## 5. Test
run annotated code in test-model.py

## 6. ImageNet pretraining
see utils.tfrecord_imagenet_utils.py

## 7. different conv backone
https://github.com/Stick-To/Deep_Conv_Backone

## 8. Instantiation of result 
 corresponding repository in https://github.com/Stick-To
 
# Experimental Environment
python3.7 tensorflow1.13
