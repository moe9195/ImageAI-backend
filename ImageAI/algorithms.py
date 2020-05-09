import sys
from ISR.models import RDN, RRDN
import argparse
import base64
import io
import cv2
import requests
import numpy as np
import pdb
import os
from PIL import Image
import scipy.misc
import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import load_model
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer
import time
sys.path.insert(1, './fast-style-transfer/src')
import vgg, transform, optimize, utils
sys.path.insert(1, './deblur-gan/deblurgan')
from model import generator_model
sys.path.insert(1, './efficientnet')
import eval_ckpt_main as eval_ckpt


def super_resolution(img):
    # First we initialize the RDN (Recurral neural network model)
    model = RDN(weights='noise-cancel')

    # Model is then used to 'predict' the higher resolution image
    processed_img_arr = model.predict(np.array(img))
    im = Image.fromarray(processed_img_arr.astype('uint8'), 'RGB')

    return im


def colorize(img):
    # Loading model and network weights from model file
    model = cv2.dnn.readNetFromCaffe(
        './bw-colorization/model/colorization_deploy_v2.prototxt',
        './bw-colorization/model/colorization_release_v2.caffemodel')
    pts = np.load('./bw-colorization/model/pts_in_hull.npy')

    # Adding the 1x1 convolutions to the model
    class8 = model.getLayerId("class8_ab")
    conv8 = model.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    model.getLayer(class8).blobs = [pts.astype("float32")]
    model.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Load the image, normalize the pixels to [0, 1] range, convert colorspace
    # to Lab color
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    # Resize image to 224x224 (this is the dimensions of the neural network input)
    # Extract the L from colorspace and center the mean
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # The network expects black and white images, this is why we extract the L channel
    # We pass the L channel to the network, which predicts the a and b channel values (colors)
    # a = green/red, b = blue/yellow, L = black/white
    model.setInput(cv2.dnn.blobFromImage(L))
    ab = model.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the output image into our input image
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    # Grab L channel from original image and concatenate to the ab channels
    # Now we have all three channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # Convert the output image from the Lab color space to RGB, then
    # Clip any values that fall outside the range [0, 1]
    # Rescale back to [0, 255] then convert it to 8bit image
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    im = Image.fromarray(colorized.astype('uint8'), 'RGB')

    return im


def deep_art(img, style):
    # check if style is valid
    if style not in [
        "la_muse",
        "rain_princess",
        "scream",
        "udnie",
        "wave",
            "wreck"]:
        return "Please enter a valid style from the list: [la_muse, rain_princess, scream, udnie, wave, wreck]"

    # whether to use GPU or CPU, setup batch size, and select which style to
    # use
    device_t = '/gpu:0'
    batch_size = 1
    checkpoint_dir = f"./fast-style-transfer/models/{style}.ckpt"

    img_shape = img.shape
    g = tf.Graph()
    curr_num = 0

    # setup GPU configurations for model
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    # setup batch shape from image size to be compatible with model input shape
    with g.as_default(), g.device(device_t), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        # 'predict' the output stylized image and save it
        pred = transform.net(img_placeholder)
        saver = tf.train.Saver()

        # use pretrained model as the checkpoint to start from
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        # in this case we do 1 image at     a time, so num_iters = 1
        num_iters = 1
        pos = batch_size

        # change image dimensions to 1xNxMx3 from NxMx3 then run the model
        styled = sess.run(
            pred, feed_dict={img_placeholder: img[np.newaxis, ...]})

        # change output back to 1xNxMx3 and convert to image
        im = Image.fromarray(styled[0, :, :].astype('uint8'), 'RGB')

        return im


def deblur(img):
    # sess = tf.Session(config=tf.ConfigProto(
    #       allow_soft_placement=True, log_device_placement=True))

    # load model and model weights
    g = generator_model()
    g.load_weights('./deblur-gan/generator.h5')

    # resize image, center mean and normalize
    img = cv2.resize(img, (256, 256))[np.newaxis, ...]
    img = (img - 127.5) / 127.5
    x_test = img

    # make prediction and format output from model
    generated_images = g.predict(x=x_test)
    generated = np.array([(img * 127.5 + 127.5).astype('uint8')
                          for img in generated_images])[0, :, :, :]
    im = Image.fromarray(generated.astype(np.uint8), 'RGB')
    return im


def classify(img):
    # first save image as a temporary file
    Image.fromarray(img).save('./images/storage/temp.jpeg')
    img_path = './images/storage/temp.jpeg'

    # load the model from the efficientnet directory
    model_name = 'efficientnet-b3'
    ckpt_dir = './efficientnet/noisy-student-efficientnet-b3'

    # load the labels file
    labels_map_file = './efficientnet/labels_map.txt'

    # get the evaluation driver and make the prediction
    eval_driver = eval_ckpt.get_eval_driver(model_name)
    pred_idx, pred_prob = eval_driver.eval_example_images(
        ckpt_dir, [img_path], labels_map_file)

    # delete the image from storage
    os.remove(img_path)

    classes = json.loads(tf.gfile.Open(labels_map_file).read())

    obj = {}
    print(pred_idx)
    print(pred_prob)
    for idx, prob in zip(pred_idx[0], pred_prob[0]):
        obj[str(round(prob, 3))] = classes[str(idx)]

    return(obj)
