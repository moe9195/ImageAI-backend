from ISR.models import RDN, RRDN
import argparse, base64, io, cv2, requests, sys, numpy as np, pdb, os
from PIL import Image
import scipy.misc
import tensorflow as tf
sys.path.insert(1, './fast-style-transfer/src')
import vgg, transform, optimize, utils
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer

def super_resolution(img):
    # First we initialize the RDN (Recurral neural network model)
    model = RDN(weights='noise-cancel')

    # Model is then used to 'predict' the higher resolution image
    processed_img_arr = model.predict(np.array(img))
    im = Image.fromarray(processed_img_arr.astype('uint8'), 'RGB')

    return im

def colorize(img):
    # Loading model and network weights from model file
    model = cv2.dnn.readNetFromCaffe('./bw-colorization/model/colorization_deploy_v2.prototxt', './bw-colorization/model/colorization_release_v2.caffemodel')
    pts = np.load('./bw-colorization/model/pts_in_hull.npy')

    # Adding the 1x1 convolutions to the model
    class8 = model.getLayerId("class8_ab")
    conv8 = model.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    model.getLayer(class8).blobs = [pts.astype("float32")]
    model.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Load the image, normalize the pixels to [0, 1] range, convert colorspace to Lab color
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
    if not style in ["la_muse", "rain_princess", "scream", "udnie", "wave", "wreck"]:
        return "Please enter a valid style from the list: [la_muse, rain_princess, scream, udnie, wave, wreck]"

    # whether to use GPU or CPU, setup batch size, and select which style to use
    device_t = '/gpu:0'
    batch_size = 1
    checkpoint_dir = f"./fast-style-transfer/models/{style}.ckpt"

    img_shape = img.shape
    g = tf.Graph()
    curr_num = 0

    # setup GPU configurations for model
    soft_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    # setup batch shape from image size to be compatible with model input shape
    with g.as_default(), g.device(device_t), \
            tf.compat.v1.Session(config=soft_config) as sess:
        batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.compat.v1.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        # 'predict' the output stylized image and save it
        pred = transform.net(img_placeholder)
        saver = tf.compat.v1.train.Saver()

        # use pretrained model as the checkpoint to start from
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        # in this case we do 1 image at a time, so num_iters = 1
        num_iters = 1
        pos = batch_size

        # change image dimensions to 1xNxMx3 from NxMx3 then run the model
        styled = sess.run(pred, feed_dict={img_placeholder:img[np.newaxis,...]})

        # change output back to 1xNxMx3 and convert to image
        im = Image.fromarray(styled[0,:,:].astype('uint8'), 'RGB')

        return im
