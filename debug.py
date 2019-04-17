# coding:utf-8

import os
import h5py
import numpy as np
import random
import tensorflow as tf
import keras.backend as K
from PIL import Image

# y
y_tmp = np.zeros((2, 640, 640, 5))
y_batch_path = r'./data/train/masks/123'
y_batch_tmp = np.zeros((640, 640, 5))
for i in range(5):
    mask = Image.open(os.path.join(y_batch_path, str(i + 1) + '.png'))
    # print(mask.size)
    mask = mask.resize((640, 640), Image.ANTIALIAS)
    mask = np.array(mask)
    mask[mask > 127] = 255
    mask[mask <= 127] = 0
    mask = mask / 255
    mask = np.uint8(mask)
    # print(mask.size)
    mask = np.uint8(mask)
    # print(y_batch_tmp.shape)
    y_batch_tmp[:, :, i] = np.array(mask)
with tf.Session() as sess:
    print(K.sum(y_batch_tmp).eval())
