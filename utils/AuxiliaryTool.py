# coding:utf-8
import os
from PIL import Image
import numpy as np
IMG_SIZE = 640

def img_mean_std(inDir):
    trainList = os.listdir(inDir)

    #caculate mean
    R_channel = 0
    G_channel = 0
    B_channel = 0

    pixel_num = 0
    for idx in range(len(trainList)):
        if not str(trainList[idx]).endswith('jpg') :continue
        filename = trainList[idx]
        img = Image.open(os.path.join(inDir, filename))
        # img = img.resize((IMG_SIZE,IMG_SIZE), Image.ANTIALIAS)
        img = 255 - np.array(img)
        R_channel = R_channel + np.sum(img[:, :, 0])
        G_channel = G_channel + np.sum(img[:, :, 1])
        B_channel = B_channel + np.sum(img[:, :, 2])
        w,h,c = img.shape
        pixel_num += w * h

    R_mean = R_channel / pixel_num
    G_mean = G_channel / pixel_num
    B_mean = B_channel / pixel_num

    print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
    print("Normed:R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean/255, G_mean/255, B_mean/255))

    #caculate std
    R_mean /= 255
    G_mean /= 255
    B_mean /= 255
    RGB_mean = [R_mean,G_mean,B_mean]

    R_channel = 0
    G_channel = 0
    B_channel = 0

    for idx in range(len(trainList)):
        filename = trainList[idx]
        img = Image.open(os.path.join(inDir, filename))
        #img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        img = 255 - np.array(img)
        img = img / 255
        w, h = img.shape[:2]
        R_channel = R_channel + np.sum((img[:, :, 0] - R_mean)**2)/(w * h)
        G_channel = G_channel + np.sum((img[:, :, 1] - G_mean)**2)/(w * h)
        B_channel = B_channel + np.sum((img[:, :, 2] - B_mean)**2)/(w * h)

    R_std = R_channel / len(trainList)
    G_std = G_channel / len(trainList)
    B_std = B_channel / len(trainList)
    R_std = R_std**0.5
    G_std = G_std**0.5
    B_std = B_std**0.5
    #print("R_std is %f, G_std is %f, B_std is %f" % (R_std, G_std, B_std))
    print("Normed:R_std is %f, G_std is %f, B_std is %f" % (R_std, G_std, B_std))

    RGB_std = [R_std,G_std,B_std]
    return RGB_mean,RGB_std