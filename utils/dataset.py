# coding:utf-8
import os
import tensorflow as tf
import numpy as np
import random
import keras.backend as K
from  PIL import Image

valPrecent = 0.2
BatchSize = 2
IMG_SIZE = 640
SEED = 0
N_CLS = 5
random.seed(SEED)

class Dataset():
    def __init__(self,inDir,pixel_mean,pixel_std):
        self.inDir = inDir
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.valPrecent = valPrecent
        self.dataSize ,self.train_Idx,self.val_Idx = self.train_val_split()
        self.BatchSize = BatchSize
        self.Img_size = IMG_SIZE
        self.N_Cls = N_CLS

    def valGenerator(self):
        return self.Generator_step(self.val_Idx)

    def tranGenerator(self):
        return self.Generator_step(self.train_Idx)

    def Generator_step(self,targetIdx):
        y_path = os.path.join(self.inDir, 'masks')
        x_path = os.path.join(self.inDir, 'restricted')

        xList = os.listdir(x_path)
        #打乱序列
        random.shuffle(xList)

        while True:
            X_tmp = np.zeros((self.BatchSize, self.Img_size, self.Img_size, 3))
            y_tmp = np.zeros((self.BatchSize, self.Img_size, self.Img_size, self.N_Cls))
            for count in range(len(targetIdx)):
                batch_idx = count % self.BatchSize
                idx = targetIdx[count]
                # X
                x_img = Image.open(os.path.join(x_path,xList[idx]))
                x_img = x_img.resize((self.Img_size, self.Img_size), Image.ANTIALIAS)
                x_img = (255 - np.array(x_img)) / 255
                for c in range(3):
                    x_img[:, :, c] = (x_img[:, :, c] - self.pixel_mean[c]) / self.pixel_std[c]
                X_tmp[batch_idx,:,:,:] = x_img
                # y
                y_batch_path = os.path.join(y_path,xList[idx].strip('.jpg'))
                y_batch_tmp = np.zeros((self.Img_size, self.Img_size, self.N_Cls))
                for i in range(5):
                    mask = Image.open(os.path.join(y_batch_path, str(i + 1) + '.png'))
                    # print(mask.size)
                    mask = mask.resize((self.Img_size, self.Img_size))
                    mask = np.array(mask)
                    mask[mask > 127] = 255
                    mask[mask <= 127] = 0
                    mask = mask / 255
                    mask = np.uint8(mask)
                    # print(mask.size)
                    # print(y_batch_tmp.shape)
                    y_batch_tmp[:, :, i] = mask
                y_tmp[batch_idx,:,:,:] = y_batch_tmp

                if count % self.BatchSize == 0 or count == len(self.train_Idx):
                    if count == len(self.train_Idx):
                        count = 0
                        yield X_tmp[:batch_idx:, :, :], y_tmp[:batch_idx, :, :, :]
                    else:

                        yield X_tmp, y_tmp


    #采用random模块随机分割训练集和验证集
    def train_val_split(self):

        trainPath = os.path.join(self.inDir, 'restricted')
        total_num = len(os.listdir(trainPath))

        total_idx = [i for i in range(total_num)]
        val_idx = random.choices(range(total_num), k = int(total_num * self.valPrecent))
        train_idx = [i for i in total_idx if i not in val_idx]

        return total_num,train_idx,val_idx

