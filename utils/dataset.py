# coding:utf-8
import os
import h5py
import numpy as np
import random

from  PIL import Image

valPrecent = 0.2
BatchSize = 8
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
        self.Generator_step(self.val_Idx)

    def tranGenerator(self):
        self.Generator_step(self.train_Idx)

    def Generator_step(self,targetIdx):
        y_path = os.path.join(self.inDir, 'masks')
        x_path = os.path.join(self.inDir, 'restricted')

        xList = os.listdir(x_path)
        #打乱序列
        random.shuffle(xList)

        while True:
            X_tmp = np.zeros((self.BatchSize,3,self.Img_size,self.Img_size))
            y_tmp = np.zeros((self.BatchSize,self.N_Cls,self.Img_size,self.Img_size))
            for count in range(len(targetIdx)):
                batch_idx = count % self.BatchSize
                idx = targetIdx[count]
                # X
                x_img = Image.open(os.path.join(x_path,xList[idx]))
                x_img = (np.array(x_img)/255 - self.pixel_mean) / self.pixel_std
                X_tmp[batch_idx,:,:,:] = x_img
                # y
                y_batch_path = os.path.join(y_path,xList[idx].strip('.jpg'))
                y_batch_tmp = np.zeros((self,N_CLS,self.Img_size,self.BatchSize))
                for i in range(5):
                    mask = Image.open(os.path.join(y_batch_path,str(i) + '.png'))
                    y_batch_tmp[i, :, :] = np.array(mask)
                y_tmp[batch_idx,:,:,:] = y_batch_tmp

                if count % self.BatchSize == 0 or count == len(self.train_Idx):
                    if count == len(self.train_Idx) : count = 0
                    yield X_tmp,y_tmp


    #采用random模块随机分割训练集和验证集
    def train_val_split(self):

        trainPath = os.path.join(self.inDir, 'restricted')
        total_num = len(os.listdir(trainPath))

        total_idx = [i for i in range(total_num)]
        val_idx = random.choices(range(total_num), k = int(total_num * self.valPrecent))
        train_idx = [i for i in total_idx if i not in val_idx]

        return total_num,train_idx,val_idx

