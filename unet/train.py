# coding:utf-8 
from Unet import Unet
from keras.utils.vis_utils import plot_model

if __name__ == '__main__':
    unet = Unet()
    plot_model(unet.model,'unet.png')
