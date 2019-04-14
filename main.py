# coding:utf-8
from utils.mask import getMask
from utils.AuxiliaryTool import *
from unet.Unet import Unet
from utils.dataset import Dataset
inDir = r'./data/train'
outDir = r'./data/train/masks'
PIXEL_MEAN = [0.126947,0.119578,0.193871]
PIXEL_STD = [0.239344,0.155906,0.210617]

if __name__ == '__main__':
    #getMask(inDir,outDir)
    #pixel_mean,pixel_std = img_mean_std(os.path.join(inDir,'restricted'))
    print('-----building dataset-----')
    dataset = Dataset(inDir,PIXEL_MEAN,PIXEL_STD)
    print('-----done-----')
    print('-----building model-----')
    unet_model = Unet(dataset)
    print('-----done-----')
    print('-----start train-----')
    unet_model.train()


