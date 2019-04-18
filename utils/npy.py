# coding:utf-8
import json
import pandas as pd
import pycocotools.mask as mask_util
import numpy as np
import os
from progressbar import *


def make_npy():
    path = r'../data/train/predict_npy'
    if not os.path.exists(path):
        os.mkdir(path)

    with open('../result0418am.json', 'r') as f:
        result = json.load(f)
    f.close()

    widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ']
    pbar = ProgressBar(widgets=widgets, maxval=len(result.keys())).start()
    i = 0
    for key, value in result.items():
        i = i + 1
        pbar.update(i)
        masks = result[key]['mask']
        height, width = result[key]['size']
        save_path = os.path.join(path, str(key).strip('.jpg'))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for idx in range(1, 6):
            # a = np.zeros((height,width),dtype=np.uint8)
            arr = mask_util.decode(masks[str(idx)])

            np.save(os.path.join(save_path, r"{}.npy".format(idx)), arr)  # 这个就是那个矩阵了


if __name__ == '__main__':
    make_npy()
