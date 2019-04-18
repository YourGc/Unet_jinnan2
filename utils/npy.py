# coding:utf-8
import json
import pandas as pd
import pycocotools.mask as mask_util
import numpy as np

path = r'./train/restricted'

with open(path + 'result0418am.json', 'r') as f:
    result = json.load(f)
f.close()

classes = pd.read_csv('./classifier-train-a.csv', index_col=False)

for i in range(len(classes)):

    masks = result[classes['name'][i]]['mask']
    height, width = result[classes['name'][i]]['size']
    for idx in range(1, 6):
        a = [[0 for _ in range(width)] for _ in range(height)]
        a = np.array(a, dtype=np.uint8)
        arr = mask_util.decode(masks[str(idx)]['counts'])
        arr  # 这个就是那个矩阵了
