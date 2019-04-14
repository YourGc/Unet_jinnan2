# coding:utf-8
import os
import json
import numpy as np
from PIL import Image
root_path = r'../data/test'
save_path = r'../data/files'
if not os.path.exists(save_path):
    os.mkdir(save_path)
names = os.listdir(root_path)

ans = dict()
for name in names:
    ans[name] = {}
    ans[name]["image_name"] = name
    img = Image.open(os.path.join(root_path,name))
    x,y = img.size[:2]
    masks = np.zeros((5,y,x))
    for i in range(5):
        np.save(save_path + '/' + name.strip('.jpg') + '_'+str(i + 1) + '.npy',masks[i])


