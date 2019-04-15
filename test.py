# coding:utf-8
import json
import pandas as pd
import pycocotools.mask as maskutil
import numpy as np

with open('./result-0414am-9w.json', 'r') as f:
    result = json.load(f)
f.close()

classes = pd.read_csv('./classifier-train-a.csv', index_col=False)

for i in range(len(classes)):
    flag = classes['class'][i]
    if flag: continue

    masks = result[classes['name'][i]]['mask']
    height, width = result[classes['name'][i]]['size']
    for idx in range(1, 6):
        a = [[0 for _ in range(width)] for _ in range(height)]
        a = np.array(a, dtype=np.uint8)
        rle = maskutil.encode(np.array(a[:, :, np.newaxis], order='F'))[0]
        rle['counts'] = rle['counts'].decode('ascii')

        # mask_cls['counts'] =rle['counts']
        # mask_cls['size'] =rle['size']
        masks[str(idx)]['counts'] = rle['counts']
        # counts = masks[str(idx)]['counts']
        # print(counts.type)
        # mask = maskutil.decode(counts)  # decode to mask
        # mask = np.uint8(0)
        # masks[idx]['count'] = maskutil.encode(mask) #encode to mask

with open('./submit0415am.json', 'w') as f:
    json.dump(result, f, sort_keys=True, indent=4)
f.close()
