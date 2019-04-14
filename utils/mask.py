# coding:utf-8
import cv2 as cv
import json
import numpy as np
import pycocotools.mask as maskutil
from progressbar import *
from scipy import misc


N_CLS = 5
#修改于津南二可视化开源代码
def get_index(image_id, load_dict):  # get seglist and label list by image_id
    seg_list = []
    label_list = []
    for i in range(len(load_dict['annotations'])):
        if image_id == load_dict['annotations'][i]['image_id']:
            seg_list.append(i)
            label_list.append(load_dict['annotations'][i]['category_id'])
    return seg_list, label_list

# def get_color(class_id):  # for Distinguish different classes
#     return class_id * 50

def getMask(inDir,outDir):
    print("start making masks")
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    with open(os.path.join(inDir,'train_restriction.json'), 'r') as f:
        load_dict = json.load(f)
        paths = os.listdir(os.path.join(inDir,'restricted'))
        #progressBar
        widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets,maxval=len(paths)).start()

        for i in range(len(paths)):
            im_path = paths[i]
            pbar.update(i)
            im = cv.imread(os.path.join(inDir,'restricted') + '/' + im_path)
            w,h = im.shape[:2]
            seg_list, label_list = get_index(int(im_path[:-4]), load_dict)

            masks = np.zeros((w,h,N_CLS))
            for (seg_idx,label_idx) in zip(seg_list,label_list):
                seg = load_dict['annotations'][seg_idx]['segmentation'][0]  # load first seg in seg list
                compactedRLE = maskutil.frPyObjects([seg], im.shape[0], im.shape[1])  # compress through RLE
                mask = maskutil.decode(compactedRLE)  # decode to mask
                #print(masks[:,:,label_idx - 1].shape,mask.shape)
                masks[:,:,label_idx - 1] += mask[:,:,0]

            masks = np.transpose(masks,(2,1,0))
            #save mask
            savePicDir = os.path.join(outDir,im_path.split('.')[0])
            if not os.path.exists(savePicDir):
                os.mkdir(savePicDir)

            for i in range(5):
                masks[i][masks[i] > 1] = 1
                misc.imsave(os.path.join(savePicDir,'{}.png'.format(i+1)), masks[i])
                # mask_idx = Image.fromarray(np.uint8(masks[i]),mode='RGBA')
                # mask_idx.save(os.path.join(savePicDir,'{}.png'.format(i+1)))
            time.sleep(0.01)

    f.close()




