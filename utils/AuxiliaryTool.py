# coding:utf-8
import os
import copy
from PIL import Image
import numpy as np
import json
from progressbar import *
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


def ImgVertical():
    path = r'../data/train'
    with open(os.path.join(path, 'train_restriction.json'), 'r') as f:
        annos = json.load(f)
    f.close()

    img_path = path + r'/restricted'
    img_names = os.listdir(img_path)

    imageInfo = annos['images']
    imageInfo = sorted(imageInfo, key=lambda x: int(x['file_name'].strip('.jpg')))
    img_nums = len(imageInfo)

    imageAnnos = annos['annotations']
    imageAnnos = sorted(imageAnnos, key=lambda x: x['id'])
    annos_nums = len(imageAnnos)
    idx = 0

    widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ']
    pbar = ProgressBar(widgets=widgets, maxval=img_nums).start()

    for i in range(img_nums):
        pbar.update(i)

        info_tmp = copy.deepcopy(imageInfo[i])

        img = Image.open(img_path + r'/' + info_tmp['file_name'])

        info_tmp['file_name'] = str(i + img_nums) + '.jpg'

        img_arr = np.array(img)
        img_arr = img_arr[::-1, :, :]
        img_v = Image.fromarray(img_arr)
        img_v.save(img_path + '/' + info_tmp['file_name'], quality=95)

        img_id = info_tmp['id']
        w, h = info_tmp['width'], info_tmp['height']
        print(w, h)

        info_tmp['id'] = info_tmp['id'] + img_nums
        imageInfo.append(info_tmp)

        while idx < annos_nums:
            if imageAnnos[idx]['image_id'] != img_id: break
            anno_tmp = copy.deepcopy(imageAnnos[idx])
            anno_tmp['id'] = anno_tmp['id'] + annos_nums
            anno_tmp['image_id'] = info_tmp['id']

            for rect in anno_tmp['minAreaRect']:
                rect[1] = h - rect[1]

            anno_tmp['bbox'][1] = h - anno_tmp['bbox'][1] - anno_tmp['bbox'][3]

            for seg_idx in range(1, len(anno_tmp['segmentation'][0]), 2):
                anno_tmp['segmentation'][0][seg_idx] = h - anno_tmp['segmentation'][0][seg_idx]

            imageAnnos.append(anno_tmp)
            idx = idx + 1

    with open(os.path.join(path, 'new_anno.json'), 'w') as f:
        annos['images'] = imageInfo
        annos['annotations'] = imageAnnos
        json.dump(annos, f, indent=4, sort_keys=True)
    f.close()


def overturn():
    img_path = r'../data/train/restricted'
    img_names = os.listdir(img_path)

    for name in img_names:
        img = Image.open(os.path.join(img_path, name))
        img = 255 - np.array(img)
        Image.fromarray(img).save(os.path.join(img_path, name), quality=95)

if __name__ == '__main__':
    overturn()
