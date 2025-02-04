import torch
import numpy as np
import cv2
import os
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image  # 这个读取数据是RGB
from torch.utils.data import DataLoader, Dataset
import random
def TargetCDF(sn = None,tn = None):


    assert ~((sn == None) and (tn == None)),"please input image"
    # 读取源图像和目标图像

   
    source_img = np.ascontiguousarray(sn)


  
    target_img = np.ascontiguousarray(tn)
   



    source_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2LAB)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB)
    #print(source_img.shape,target_img.shape)
    

    lsm, asm, bsm = source_img[ ..., 0 ], source_img[ ..., 1 ], source_img[ ..., 2 ]
    lum, aum, bum = target_img[ ..., 0 ], target_img[ ..., 1 ], target_img[ ..., 2 ]

    newl,lcdf = mapping(lsm, lum)
    newa,acdf = mapping(asm, aum)
    newb,bcdf = mapping(bsm, bum)

    #lsm = lsm.astype(np.uint8)
    C = lcdf.shape[0]



    cdf = np.stack((lcdf,acdf,bcdf),axis = 1)

    newlab = cv2.merge([ newl, newa, newb ])
    img = cv2.cvtColor(newlab, cv2.COLOR_LAB2RGB)


    return img,cdf




def mapping(source_img, target_img):
        # 计算直方图
        source_hist = cv2.calcHist([ source_img ], [ 0 ], None, [ 256 ], [ 0, 256 ])
        target_hist = cv2.calcHist([ target_img ], [ 0 ], None, [ 256 ], [ 0, 256 ])



        # 计算累积分布函数 TODO?
        source_cdf = np.cumsum(source_hist) / (np.sum(source_hist))
        target_cdf = np.cumsum(target_hist) / (np.sum(target_hist))

        mean = (target_cdf - np.min(source_cdf)) / (target_cdf - source_cdf + 1e-10)
        # 创建匹配映射

        target_cdf = np.where(source_cdf > mean, source_cdf, target_cdf)

        mapping = np.interp(source_cdf, target_cdf, range(256)).astype(np.uint8)

        # 应用匹配映射
        matched_img = mapping[ source_img ]
        return matched_img,target_cdf




class UnetDataset(Dataset):
    def __init__(self, input_shape = (128, 128),
                 source_file = None,
                 target_file = None,
                 num_classes = 1 + 1,
                 mode = 'rank'):

        self.image_size = input_shape

        self.mode = mode

        self.num_class = num_classes #类别

        self.source = [] #收集原始图像
        self.target= [] #收集掩码图像
        sourcelines = open(source_file)
        targetlines = open(target_file)




        for line in sourcelines.readlines():
            splited = line.split(' ')[0].strip()
            self.source.append(splited)

        for line in targetlines.readlines():
            splited = line.split(' ')[0].strip()
            self.target.append(splited)


        sourcelines.close()
        targetlines.close()
        self.l = len(self.target)



    def __getitem__(self, idx):
        ind1 = idx % len(self.source)
        sourcename = self.source[ind1]


        ind2 = idx % len(self.target)
        targetname = self.target[ind2]

        

        img,cdf = TargetCDF(sourcename,targetname)
       
        img = np.transpose(img,(2,0,1))


        return self._norm(img),cdf


    def __len__(self):
        return self.l
    # ------------数据增强-----------------#
    def _norm(self, img):
        img = img/255.
        # img -= self.mean
        # img /= self.std
        return img




def collate_seg(batch):



     imgs, cdfs =  [],[]

     for img,cdf in batch:
        imgs.append(img)
        cdfs.append(cdf)


     imgs = np.array(imgs)
     
     imgs = torch.from_numpy(imgs).float()

     cdfs = np.array(cdfs)
     cdfs = torch.from_numpy(cdfs).float()


     return imgs,cdfs




if __name__ == '__main__':

    path1 = ''

    path2 = ''
    fs = os.listdir(path1)
    txt1 = 'source.txt'

    wf = open(txt1,'w+')

    #txt2 = 'target.txt'
    for i in fs:
     p = os.path.join(path1,i)

     wf.write(p+'\n')


