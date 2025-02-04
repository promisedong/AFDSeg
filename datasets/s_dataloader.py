import torch
import numpy as np
import cv2
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image  # 这个读取数据是RGB
from torch.utils.data import DataLoader, Dataset
import random
from utils.tools import resize

from datasets.transformer import *
from datasets.boundary import convert_boundary,draw_gassuian
from nets.Transim.datasets import TargetCDF

class BuildDataset(Dataset):
    def __init__(self, input_shape = (512, 512),
                 train_file = None,
                 num_classes = 1 + 1,
                 augment = False #训练 True
                 ):

        self.image_size = input_shape

        self.num_class = num_classes #类别

        self.train = []
        self.test = []
        self.val = []

        trainlines = open(train_file)


        #-----------------------------------------#
        #                    ACDC
        #-----------------------------------------#
        self.in2cls = np.array([ 0, 0, 0,  # 前景 白
                           255, 0, 0,  # 蓝色
                           0, 255, 0,  # 背景 红
                           0, 0, 255,
                           153, 230, 255,
                           173, 203, 248,
                           180, 224, 197,
                           226, 182, 213
                           ])  # 黄色

        self.in2cls = self.in2cls.reshape((-1, 3))

        self.augment = augment
        if augment: #推理
            self.data_aug = Data_Augment([#RandomHue(),
                          #RandomSaturation(),
                         #RandomBrightness(),
                          RandomHFlip(),
                          RandomVFlip(),
                          #RandomBlur(),
                          RandomRotate(),
                          Noramlize()        ])

        else:
            self.data_aug = Data_Augment([

                Noramlize()]) #

        for line in trainlines.readlines():
            splited = line.strip().split()
            self.train.append([splited[ 0 ],splited[1]])


        trainlines.close()



    def _get_point(self,binary_mask):


        # Sobel h v
        grad_x = cv2.Sobel(binary_mask, cv2.CV_64F, 1, 0, ksize = 1)
        grad_y = cv2.Sobel(binary_mask, cv2.CV_64F, 0, 1, ksize = 1)

        # length
        gradient_magnitude = cv2.magnitude(grad_x, grad_y)

        # edge
        _, edge_map = cv2.threshold(gradient_magnitude, 20, 1, cv2.THRESH_BINARY)



        edge_map = edge_map.astype(np.uint8)




        # find contours
        contours, _ = cv2.findContours(edge_map,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        instance = [ ]
        alpha = 5.0
        for c in contours:
            t = c.reshape(-1, 2)

            t = cv2.approxPolyDP(t, alpha, True)# TODO? key point

            instance.append(t.reshape(-1,2))

        #instance = np.array(instance)


        #cv2.imshow('im0',edge_map )

        #mask = np.zeros_like(edge_map)

        return edge_map,instance

    def handler(self,fname,maskname):
        image = Image.open(fname).convert('RGB')

        mask = Image.open(maskname).convert('L')

        w,h = image.size #TODO

        # TODO multi
        assert np.array(image.size).all() == np.array(mask.size).all(), "Not Match! Error"

        if  (self.augment) :
            if random.uniform(0, 1) > 0.5:
                patch_h = random.randint(0,self.image_size[0] // 2  + self.image_size[0] // 4)
                patch_w = random.randint(0,self.image_size[1] // 2 + self.image_size[0] // 4)

                _w = random.choices(np.array([32,64,128,256,512]))[0]
                _h = random.choices(np.array([32,64,128,256,512]))[0]

                image = image.crop((patch_w,patch_h ,patch_w + _w,patch_h + _h))
                mask = mask.crop((patch_w, patch_h , patch_w + _w, patch_h + _h))

            if random.uniform(0,1) > 0.5:
                w, h = image.size

                scalew, scaleh = self.image_size[ 0 ] / w, self.image_size[ 1 ] / h
                scale = min(scaleh, scalew)
                neww, newh = int(scale * w), int(scale * h)

                dx = self.image_size[ 0 ] - neww
                dy = self.image_size[ 1 ] - newh

                # 128 过于接近云
                new_image = Image.new("RGB", self.image_size, (0, 0, 0))
                image = image.resize((neww, newh))

                new_mask = Image.new("L", self.image_size, (0))
                mask = mask.resize((neww, newh))

                if random.uniform(0, 1) > 0.5:
                    dx //= 2
                    dy //= 2

                new_image.paste(image, (dx, dy))
                new_mask.paste(mask, (dx, dy))

            else :
                new_image = image.resize(self.image_size)
                new_mask = mask.resize(self.image_size)


        else :
            new_image = image.resize(self.image_size)
            new_mask = mask.resize(self.image_size)



        image = np.array(new_image, dtype = np.float32)


        # t = image.copy()
        # t = t.astype(np.uint8)
        # crop = image[dy:dy+newh,dx:dx+neww,:]
        # crop = crop.astype(np.uint8)
        # #crop = cv2.resize(crop,(ow,oh))
        # print(crop.shape)
        # cv2.imshow('im', t)
        # cv2.imshow('crop',crop)
        # cv2.waitKey(0)

        # ----------------------------------#
        # 二分类
        # ----------------------------------#
        mask = np.array(new_mask)  # 单通道图像

        result = self.data_aug({
            'image': image,
            'mask': mask
        })

        image = result[ 'image' ]

        mask = result[ 'mask' ]

        modify_png = np.zeros_like(mask)

        # color2id
        # for c in range(len(self.in2cls)):
        #     ind = (mask == c)
        #     mask[ ind ] = color[ c ]

        modify_png[ mask > 150 ] = 1

        _,boundary1 = convert_boundary(mask,1)
        heat2, boundary2 = convert_boundary(mask, 2)
        heat4, boundary4 = convert_boundary(mask, 4)
        heat8, boundary8 = convert_boundary(mask, 8)
        heat16, boundary16 = convert_boundary(mask, 16)

        instance_edge,instance = self._get_point(mask)

        stride = 2

        instance_point = np.zeros((self.image_size[0] // stride,self.image_size[1] // stride))

        for i in instance:
         for box in i:
             x,y = box

             x , y = x//stride, y//stride

             instance_point[y,x] = 1


             instance_point = draw_gassuian(instance_point,x,y,2,self.image_size[0]//stride,self.image_size[1]//stride)



        # 关键点
        instance_point = instance_point.astype(np.float32)


        # -------------------------------#
        # 多分类
        # -------------------------------#
        # for c in range(self.num_class):
        #     mask[mask==c] = c

        T_mask = np.zeros((self.image_size[ 1 ], self.image_size[ 0 ], self.num_class))

        # --------------------------------------#
        # 两种构建one-hot编码形式
        # --------------------------------------#
        for c in range(self.num_class):
            T_mask[ modify_png == c, c ] = 1
        T_mask = np.transpose(T_mask, (2, 0, 1))
        # T_mask = np.eye(self.num_class)[mask.reshape(-1)] #
        # T_mask = np.reshape(T_mask,(self.image_size[1],self.image_size[0],self.num_class))

        # vision
        """
        back = T_mask[0,...]
        fg = T_mask[1,...]
        cv2.imshow('im',image.astype(np.uint8))
        cv2.imshow('bg',back)
        cv2.imshow('fg',fg)
        cv2.waitKey(0)
        """

        img = np.transpose(image, (2, 0, 1))
        heat2 = np.transpose(heat2,(2,0,1))
        heat4 = np.transpose(heat4, (2, 0, 1))

        boundary1 = np.transpose(boundary1, (2, 0, 1))
        boundary2 = np.transpose(boundary2, (2, 0, 1))
        boundary4 = np.transpose(boundary4, (2, 0, 1))
        boundary8 = np.transpose(boundary8, (2, 0, 1))
        boundary16 = np.transpose(boundary16, (2, 0, 1))


        return new_image,img,T_mask,(modify_png,instance_point,instance_edge),\
               (heat2,heat4),((boundary1,boundary2),boundary4,boundary8,boundary16)

    def __getitem__(self, idx):
        ind1 = idx % len(self.train)

        imagename = self.train[ind1][0]
        maskname = self.train[ind1][1]


        # select other image
        ind2 = idx % random.randint(1,len(self.train))
        imagename2 = self.train[ ind2 ][ 0 ]
        maskname2 = self.train[ ind2 ][ 1 ]



        newim1,image,label,(png,point,edge),(heat2,heat4),\
        ((boundary1,boundary2),boundary4,boundary8,boundary16) = \
            self.handler(imagename,maskname)

        newim2, _, _, *_ = \
            self.handler(imagename2, maskname2)


        recon,cdf = TargetCDF(newim1,newim2)
        recon = self._norm(recon)
        recon = np.transpose(recon,(2,0,1))

        return  newim1,image,label,(png,point,edge,recon,cdf),(heat2,heat4),\
        ((boundary1,boundary2),boundary4,boundary8,boundary16)

    def __len__(self):
        return len(self.train)

    # ------------数据增强-----------------#
    def _norm(self, img):
        img = img/255.
        # img -= self.mean
        # img /= self.std
        return img


def convert(data):
    gtlables = np.array(data, np.float32)
    gtlables = torch.from_numpy(gtlables).float()

    return gtlables


def collate_seg(batch):

     gtimages,gtlables,gtpng,heat2s,\
     heat4s,boundary2s,boundary4s,boundary8s,boundary16s= \
         [],[],[],[],[],[],[],[],[]

     boundary1s = []



     edges = []
     points = []

     recons,cdfs = [],[]

     orimgs = []

     for newim,image,label,(png,point,edge,recon,cdf),(heat2,heat4),((boundary1,boundary2),boundary4,boundary8,boundary16) in batch:

         orimgs.append(np.array(newim))
         gtimages.append(image)
         gtlables.append(label)
         gtpng.append(png)

         points.append(point)
         edges.append(edge)

         recons.append(recon)
         cdfs.append(cdf)

         #loc loss
         heat2s.append(heat2)
         heat4s.append(heat4)

         # boundary loss
         boundary1s.append(boundary1)

         boundary2s.append(boundary2)
         boundary4s.append(boundary4)
         boundary8s.append(boundary8)
         boundary16s.append(boundary16)

     # image
     gtimages = convert(gtimages)

     # label
     gtlables = convert(gtlables)

     # one-hot
     gtpng = np.array(gtpng,np.int64)
     gtpng = torch.from_numpy(gtpng).long()

     # heatmap loc
     heat2s = convert(heat2s)
     heat4s = convert(heat4s)

     # boundary
     boundary1s = convert(boundary1s)

     boundary2s = convert(boundary2s)
     boundary4s = convert(boundary4s)
     boundary8s = convert(boundary8s)
     boundary16s = convert(boundary16s)

     #

     points = np.asarray(points)

     points = torch.from_numpy(points).float()

     edges = np.array(edges,dtype = np.int64)
     edges = torch.from_numpy(edges)


     recons = torch.from_numpy(np.array(recons)).float()
     cdfs = torch.from_numpy(np.array(cdfs)).float()

     orimgs = np.array(orimgs)

     # 使用边界和关键点进行损失指导
     return orimgs,gtimages,gtlables,(gtpng,heat2s,edges,recons,cdfs),(heat2s,heat4s),\
            ((boundary1s,boundary2s),boundary4s,boundary8s,boundary16s)


if __name__ == '__main__':
 def gt():
    data = BuildDataset(train_file = '../kvasir_seg_train.txt',augment = True)
    train_loader = DataLoader(data, batch_size = 8, shuffle = True,collate_fn = collate_seg)
    train_iter = iter(train_loader)
    for i,batch in enumerate(train_loader):

        orimgs,image, label, (png,points,edges,recons,cdfs), (heat2, heat4), \
        ((boundary1,boundary2), boundary4, boundary8, boundary16) = batch

        image8 = F.interpolate(image,(512//4,512//4),mode='bilinear',align_corners=True)

        print(image.shape,label.shape,png.shape,png.dtype,boundary1.shape,boundary2.shape)

        print(points.shape,edges.shape)

        for ind in range(image.shape[0]):

         sim = image[ind].numpy()
         sim = np.transpose(sim,(1,2,0)) * 255.
         sim = sim.astype(np.uint8)[...,::-1]

         sim8 = image8[ind].numpy()
         sim8 = np.transpose(sim8, (1, 2, 0)) * 255.
         sim8 = sim8.astype(np.uint8)[..., ::-1]

         # 热力图
         h2 = points[ind].numpy()
         h2 = h2 * 255.
         h2 = h2.astype(np.uint8)

         # 边界
         boundary2ss = edges[ ind ].numpy()
         boundary2ss = boundary2ss * 255.
         boundary2ss = boundary2ss.astype(np.uint8)

         # 重构图像
         recon = recons[ind].numpy()
         recon = np.transpose(recon, (1, 2, 0)) * 255.

         recon = recon.astype(np.uint8)

         # orimg
         orim = orimgs[ind]

         cv2.imshow('orim',orim[:,:,::-1])

         for c in range(label.shape[1]):

             T = label[ind].numpy()

             T = np.transpose(T,(1,2,0))

             cv2.imshow(f"im_{c}",T[...,c])


         cv2.imshow('h2',h2)
         cv2.imshow('b2',boundary2ss)
         cv2.imshow('sim',sim)
         cv2.imshow('sim8', sim8)
         cv2.imshow('recon',recon[:,:,::-1])
         cv2.waitKey(0)


 gt()


