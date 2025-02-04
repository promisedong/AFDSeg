# -------------------------------------#
# 更换模型
# -------------------------------------#
from nets.BuildRT import BuildRT
from nets.Bisenet import bisenetv1, bisenetv2
from nets.Pidnet import pidnet
from nets.STDC import model_stage
from nets.lpsnet import get_lspnet_m
from nets.FFNet.ffnet_gpu_small import segmentation_ffnet50_dAAA
from nets.DDRNet.DDRNet_23_slim import get_seg_model
from nets.unet import SegHead

from utils.tools import Yam
# -----------------------------------------#
#           模块消融实验
# -----------------------------------------#
from nets.Albation_module import Albation

# Dice loss
from nets.criterion import Dice_loss, focal_loss, \
    Heatmaploss
from datasets.dataloader import convert
from datasets.boundary import convert_boundary

import cv2
import os
import torch
import argparse
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

from nets.Unet.Unet import UNet
from nets.UnetPlus.Unetplus import NestedUNet
from nets.ResUnet.ResUnet import Res50Unet
from nets.ResUnetPlus.ResUnetPlus import Res50UnetPlus

from nets.SETR.transformer_seg import SETRModel
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#    TransUnet
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
from nets.TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from nets.TransUnet.vit_seg_modeling import CONFIGS
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from nets.Swin_Unet.vision_transformer import SwinUnet
from nets.MIssFormer.MISSFormer import MISSFormer
from nets.TSFormer.TSFormer import TLGFormer as TSFormer
from nets.FHNet.TSFormer import TLGFormer as FHNet

from nets.GCN.torch_vertex import GraphSeg
from nets.GCN.GEDNet import GEDNet

# -----------------------------------------------------#
#           TODO?
# -----------------------------------------------------#
from nets.Deeplab.deeplab import DeepLab
from nets.DconnNet import DconnNet
from nets.HiFormer import HiFormer
from nets.HiFormer.configs import get_hiformer_b_configs

from nets.AHF import AHF_Fusion_U_Net
from nets.DAEFormer import daeformer
from nets.PSPnet import PSPnet

# -----------------------------------------------------#
from nets.FTMNet import FTMNet

from nets.Unet.FUnet import FUNet

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#                Mamba
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++#
from nets.Mamba import vision_mamba
from nets.vmunetv2 import vmunetv2


class Grad_CAM(object):
    def __init__(self, modules = None,
                 pth_path = None,
                 num_classes = 2,
                 args = None):

        assert pth_path != None, '请输入权重...'
        assert args != None, '请输入配置....'

        self.args = args  # get varible

        method = args.method
        num_classes = args.classes
        input_shape = args.h

        if (method == 'SFGNet'):
            self.net = BuildRT(n_classes = num_classes)

        elif method == 'bisenetv1':
            self.net = bisenetv1.BiSeNetV1(num_classes)
        elif method == 'ablation':
            self.net = Albation(num_classes, args = args)

        elif method == 'bisenetv2':
            self.net = bisenetv2.BiSeNetV2(num_classes)

        elif method == 'pidnet':
            self.net = pidnet.get_pred_model(name = 'pidnet_s', num_classes = num_classes)

        elif method == 'ddrnet':
            self.net = get_seg_model(pretrained = False, num_classes = num_classes, augment = False)

        elif method == 'ffnet':
            self.net = segmentation_ffnet50_dAAA()

        elif method == 'stdc':
            self.net = model_stage.BiSeNet('STDCNet813', num_classes, use_boundary_8 = True)

        elif method == 'lpsnet':
            self.net = get_lspnet_m()

        elif method == 'unet':

            self.net = UNet(n_channels = 3, n_classes = num_classes, bilinear = True)

        elif method == 'unetplus':

            # double

            self.net = NestedUNet(num_classes = num_classes, deep_supervision = True)

        elif method == 'resunet':

            self.net = Res50Unet(pretrained = True, num_class = num_classes)

        elif method == 'resunetplus':

            self.net = Res50UnetPlus(pretrained = True, num_class = num_classes, deep_supervision = True)

        elif method == 'transunet':

            config_vit = CONFIGS[ 'R50-ViT-B_16' ]

            config_vit.n_classes = num_classes

            config_vit.n_skip = 3

            img_size = input_shape

            vit_patches_size = 16

            # if config_vit.vit_name.find('R50') != -1:

            #     config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

            self.net = ViT_seg(config_vit, img_size = img_size, num_classes = config_vit.n_classes).cuda()

            self.net.load_from(
                weights = np.load(r'./nets/TransUnet/checkpoints/imagenet21k+imagenet2012_R50+ViT-B_16.npz'))

            # from nets.USI import flops_parameters

            # flops_parameters(model,torch.randn(1,3,256,256))


        elif method == 'setr':

            # 没有预训练权重

            self.net = SETRModel(patch_size = (32, 32),

                                 in_channels = 3,

                                 out_channels = num_classes,

                                 hidden_size = 1024,

                                 sample_rate = 5,

                                 num_hidden_layers = 4,

                                 num_attention_heads = 16,

                                 decode_features = [ 512, 256, 128, 64 ])

        elif method == 'swin_unet':

            img_size = input_shape

            num_classes = num_classes

            # 配置文件

            config = {'img_size': img_size,

                      'embed_dim': 96,

                      'depths': [ 2, 2, 2, 2 ],

                      'num_heads': [ 3, 6, 12, 24 ],

                      'window_size': 7,

                      'patch_size': 4,

                      'in_chans': 3,  # 输入通道数

                      'mlp_ratio': 4.,

                      'qkv_bias': True,

                      'qk_scale': None,

                      'ape': False,

                      'drop_rate': 0.0,

                      'patch_norm': True,

                      'drop_path_rate': 0.2,

                      'use_checkpoint': False,

                      # 预训练权重

                      'pretrained_path': r"./nets/Swin_Unet/checkpoints/swin_tiny_patch4_window7_224.pth"

                      }

            yaml = Yam(config)

            # print(yaml.img_size)

            self.net = SwinUnet(config = yaml, img_size = img_size, num_classes = num_classes)

            self.net.load_from()  # 加载预训练模型

        elif method == 'missformer':  # MISSFormer

            self.net = MISSFormer(num_classes = num_classes, img_size = input_shape)

        elif method == 'tsformer':

            self.net = TSFormer(num_classes = num_classes, img_size = input_shape, pretrained = True)

        elif method == 'gcn':

            self.net = GEDNet(n_channels = 3, num_classes = num_classes)


        elif method == 'fhnet':

            ablation = args.is_ablation

            if ablation:

                backbone = args.backbone
                encoder_frame = args.encoder_frame
                is_mamba = args.is_mamba
                is_transformer = args.is_transformer
                mamba_version = args.mamba_version
                up_type = args.up_type
                is_net = args.is_net
                is_mix = args.is_mix
                is_low = args.is_low
                stage1 = args.stage1
                stage2 = args.stage2

                from nets.Alation_FHNet.TSFormer import TLGFormer as FHNet
                from nets.Alation_FHNet._seg import _Seg
                self.net = _Seg(pretrained = None,
                                num_class = num_classes,
                                backbone = backbone,
                                up_type = up_type,
                                is_net = is_net,
                                is_mix = is_mix,
                                is_low = is_low,
                                stage1 = stage1,
                                stage2 = stage2)

            else:

                self.net = FHNet(num_classes = num_classes, img_size = input_shape)

        elif method == 'deeplabv3':

            self.net = DeepLab(pretrain = True,

                               num_classes = num_classes)

        elif method == 'pspnet':

            self.net = PSPnet.PSPNet(num_classes = num_classes)

        elif method == 'dconnNet':

            self.net = DconnNet.DconnNet(num_class = num_classes)

        elif method == 'Hiformer':

            config = get_hiformer_b_configs()

            self.net = HiFormer.HiFormer(config = config,

                                         img_size = input_shape,

                                         n_classes = num_classes)

        elif method == 'ahfunet':

            self.net = AHF_Fusion_U_Net.UNetAFF(n_channels = 3,

                                                n_classes = num_classes)

        elif method == 'daeformer':

            self.net = daeformer.DAEFormer(num_classes = num_classes)

        elif method == 'vmunetv1':

            pretrained = r"./nets/Swin_Unet/checkpoints/swin_tiny_patch4_window7_224.pth"

            self.net = vision_mamba.VMUNet(input_channels = 3,

                                           num_classes = num_classes,

                                           load_ckpt_path = pretrained)

        elif method == 'vmunetv2':

            pretrained = r"./nets/Swin_Unet/checkpoints/swin_tiny_patch4_window7_224.pth"

            self.net = vmunetv2.VMUNetV2(input_channels = 3,

                                         num_classes = num_classes,

                                         deep_supervision = False,

                                         load_ckpt_path = pretrained)


        else:
            raise NotImplementedError('No exist Method!!!')

        # print(self.net)

        self.num_classes = num_classes

        self.input_shape = (args.h, args.w)  # direct resize

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        #       TODO? BUG?
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = torch.load(pth_path, map_location = self.device)

        if args.is_ablation:

            updates = {}
            miss = [ ]

            # TODO? 模型的参数
            model_state = self.net.state_dict()

            # TODO? 权重的参数
            # pth_state = model.items()

            # for k,v in model.items():
            #     print(k,np.shape(v))

            for k, v in model.items():
                k = k.replace('segmodel.', '')

                if (k in model_state.keys()) and (np.shape(v) == np.shape(model_state[ k ])):
                    updates[ k ] = v
                else:
                    miss.append(k)

            model_state.update(updates)
            print(f'===========Miss:{miss}=============')

            self.net.load_state_dict(model_state)



        else:
            self.net.load_state_dict(model)

        self.net = self.net.eval()

        getattr(self.net, modules).register_forward_hook(self.__register_forward_hook)
        getattr(self.net, modules).register_backward_hook(self.__register_backward_hook)

        self.modules = modules

        # 保存梯度信息
        self.input_grad = [ ]
        # 收集feature map
        self.output_grad = [ ]

        # 特征
        self.feature_grad = [ ]

    def __register_backward_hook(self,
                                 module,
                                 grad_in,

                                 grad_out):

        # print(len(grad_in), len(grad_out))

        self.input_grad.append(grad_out[ 0 ].detach().data.cpu().numpy())

    def __register_forward_hook(self,
                                module,
                                grad_in,
                                grad_out):
        self.output_grad.append(grad_out)
        self.feature_grad.append(grad_out.detach().data.cpu().numpy())

    def _get_cam(self, feature_map, grads):
        # -------------------------------------------------------#
        #                  feature_map: [c,h,w]
        #                  grads: [c,h,w]
        #                  return [h,w]
        # -------------------------------------------------------#
        cam = np.zeros(feature_map.shape[ 2: ], dtype = np.float32)
        alpha = np.mean(grads, axis = (2, 3))

        for ind, c in enumerate(alpha):
            cam += c[ ind ] * feature_map[ 0 ][ ind ].cpu().detach().numpy()

        heatmap = np.maximum(cam, 0)

        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) + 1e-8)

        heatmap = cv2.resize(heatmap, self.input_shape)

        return heatmap

    def __get_cam(self, feature_map, grads):
        # -------------------------------------------------------#
        #                  feature_map: [c,h,w]
        #                  grads: [c,h,w]
        #                  return [h,w]
        # -------------------------------------------------------#
        # cam = np.zeros(feature_map.shape[2:], dtype=np.float32)
        print(grads.shape)
        cam = np.mean(grads, axis = (0, 1))
        print(cam.shape)

        heatmap = np.maximum(cam, 0)

        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) + 1e-8)

        heatmap = cv2.resize(heatmap, self.input_shape)

        return heatmap

    def show_cam_to_image(self, image,
                          heatmap,
                          is_show = False,
                          is_write = False,
                          name = None):
        # heatmap = np.transpose(heatmap,(1,2,0))

        heatmap = np.array(heatmap * 255, np.uint8)

        heatmap = cv2.applyColorMap(heatmap,
                                    cv2.COLORMAP_JET)

        heatmap = np.float32(heatmap) / 255.

        image = np.transpose(image, (1, 2, 0))

        img = 0.4 * heatmap + 0.6 * np.array(image)

        # --------------------------------------------#
        #               clip pix value
        # --------------------------------------------#

        img = (img - np.min(img)) / np.max(img)

        img = np.uint8(img * 255)

        if is_show:
            plt.imshow(img[ :, :, ::-1 ])
            plt.show()
        if is_write:
            cv2.imwrite(f'{args.save_path}/{args.method}/{name}_cam.jpg', img, [ cv2.IMWRITE_JPEG_QUALITY, 100 ])

    def vis_pix(self, pred, label, name):
        # ---------------------------------#
        # 预测 真实
        # 真实
        # TP: 预测为真，实际为真
        # FP：预测为真，实际为假
        # TN：预测为假，实际是假
        # FN：预测为假，实际是正
        # ---------------------------------#

        h, w, *_ = pred.shape

        # print(np.unique(np.reshape(pred, (-1,))))
        # print(np.unique(np.reshape(label, (-1,))))

        mask = np.ones((h, w, 3), dtype = np.uint8)

        color = np.array([ 147, 208, 80,  # 前景 绿
                           110, 147, 208,  # 蓝色
                           250, 250, 250,  # 背景 白
                           255, 230, 153 ])  # 黄色

        color = color.reshape((-1, 3))

        color = color[ :, ::-1 ]

        mask[ (label == 1) & (pred == 1) ] = color[ 0 ]

        mask[ (label == 0) & (pred == 1) ] = color[ 1 ]

        mask[ (label == 0) & (pred == 0) ] = color[ 2 ]

        mask[ (label == 1) & (pred == 0) ] = color[ 3 ]

        # mask = Image.fromarray(mask)

        cv2.imwrite(f'{args.save_path}/{args.method}/{name}_pix_vis.png', mask)

    def forward(self, image,
                label,
                is_show = False,
                is_write = False,
                name = None,
                classes = 2,
                datasetname = 'ACDC'):

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        #   TODO? 可视化
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        image = image.resize(self.input_shape)
        label = label.resize(self.input_shape)

        if not os.path.exists(os.path.join(args.save_path, args.method)):
            os.makedirs(os.path.join(args.save_path, args.method))

        image.save(os.path.join(args.save_path, args.method, name + '.jpg'))
        # label.save(os.path.join(args.save_path,args.method,name+'gt1.jpg'))

        image = np.array(image, dtype = np.float32) / 255.
        image = np.transpose(image, (2, 0, 1))

        # 网络模型输入
        x = torch.from_numpy(image).float()
        x = x.unsqueeze(dim = 0).to(self.device)
        print('x:', x.shape, args.method)

        self.net.zero_grad()  # 清空梯度

        self.net = self.net.eval()

        method = self.args.method
        self.net = self.net.to(self.device)

        h, w = args.h, args.w

        up = nn.Upsample(size = (h, w),
                         mode = 'bilinear', align_corners = True)

        if method == 'Ours':
            (feat_out8, feat_out16, feat_out32), \
            (feat_out_sp2, feat_out_sp4, feat_out_sp8, feat_out_sp16), \
            (pred_loc2, pred_loc4) = self.net(x)
        # (feat_out8) = self.net(x)

        elif (method == 'bisenetv1') | (method == 'bisenetv2') | (method == 'ablation'):
            out = self.net(x)
            feat_out8 = out[ 0 ]
        elif (args.method == 'ddrnet') | (args.method == 'ffnet') | (args.method == 'lpsnet') \
                | (args.method == 'pidnet'):

            feat_out8 = self.net(x)

            if (args.method == 'pidnet'):
                feat_out8 = feat_out8[ 0 ]

            # output = up(output[0])
            # TODO ?

            feat_out8 = up(feat_out8)

        elif args.method == 'stdc':

            *output, boundary = self.net(x)

            feat_out8 = up(output[ 0 ])

        elif ((args.method) == 'resunet') \
                | (args.method == 'transunet') | (args.method == 'swin_unet') \
                | ((args.method == 'setr')) | (args.method == 'missformer') | (args.method == 'gcn') | \
                (args.method == 'deeplabv3') | (args.method == 'pspnet') | (args.method == 'daeformer') | \
                (args.method == 'Hiformer') | (args.method == 'ahfunet') | (args.method == 'vmunetv1') | (
                args.method == 'vmunetv2'):

            output = self.net(x)
            feat_out8 = F.softmax((output), dim = 1)

        elif (args.method == 'unet'):

            print('进入')
            output = self.net(x)
            feat_out8 = F.softmax((output), dim = 1)

        elif (args.method == 'ftmnet') | (args.method == 'funet'):

            output, *_ = self.net(x)
            feat_out8 = F.softmax((output), dim = 1)

        elif (args.method == 'tsformer') | (args.method == 'fhnet'):

            cseg, tseg, *c = self.net(x)

            feat_out8 = F.softmax((cseg), dim = 1)
            # OneHotlabel1 = F.softmax((tseg), dim = 1)


        elif (args.method == 'unetplus') \
                | (args.method == 'resunetplus'):

            output = self.net(x)
            feat_out8 = F.softmax((output[ 0 ]), dim = 1)



        else:
            raise NotImplementedError('没有该对比方法....')
            # ---------------------------------#
        #            损失函数定义
        # ---------------------------------#

        # b,c,h,w = feat_out8.shape
        # label = torch.ones((b,c,h,w),requires_grad = True).float()
        # print(output_main.shape,label.shape)

        # --------------------------------------------#
        #                   取值
        # --------------------------------------------#

        unmask_main = torch.argmax(feat_out8, dim = 1)

        unmask = (unmask_main)

        # ---------------------------------------------------------#
        #  TODO? 真实标签
        # ---------------------------------------------------------#

        if (classes == 2) | ((datasetname == 'kvasir-seg') | (datasetname == 'benign') | (datasetname == 'malignant')):

            label = np.array(label)
            print(np.unique(label.reshape(-1)))

            _label = np.zeros_like(label)

            # print(_label.shape)

            _label[ label == 0 ] = 0
            _label[ label > 150 ] = 1

            print(np.unique(_label.reshape(-1)))






        elif ((datasetname == 'Synpase') | (datasetname == 'ACDC')):

            if datasetname == 'Synpase':
                platte = np.array([ 0, 0, 0, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 0, 225, 0, 177, 0, 186, 227,
                                    111, 42, 151, 255, 188, 0 ])
            else:
                platte = np.array([ 0, 0, 0,  # 前景 黑色
                                    255, 0, 0,  # 蓝色
                                    0, 255, 0,  # 背景 红
                                    0, 0, 255,
                                    ])  # 黄色

            platte = platte.reshape((-1, 3))

            _label = np.zeros((h, w))

            for c, id in enumerate(platte):
                ind = np.array(id == label).sum(axis = -1)

                i = (ind == 3)

                _label[ i ] = c


        else:
            raise NotImplementedError("datasetname error!!!")

        # ----------------------------------------------------------#
        #   生成的标签
        # -----------------------------------------------------------#
        glabel = unmask.detach().cpu().numpy()[ 0 ]  # 取出

        T_mask = np.zeros((self.input_shape[ 1 ], self.input_shape[ 0 ], self.num_classes))

        # --------------------------------------#
        # # 两种构建one-hot编码形式
        # # --------------------------------------#
        for c in range(self.num_classes):
            T_mask[ glabel == c, c ] = 1
        T_mask = np.transpose(T_mask, (2, 0, 1))
        T_mask = convert(T_mask)  # dice label
        T_mask = torch.unsqueeze(T_mask, dim = 0)

        # heat_map2,boundary2 = convert_boundary(label,2)
        # heat_map4,boundary4 = convert_boundary(label,4)
        # _,boundary8 = convert_boundary(label,8)

        # heat_map2 = convert(np.transpose(heat_map2, (2, 0, 1)))
        # heat_map4 = convert(np.transpose(heat_map4, (2, 0, 1)))
        # heat_map2 = torch.unsqueeze(heat_map2, dim=0)
        # heat_map4 = torch.unsqueeze(heat_map4, dim=0)

        # boundary2 = convert(np.transpose(boundary2, (2, 0, 1)))
        # boundary4 = convert(np.transpose(boundary4, (2, 0, 1)))
        # boundary8 = convert(np.transpose(boundary8, (2, 0, 1)))

        # boundary2 = torch.unsqueeze(boundary2, dim=0)

        # boundary4 = torch.unsqueeze(boundary4, dim=0)
        # boundary8 = torch.unsqueeze(boundary8, dim=0)

        # glabel = torch.from_numpy(glabel).unsqueeze(dim=0).long()

        # heatloss = Heatmaploss() #没有改
        # criteror = nn.BCEWithLogitsLoss()
        # with torch.no_grad():
        #     # TODO 预测图
        #     self.vis(unmask.detach().cpu().numpy()[ 0 ], name + '_pred', datasetname)
        #     # label = np.array(label)

        #     # TODO 真值图
        #     self.vis(_label, name + '_gt', datasetname)

        #     # TODO 色差图
        #     if classes == 2:
        #         self.vis_pix(unmask.detach().cpu().numpy()[ 0 ], _label, name)

        # self.vis_entropy(feat_out8.detach().cpu(),name)

        # self.net = self.net.train()
        # ----------------------------------------------------------#
        #                 损失函数和训练时保持一致
        #          TODO?
        # ----------------------------------------------------------#
        # 边界自蒸馏损失
        #  _feat = up(feat_out_sp4)
        #  _kl_s = nn.MSELoss()(F.sigmoid(up(feat_out_sp2)), F.sigmoid(_feat))
        #
        #
        #  # 分割自蒸馏损失
        #  _seg_kl1 = nn.KLDivLoss()(F.log_softmax(feat_out16, dim=1), F.softmax(feat_out8, dim=1))
        #  _seg_kl2 = nn.KLDivLoss()(F.log_softmax(feat_out8, dim=1), F.softmax(feat_out16, dim=1))
        #  _seg_kl = (_seg_kl1 + _seg_kl2) * args.A
        #
        #
        #  #-------------------------------------------------------------------------------------#
        #  #  分割损失
        #  #-------------------------------------------------------------------------------------#
        #  #with torch.no_grad():
        #  alpha = (feat_out8 - T_mask).mean()
        #  alpha = alpha ** 2
        #
        #  segloss8 = Dice_loss(feat_out8, T_mask) + nn.CrossEntropyLoss()(feat_out8, glabel)
        #  segloss16 = Dice_loss(feat_out16, T_mask) + nn.CrossEntropyLoss()(feat_out16, glabel)
        #  segloss32 = Dice_loss(feat_out32, T_mask) + nn.CrossEntropyLoss()(feat_out32, glabel)
        #
        #  segloss = (alpha * segloss8 + (segloss16 + segloss32) + _seg_kl)
        #
        #
        #  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        #  # 边界损失
        #  # 标签
        # # with torch.no_grad():
        #  bate = (feat_out_sp2 - boundary2).mean() ** 2
        #  boundaryloss2 = heatloss(feat_out_sp2, boundary2)
        #  boundaryloss4 = heatloss(feat_out_sp4, boundary4)
        #  boundaryloss8 = criteror(feat_out_sp8, boundary8)
        #
        #  boundaryloss = (bate * boundaryloss2 + boundaryloss4 + boundaryloss8)
        #
        #
        #  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        #  # 定位损失
        #  loc2loss = focal_loss(pred_loc2, heat_map2, args.alpha, args.bate)
        #  loc4loss = focal_loss(pred_loc4, heat_map4, args.alpha, args.bate)
        #  locloss = (loc2loss + args.L * loc4loss)
        #

        loss1 = nn.CrossEntropyLoss()(feat_out8, torch.from_numpy(_label).long().unsqueeze(dim = 0).cuda())
        loss2 = Dice_loss(feat_out8, T_mask.cuda())
        Tloss = torch.exp(loss1 + loss2)
        Tloss.backward()

        # generate CAM
        grad = self.input_grad[ 0 ]

        fmap = self.output_grad[ 0 ]

        # fmap = self.feature_grad[0]

        cam = self._get_cam(fmap, grad)
        # cam = self.__get_cam(fmap,fmap)

        # show
        image = np.float32(image)

        self.show_cam_to_image(image, cam, is_show,
                               is_write, name)

        self.input_grad.clear()
        self.output_grad.clear()

    # TODO?
    def vis(self, mask, name, datasetname):
        # ------------------------------#
        #           可视化
        # ------------------------------#

        # TODO? 上颜色 二分类

        if ((datasetname == 'kvasir-seg') | (datasetname == 'benign') | (datasetname == 'malignant')):

            platte = [ 0, 0, 0, 255, 255, 255, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153,
                       250,
                       170, 30,
                       220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142,
                       0,
                       0, 70,
                       0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32 ]


        elif (datasetname == 'ACDC'):

            platte = [ 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255 ]

        elif (datasetname == 'Synpase'):

            platte = [ 0, 0, 0, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 0, 225, 0, 177, 0, 186, 227,
                       111, 42, 151, 255, 188, 0 ]

        for i in range(256 * 3 - len(platte)):
            platte.append(0)

        platte = np.array(platte, dtype = np.uint8)

        mask = np.array(mask, dtype = np.uint8)

        mask = Image.fromarray(mask).convert('P')

        mask.putpalette(platte)

        # mask.show()

        mask.save(f'{args.save_path}/{args.method}/{name}_vis.png')

    def prob2entropy(self, prob):

        b, c, h, w = prob.shape

        x = -torch.mul(prob, torch.log2(prob + 1e-20)) / np.log2(c)

        return x

    def vis_entropy(self,
                    prob,
                    name):
        # ----------------------------------------#
        #               熵值结果
        # ----------------------------------------#
        entropy = self.prob2entropy(F.softmax(prob, dim = 1))  #
        # entropy = F.softmax(prob,dim=1)

        heatmap = entropy.detach().cpu().numpy()

        # entropy_background = np.array(entropy[0, ...] * 255)
        # entropy_foreground = np.array(entropy[1, ...] * 255)
        # #
        # heatmap = (0.5 * entropy_background + 0.5 * entropy_foreground)

        heatmap = (heatmap - np.max(heatmap)) / (np.min(heatmap))

        heatmap = np.array(heatmap, np.uint8)
        entropy = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # cv2.imshow('entropy',entropy)

        cv2.imwrite(f'{args.save_path}/{args.method}/{name}_entropy.png', entropy)
        # cv2.waitKey(0)


if __name__ == '__main__':

    args = argparse.ArgumentParser(description = 'inference....')

    args.add_argument('--image_path', '--i', default = '', help = 'inference image path...')
    args.add_argument('--model_path', '--m', default = '', help = 'inference model path...')

    # -----------------------------------------------------------------------------------------#
    #            增加文件读取
    # -----------------------------------------------------------------------------------------#
    args.add_argument('--val_txt', '--vt', required = True, help = 'inference txt....')
    args.add_argument('--vis_layer', '--v', default = 'conv_out', help = 'vis layer...')
    args.add_argument('--is_show', action = 'store_true', help = 'Using vis image...')
    args.add_argument('--is_write', action = 'store_true', help = 'Using save vis...')
    args.add_argument('--save_path', default = 'vis', type = str,
                      help = 'save inferece result path(source domain ->  target domain)...', required = True)
    args.add_argument('--method', type = str, required = True)

    args.add_argument('--Filepath', type = str, required = True, help = 'inference file path...')
    args.add_argument('--datasetname', type = str, default = 'ACDC', help = 'datasetname: ACDC,Synpase,...')
    args.add_argument('--classes', type = int, default = 2)
    # ------------------------------------------------------------------------------------------#
    #                                文件夹推理
    # ------------------------------------------------------------------------------------------#
    args.add_argument('--dir', type = str, default = '')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #    h,w
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    args.add_argument('--h', type = int, default = 224, help = 'inference image h...')
    args.add_argument('--w', type = int, default = 224, help = 'inference image w...')

    #
    # 模块消融实验
    args.add_argument('--flow', '--f', action = 'store_true',
                      help = 'Multi-scale feature aggregation')
    args.add_argument('--output', '--out', action = 'store_true',
                      help = 'Feature modification')
    args.add_argument('--block', '--bl', action = 'store_true',
                      help = 'SDCM Block...')

    # 超参数
    args.add_argument('--A', type = float, default = 0.1,
                      help = 'Self-distillation segmentation alignment loss coefficient')

    args.add_argument('--L', type = float, default = 0.01,
                      help = 'Location loss factor')

    args.add_argument('--alpha', type = float, default = 0.05,
                      help = 'classifier...')
    args.add_argument('--bate', type = float, default = 0.5,
                      help = 'classifier...')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #       TODO? AFDSeg消融实验 模块消融
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    args.add_argument('--is_ablation', action = 'store_true',
                      help = 'using ablation!!!')
    args.add_argument('--is_net', action = 'store_true',
                      help = 'whether use net!!!')

    args.add_argument('--is_low', action = 'store_true',
                      help = 'whether use low')

    args.add_argument('--is_mix', action = 'store_true',
                      help = 'mix module')

    args.add_argument('--up_type', type = str, default = 'standard',
                      help = 'Upsample type!!!')

    args.add_argument('--encoder_frame', type = str, default = 'B1',
                      help = 'Need match is_transformer !!!')

    args.add_argument('--mamba_version', type = str, default = 'v1',
                      help = 'Need match is_mamba !!!')

    args.add_argument('--backbone', type = str, default = 'resnet50',
                      help = 'CNN Encoder type!!!')
    args.add_argument('--stage1', type = int, default = 10,
                      help = 'FFT filter stage1 Window size!!!')
    args.add_argument('--stage2', type = int, default = 8,
                      help = 'FFT filter stage2 Window size!!!')

    # Transformer 结构
    args.add_argument('--is_transformer', action = 'store_true', help = 'Transforer Encoder model')

    # Mamba 结构
    args.add_argument('--is_mamba', action = 'store_true', help = 'Mamba Encoder model')

    args = args.parse_args()

    # path = args.image_path

    model_path = args.model_path

    path = args.Filepath
    classes = args.classes

    datasetname = args.datasetname

    # ----------------------------------------------------------#
    #
    # ----------------------------------------------------------#
    # from glob import glob
    #
    # images = glob(f'{args.dir}\*.tiff')

    # TODO 数据读取

    train_file = open(args.val_txt, 'r')

    # TODO? 图像和标签
    images = [ ]
    labels = [ ]
    for line in train_file.readlines():
        if (datasetname == 'benign') | (datasetname == 'malignant'):
            ll = line.strip()
            index = ll.index('g ')

            imagename = ll[ :index + 1 ]
            maskname = ll[ index + 2: ]

            images.append(imagename)
            labels.append(maskname)

        else:
            splited = line.strip().split()
            images.append(splited[ 0 ])
            labels.append(splited[ 1 ])

    # 每个模型需要获取的moduel name都不一样
    cam = Grad_CAM(modules = args.vis_layer, pth_path = model_path, args = args)
    for ind in range(len(images)):
        imagename = os.path.join(path, images[ ind ])
        maskname = os.path.join(path, labels[ ind ])
        maskname = maskname.replace('images', 'labels')
        imagename = imagename.replace('\\', '/')
        maskname = maskname.replace('\\', '/')

        name = os.path.basename(imagename).split('.')[ 0 ]
        print(name)

        image = Image.open(imagename).convert('RGB')

        # ----------------------------------------------#
        #   TODO？ 标签图分为2分类和多分类
        # ----------------------------------------------#
        # 制作标签
        label = Image.open(maskname)  # .convert('L')

        # TODO? 增加类别信息和数据集
        cam.forward(image = image, label = label, is_show = args.is_show, is_write = args.is_write, name = name,
                    classes = classes, datasetname = datasetname)