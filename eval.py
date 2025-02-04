import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import copy
import time

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

# -------------------------------------------------------------#
#
# -------------------------------------------------------------#
from nets.Unet.Unet import UNet
from nets.UnetPlus.Unetplus import NestedUNet
from nets.ResUnet.ResUnet import Res50Unet
from nets.ResUnetPlus.ResUnetPlus import Res50UnetPlus

from utils.tools import Yam
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
from nets.FTMNet import FTMNet
from nets.Unet.FUnet import FUNet
from nets.FHNet.TSFormer import TLGFormer as FHNet
from nets.GCN.torch_vertex import GraphSeg
from nets.GCN.GEDNet import GEDNet



#-----------------------------------------------------#
#           TODO?
#-----------------------------------------------------#
from nets.Deeplab.deeplab import DeepLab
from nets.DconnNet import DconnNet
from nets.HiFormer import HiFormer
from nets.HiFormer.configs import get_hiformer_b_configs

from nets.AHF import AHF_Fusion_U_Net
from nets.DAEFormer import daeformer
from nets.PSPnet import PSPnet

#-----------------------------------------------------#
from nets.FTMNet import FTMNet

from nets.Unet.FUnet import FUNet
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#


# -------------------------------------------------------------#
#  模块消融实验
# -------------------------------------------------------------#
from nets.Albation_module import Albation
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
from torch.utils.data import DataLoader
from datasets.dataloader import BuildDataset, collate_seg
from utils.utils_eval import Eval
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



#++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#                Mamba
#++++++++++++++++++++++++++++++++++++++++++++++++++++++#
from nets.Mamba import vision_mamba
from nets.vmunetv2 import vmunetv2

# 云的掩码
Cloud = np.array([
    [ 255, 255, 255 ],
    [ 50, 120, 90 ]
]) / 255.


def plot_(fig,
          data,
          label,
          name,
          mode = None):
    x_min, x_max = np.min(data, 0), np.max(data, 0)

    data = (data - x_min) / (x_max - x_min)  # normalization

    color = {
        'source': {1: '#E72929', 0: '#41C9E2'},
        'target': {1: '#5755FE', 0: '#FDA403'}
    }

    # color = {
    #     'source':  'r',
    #     'target':  'b'
    # }
    # map_color = {i: color[ i ] for i in range(2)}
    map_size = {i: 10 for i in range(2)}

    # color = list(map(lambda x: map_color[ x ], label))
    size = list(map(lambda x: map_size[ x ], label))

    # print(len(color),len(size),data.shape)
    #

    l = data.shape[ 0 ]
    # print('start draw...')
    # for ind in range(l):
    #     plt.scatter(data[ind,0],data[ind,1],c = color[mode][label[ind]])
    # 以颜色代表特征
    # print('start draw...')
    # plt.scatter(data[ :, 0 ], data[ :, 1 ])

    # 以数字代表特征

    print('start draw...')
    for ind in range(l):
        plt.text(data[ ind, 0 ], data[ ind, 1 ], str(int(label[ ind ])), color = color[ mode ][ label[ ind ] ],
                 fontdict = {'size': 8})

    # plt.show()

    return fig


def draw(feature, label, name, mode = None,
         form_mode = None, fig = None, tsne = None):
    b, c, h, w = feature.shape

    feature = feature.permute((0, 2, 3, 1))
    # label = label.permute((1, 2, 0))

    feature = feature.reshape((-1, c))
    label = label.reshape((-1,))

    TSNE_result = tsne.fit_transform(feature.data.cpu().numpy())

    # print(f'TSNE: {TSNE_result.shape}')
    TSNE_label = label.data.cpu().numpy()

    fig = plot_(fig, TSNE_result, TSNE_label, name, form_mode)
    return fig


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser('eval')

    parse.add_argument('--method', type = str,
                       required = True, help = 'model...')
    parse.add_argument('--num_classes', type = int, default = 8,
                       help = 'classes...')
    parse.add_argument('--model_path', type = str, required = True,
                       help = 'infer model...')
    parse.add_argument('--datasetname', type = str, required = True,
                       help = 'datasetname...')
    parse.add_argument('--val_file', type = str, required = True,
                       help = 'val_file...')

    # 测试集
    parse.add_argument('--h', type = int, default = 512,
                       help = 'inference image h...')
    parse.add_argument('--w', type = int, default = 512,
                       help = 'inference image w...')

    # 精度记录
    parse.add_argument('--record', '--r', default = 'log.txt', help = 'record accuracy....')

    # 模块消融实验
    parse.add_argument('--flow', '--f', action = 'store_true',
                       help = 'Multi-scale feature aggregation')
    parse.add_argument('--output', '--out', action = 'store_true',
                       help = 'Feature modification')
    parse.add_argument('--block', '--bl', action = 'store_true',
                       help = 'SDCM Block...')

    args = parse.parse_args()

    method = args.method
    class_num = args.num_classes

    record = args.record

    # 记录数据
    logger = open(record, 'a+')

    model_path = args.model_path
    val_file = args.val_file
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if (method == 'Ours'):
        model = BuildRT(n_classes = class_num)

    elif method == 'bisenetv1':
        model = bisenetv1.BiSeNetV1(class_num)
    elif method == 'ablation':
        model = Albation(class_num, args = args)

    elif method == 'bisenetv2':
        model = bisenetv2.BiSeNetV2(class_num)

    elif method == 'pidnet':
        model = pidnet.get_pred_model(name = 'pidnet_s', num_classes = class_num)

    elif method == 'ddrnet':
        model = get_seg_model(pretrained = False, num_classes = class_num, augment = False)

    elif method == 'ffnet':
        model = segmentation_ffnet50_dAAA()

    elif method == 'stdc':
        model = model_stage.BiSeNet('STDCNet813', class_num, use_boundary_8 = True)

    elif method == 'lpsnet':
        model = get_lspnet_m()


    elif method == 'unet':

        model = UNet(n_channels = 3, n_classes = class_num, bilinear = True)


    elif method == 'unetplus':

        # double

        model = NestedUNet(num_classes = class_num, deep_supervision = True)


    elif method == 'resunet':

        model = Res50Unet(pretrained = True, num_class = class_num)


    elif method == 'resunetplus':

        model = Res50UnetPlus(pretrained = True, num_class = class_num, deep_supervision = True)


    elif method == 'transunet':

        config_vit = CONFIGS[ 'R50-ViT-B_16' ]

        config_vit.n_classes = class_num

        config_vit.n_skip = 3

        img_size = args.h

        vit_patches_size = 16

        # if config_vit.vit_name.find('R50') != -1:

        #     config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

        model = ViT_seg(config_vit, img_size = img_size, num_classes = config_vit.n_classes).cuda()

        # model.load_from(weights = np.load(r'./nets/TransUnet/checkpoints/imagenet21k+imagenet2012_R50+ViT-B_16.npz'))


    elif method == 'setr':

        # 没有预训练权重

        model = SETRModel(patch_size = (32, 32),

                          in_channels = 3,

                          out_channels = class_num,

                          hidden_size = 1024,

                          sample_rate = 5,

                          num_hidden_layers = 4,

                          num_attention_heads = 16,

                          decode_features = [ 512, 256, 128, 64 ])


    elif method == 'swin_unet':

        img_size = args.h

        num_classes = class_num

        # 配置文件

        config = {'img_size': img_size,

                  'embed_dim': 96,

                  'depths': [ 2, 2, 2, 2 ],

                  'num_heads': [ 3, 6, 12, 24 ],

                  'window_size': 8,

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

        model = SwinUnet(config = yaml, img_size = img_size, num_classes = num_classes)

        # net.load_from()  # 加载预训练模型


    elif method == 'missformer':  # MISSFormer

        image_size = args.h
        model = MISSFormer(num_classes = class_num, img_size = image_size)


    elif method == 'tsformer':
        image_size = args.h
        model = TSFormer(num_classes = class_num, img_size = image_size)
    elif method == 'ftmnet':
        image_size = args.h
        model = FTMNet(image_size, class_num)

    elif method == 'funet':

        model = FUNet(n_channels = 3, n_classes = class_num)


    elif method == 'fhnet':
        model = FHNet(num_classes = class_num,
                      img_size = args.h)


    elif method == 'gcn':
        model = GEDNet(n_channels = 3, num_classes = class_num)
    elif method == 'ftmnet':

        image_size = args.h
        model = FTMNet(image_size, class_num)

    elif method == 'funet':
        model = FUNet(n_channels = 3, n_classes = class_num)


    elif method == 'fhnet':
        image_size = args.h
        model = FHNet(num_classes = class_num, img_size = image_size)

    elif method == 'deeplabv3':
        model = DeepLab(pretrain = True,
                        num_classes = class_num)

    elif method == 'pspnet':
        model = PSPnet.PSPNet(num_classes = class_num)

    elif method == 'dconnNet':
        model = DconnNet.DconnNet(num_class = class_num)

    elif method == 'Hiformer':
        image_size = args.h
        config = get_hiformer_b_configs()
        model = HiFormer.HiFormer(config = config,
                                  img_size = image_size,
                                  n_classes = class_num)

    elif method == 'ahfunet':
        model = AHF_Fusion_U_Net.UNetAFF(n_channels = 3,
                                         n_classes = class_num)
    elif method == 'daeformer':
        model = daeformer.DAEFormer(num_classes = class_num)

    elif method == 'vmunetv1':
        pretrained = None
        model = vision_mamba.VMUNet(input_channels = 3,
                                    num_classes = class_num,
                                    load_ckpt_path = pretrained)

    elif method == 'vmunetv2':
        pretrained = None
        model = vmunetv2.VMUNetV2(input_channels = 3,
                                  num_classes = class_num,
                                  deep_supervision = False,
                                  load_ckpt_path = pretrained)

    else:

        raise NotImplementedError('No exist Method!!!')

    # model = torch.nn.DataParallel(model)

    seg = torch.load(model_path, device)

    model.load_state_dict(seg)

    model = model.to(device)

    model.eval()

    # 类别数量
    evaltor = Eval(class_num)

    valdata = BuildDataset(train_file = val_file, input_shape = (args.h, args.w))

    valdataloader = DataLoader(valdata,
                               batch_size = 1,
                               num_workers = 0,
                               shuffle = False,
                               collate_fn = collate_seg
                               )
    lists = [ ]  # 收集
    print(len(valdataloader))

    savedir = rf'Vis/{args.datasetname}/{method}'

    if os.path.exists(savedir):
        import shutil

        shutil.rmtree(savedir)

    os.makedirs(savedir)

    print("start Segmentation...")
    is_show = False

    # 扩充数据集
    up = nn.Upsample(size = (args.h, args.w),
                     mode = 'bilinear', align_corners = True)
    # tsne = TSNE(n_components=2,  random_state=0, perplexity=5,early_exaggeration=35,learning_rate='auto')
    for ind, batch in enumerate(valdataloader):
        with torch.no_grad():
            orimgs, image, label, (png, points, edges, recons, cdfs), (heat2, heat4), (
            (boundary1, boundary2), boundary4, boundary8, boundary16) = batch

            # orimgs,image, label,( png,_,_,_,_,_), (heat2, heat4), \
            #     ((boundary1, boundary2), boundary4, boundary8, boundary16) = batch

            image = image.to(device)
            png = png.to(device)
            boundary1 = boundary1.float().to(device)
            boundary2 = boundary2.float().to(device)
            boundary4 = boundary4.float().to(device)
            boundary8 = boundary8.float().to(device)
            boundary16 = boundary16.float().to(device)
            label = label.float().to(device)

            # ------------------------------------------------------------------------------------#
            h, w = args.h // 4, args.w // 4
            # _label = F.interpolate(label, size=(h, w), mode='bilinear', align_corners=True)
            # _label = torch.randn((1,2,h,w)).float().to(device)
            _label = F.interpolate(label, size = (h, w), mode = 'bilinear', align_corners = True)
            _label = torch.randn_like(_label)

            gt2 = up(boundary2)

            boundary2 += torch.randn_like(boundary2)
            boundary4 += torch.randn_like(boundary4)
            boundary8 = torch.randn_like(boundary8)
            boundary16 = torch.randn_like(boundary16)

            if args.method == 'Ours':
                t0 = time.time()
                (feat_out8, feat_out16, feat_out32), \
                (feat_out_sp2, feat_out_sp4, feat_out_sp8, feat_out_sp16), \
                (loc2, loc4) = model(image)  # model(image, _label,*(boundary2, boundary4, boundary8, boundary16))

                # pred_loc2 = F.sigmoid(pred_loc2)
                # pred_loc4 = F.sigmoid(pred_loc4)
                #
                #
                # #feat_out_sp2 = F.sigmoid(feat_out_sp2)
                #
                # keep = F.max_pool2d(pred_loc2,kernel_size=3,stride=1,padding=1)
                #
                # #print(torch.max(keep),torch.min(keep))
                #
                # keep = (pred_loc2 == keep).float()
                #
                # #print(torch.max(keep), torch.min(keep))
                #
                # pred_loc2 *= keep
                #
                # #pred_loc2 = pred_loc4
                #
                # ksize = np.array([ -1, -1, -1, -1, 8, -1, -1, -1, -1 ]).reshape((1, 1, 3, 3))
                # ksize = torch.from_numpy(ksize).float().to(device)
                #
                # #pred_loc2 = F.sigmoid(F.conv2d(pred_loc2,weight = ksize,stride=1,padding=1))
                #
                #
                #
                # #feat_out_sp2 = F.conv2d(feat_out_sp2, ksize, padding = 1)
                # feat_out_sp2 = F.sigmoid(feat_out_sp2)
                # pred_loc2 = up(pred_loc2)
                # feat_out_sp2 = up(feat_out_sp2)
                # pred_loc2 = (pred_loc2 > 0.7).float()

                # feat_out8 *= pred_loc2

                # feat_out8 = (feat_out8 + feat_out8  * (feat_out_sp2)  + (1 - feat_out_sp2))

                # feat_out8 = F.softmax(feat_out8, dim=1)
                # TODO 边界
                # OneHotlabel = feat_out8  * (1 - feat_out_sp2) + feat_out8 + feat_out_sp2
                # OneHotlabel = torch.sigmoid(OneHotlabel)
                # OneHotlabel = feat_out8

                # feat_out8 = feat_out8 * ( 1- pred_loc2) + feat_out8 + pred_loc2

                OneHotlabel = F.softmax(feat_out8, dim = 1)
                print(1)
                # OneHotlabel1 = F.softmax(feat_out16, dim = 1)
                # OneHotlabel2 = F.softmax(feat_out32, dim = 1)
                #
                # from torchvision.utils import save_image
                #
                # save_image(pred_loc2 * 255, f'{savedir}/result_2_{ind}.png')
                # save_image(pred_loc4, f'{savedir}/result_4_{ind}.png')
                # save_image(feat_out_sp2 *255, f'{savedir}/result_boundary_{ind}.png')
                # save_image(gt2,f'{savedir}/result_boundary_gt_{ind}.png')
            elif (method == 'bisenetv1') | (method == 'bisenetv2') | (method == 'ablation'):
                print(2)
                t0 = time.time()
                out = model(image)
                OneHotlabel = F.softmax((out[ 0 ]), dim = 1)
            elif (args.method == 'ddrnet') | (args.method == 'ffnet') | (args.method == 'lpsnet') \
                    | (args.method == 'pidnet'):

                print(3)
                t0 = time.time()
                output = model(image)

                if (args.method == 'pidnet'):
                    output = output[ 0 ]

                # output = up(output[0])
                # TODO ?

                OneHotlabel = F.softmax(up(output), dim = 1)

            elif args.method == 'stdc':
                print(4)
                t0 = time.time()
                *output, boundary = model(image)

                OneHotlabel = F.softmax(up(output[ 0 ]), dim = 1)


            elif ((args.method) == 'resunet') \
                    | (args.method == 'transunet') | (args.method == 'swin_unet') \
                    | ((args.method == 'setr')) | (args.method == 'missformer')| (args.method == 'gcn')|\
                    (args.method == 'deeplabv3')|(args.method == 'pspnet')|(args.method == 'daeformer')|\
                    (args.method == 'Hiformer')|(args.method == 'ahfunet'):

                output = model(image)
                OneHotlabel = F.softmax((output), dim = 1)

            elif ((args.method) == 'unet'):

                output = model(image)
                OneHotlabel = F.softmax((output), dim = 1)
                print(f"6: {OneHotlabel.shape}")
            elif (args.method == 'ftmnet') | (args.method == 'funet'):

                output, *_ = model(image)
                OneHotlabel = F.softmax((output), dim = 1)

            elif (args.method == 'tsformer') | (args.method == 'fhnet'):

                cseg, tseg, *c = model(image)

                OneHotlabel = F.softmax((cseg), dim = 1)
                # OneHotlabel1 = F.softmax((tseg), dim = 1)


            elif (args.method == 'unetplus') \
                    | (args.method == 'resunetplus'):

                output = model(image)
                OneHotlabel = F.softmax((output[ 0 ]), dim = 1)


            # TODO? add mamba
            elif (args.method == 'unetplus'):
                pass

            else:
                raise NotImplementedError('没有该对比方法....')

            OneHotlabel = torch.argmax(OneHotlabel, dim = 1)

            t1 = time.time()

            # print(f'FPS: {1/((t1-t0))}')
            # OneHotlabel1 = torch.argmax(OneHotlabel1, dim = 1)

            # OneHotlabel = (OneHotlabel + OneHotlabel1)
            # OneHotlabel2 = torch.argmax(OneHotlabel2, dim = 1)

            if is_show:

                fig = plt.figure()

                ax = plt.gca()

                # draw(pred, OneHotlabel.long(), name = targetname[0],mode= mode)  # 可视化
                # tpred = torch.cat((tpred,spred),dim=1)
                # targetOneHotlabel = torch.cat((targetOneHotlabel,sourceOneHotlabel),dim=1)
                fig = draw(tpred, targetOneHotlabel, name = targetname[ 0 ], mode = mode, form_mode = 'target',
                           fig = fig, tsne = tsne)  # 可视化
                draw(spred, sourceOneHotlabel, name = targetname[ 0 ], mode = mode, form_mode = 'source', fig = fig,
                     tsne = tsne)

                # plt.xticks([-0.5,0.5])
                #
                # plt.yticks([-0.5,0.5])
                #
                # plt.title(targetname[0])

                # save = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + f'_{mode}.tif'

                ax.spines[ 'right' ].set_visible(False)
                ax.spines[ 'left' ].set_visible(False)
                ax.spines[ 'top' ].set_visible(False)
                ax.spines[ 'bottom' ].set_visible(False)

                plt.xticks([ ])
                plt.yticks([ ])

                save = targetname[ 0 ][ targetname[ 0 ].rindex('\\') + 1: ]

                save = save.split('.')[ 0 ]

                plt.savefig(os.path.join(savedir, f'{save}_{mode}.tif'), dpi = 150, bbox_inches = 'tight')

                print("Generate TSNE finined...")

            else:
                OneHotlabel = OneHotlabel.cpu().numpy()
                # OneHotlabel1 = OneHotlabel1[ 0 ].cpu().numpy()
                # OneHotlabel2 = OneHotlabel2[ 0 ].cpu().numpy()

                # OneHotlabel = (OneHotlabel | OneHotlabel1 | OneHotlabel2)

                png = png.cpu().numpy()

                # print(targetmap.shape,targetpng.shape)

                evaltor.init(OneHotlabel, png)

                image = image[ 0 ].detach().cpu().numpy().transpose((1, 2, 0))
                # evaltor.vis(image, OneHotlabel, str(ind), savedir)
                # print(image.shape,png.shape)
                evaltor.vis(image, png[ 0 ], str(ind), savedir)
    res = evaltor.get_res()

    res = f'{method}_{args.datasetname}: {res}\n'
    logger.write(res)
    evaltor.show()







