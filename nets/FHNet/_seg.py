
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.getcwd())

from nets.backbone import resnet50
from nets.FTMNet import MixFreFeature, Attention
from nets.Transim.network import LowFR, DF

from nets.backbone import CBA


class UP(nn.Module):
    def __init__(self, inc, outc):
        super(UP, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.conv1 = nn.Conv2d(in_channels = inc, out_channels = outc, kernel_size = 3, padding = 1)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(in_channels = outc, out_channels = outc, kernel_size = 3, padding = 1)
     

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x1):
        # --------------------------#
        # x1降采样
        # --------------------------#
        outputs = torch.cat((x1, self.up(x)), dim = 1)
        outputs = self.conv1(outputs)
        outputs = self.relu1(outputs)

        atten = F.avg_pool2d(outputs, (1,1))
        atten = self.conv2(atten)
        atten = self.relu1(atten)


        atten = self.sigmoid(atten)
        feat_atten = torch.mul(outputs, atten)
        feat_out = feat_atten + outputs
        return feat_out




class _Seg(nn.Module):
    def __init__(self, pretrained = '/root/data1/Gasking_Segmentation/model_data/resnet50-19c8e357.pth',
                 num_class = 1 + 1):
        super(_Seg, self).__init__()
        self.backbone = resnet50(pretrained)
        for i in range(len(self.backbone)):
            setattr(self, f'layer{i + 1}', self.backbone[ i ])
        # 3072,1536,768,320
        # 2048 1024 512 256
        in_filter = [ 3072, 1536, 768, 320 ]
        out_filter = [ 1024, 512, 256, 64 ]

        for i in range(len(in_filter)):
            setattr(self, f'up{i + 1}', UP(in_filter[ i ], out_filter[ i ]))

        # long distance
        self.net = DF(inplanes = 64)
        
        self.freq1 = MixFreFeature(64)
        self.freq2 = MixFreFeature(256)
        self.df1 = LowFR(64)
        self.df3 = LowFR(512)

        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor = 2),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
            nn.ReLU()
        )

        self.out = nn.Conv2d(64, num_class, kernel_size = 3, padding = 1, bias = False)

        self.conv = CBA(k = 3,
                        c = 2048 + 64, outc = 2048,p = 1)
       # self.boundary = nn.Conv2d(64, 1, kernel_size = 3, padding = 1, bias = False)

    def forward(self, x):
        # --------------------------------#
        # 降采样
        # --------------------------------#
        feat1 = self.layer1(x)  # 第一层
        x1 = self.net(feat1)
        feat1 = self.freq1(feat1,10)
        feat1 = self.df1(feat1)
        feat2 = self.layer2(feat1)
        feat2 = self.freq2(feat2,8)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        feat5 = self.layer5(feat4)

        b, c, h, w = feat5.shape
        x1 = F.interpolate(x1, (h, w), mode = 'bilinear',
                           align_corners = True)

        out = torch.cat((x1, feat5), dim = 1)
        feat5 = self.conv(out)

        # --------------------------------#
        # 上采样
        # --------------------------------#

        up1 = self.up1(feat5, feat4)  # 16
        up2 = self.up2(up1, feat3)  # 8
        up2 = self.df3(up2)

        up3 = self.up3(up2, feat2)  # 4
        up4 = self.up4(up3, feat1)  # 2

        #boundary = self.boundary(up4)

        # --------------------------------#
        # output
        # --------------------------------#
        out = self.up_conv(up4)

        out = self.out(out)
        return out, up3, up2, up1

    def freeze_backbone(self):
        for mm in self.backbone:
            for m in mm.parameters():
                m.requires_grad = False

    def unfreeze_backbone(self):
        for mm in self.backbone:
            for m in mm.parameters():
                m.requires_grad = True


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    print(x.shape)
    net = _Seg(pretrained = False, num_class = 1 + 1)
    #out = net(x)

    from nets.USI import flops_parameters
    flops_parameters(net,(x,))
    # for o in out:
    #     print(o.shape)


