import torch
import math
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from nets.Transim.network import LowFR,DF
class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan1,in_chan2,out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        


    def forward(self, fsp, fcp):
        
        return fsp

class FeedForWard(nn.Module):
    def __init__(self,
                 dim,
                 hidden_dim,
                 drop = 0.5):
        super(FeedForWard, self).__init__()

        self.project = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(drop)
        )


        self.conv = res_basic(inplanes = dim,outplanes = dim,
                              ksize = 1)

        self.out = nn.Conv2d(in_channels = dim * 2,out_channels = dim,
                             kernel_size = 3,padding = 1)


    def forward(self,x,H,W):

        B,N,C = x.shape
        tx = x.transpose(1,2).view(B,C,H,W)
        tx1 = self.conv(tx)

        tx2 = self.project(x).transpose(1,2).view(B,-1,H,W)



        tx = torch.cat((tx1,tx2),dim = 1)


        tx = self.out(tx).flatten(2).transpose(1,2)

        return tx


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 hidden_dim = 64,
                 heads = 8,
                 drop = 0.5):
        super(Attention, self).__init__()

        inner_dim = hidden_dim * heads

        project_out = not (heads == 1 and hidden_dim == dim)

        self.heads = heads

        self.scale = hidden_dim ** (-0.5)

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.drop = nn.Dropout(drop)


        self.out = nn.Sequential(
                   nn.Linear(inner_dim,dim),
                   nn.Dropout(drop)
        ) if project_out else nn.Identity()


    def forward(self,q,k):

        q = self.norm(q)
        k = self.norm(k)
        v = k



        out = F.softmax((q @ k.permute((0, 2, 1))) * self.dim, dim = -1)

        out = out @ v

        out = self.attn(out)
        # 线性层
        out = self.linear(out)
        x = self.g(out + x)

        return x
        return

class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size = 224, patch_size = 7, stride = 4, padding = 1, in_ch = 3, dim = 768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)
        _, _, H, W = px.shape
        # TODO?
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W

class EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias = True)
        self.kv = nn.Linear(dim, dim * 2, bias = True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(sp_x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[ 0 ], kv[ 1 ]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim = -1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out




class TransformerBlock(nn.Module):
    def __init__(self, dim, head, reduction_ratio = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)


        self.mlp = FeedForWard(dim,int(dim * 4))


    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        tx = x

        t1 = self.attn(self.norm1(x), H, W)

        tx = tx + t1
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx




class VisionFormer(nn.Module):
    def __init__(self,size,inplanes,outplanes,k = 3,stride = 1,depth = 2,heads = 4):
        super(VisionFormer, self).__init__()

        self.conv = OverlapPatchEmbeddings(img_size = size,
                                           in_ch = inplanes,
                                           patch_size = k,
                                           padding = k // 2,
                                           stride = stride,
                                           dim = outplanes)

        self.layer = nn.ModuleList([])



        for i in range(depth):

            self.layer.append(TransformerBlock(dim = outplanes ,head = heads))



    def forward(self,x):
        x,H,W = self.conv(x)


        for layer in self.layer:
            x = layer(x,H,W)


        return x

class CBA(nn.Module):
    def __init__(self, in_planes, out_planes, kernel = 3, stride = 1):
        super(CBA, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel, stride = stride,
                      padding = kernel // 2, bias = False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):

        return (self.layer(x))




class res_basic(nn.Module):
    def __init__(self,inplanes,outplanes,ksize = 1,stride = 1):
        super(res_basic, self).__init__()

        self.conv1 = CBA(in_planes = inplanes,out_planes = outplanes,
                         kernel = ksize,stride = stride)

        self.relu = nn.ReLU(inplace = True)

        self.bn = nn.BatchNorm2d(outplanes)

        self.conv2 = nn.Conv2d(in_channels = outplanes,
                               out_channels = outplanes,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1)

        self.bn1 = nn.BatchNorm2d(outplanes)

        self.conv3 = nn.Conv2d(in_channels = inplanes,
                               out_channels = outplanes,
                               kernel_size = 1,
                               stride = stride,
                               padding = 0)

        self.bn2 = nn.BatchNorm2d(outplanes)

        self.out = nn.Conv2d(in_channels = 2 * outplanes,
                               out_channels = outplanes,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0)



    def forward(self,x):

        x1 = x #
        x = self.conv1(x)
        x = self.bn1(self.conv2(x))
        x1 = self.relu(self.bn2(self.conv3(x1)))

        out = torch.cat((x1,x),dim = 1)

        return  self.out(out)


class MixFreFeature(nn.Module):
    def __init__(self,inplanes,alpha = 2):
        # TODO 下采样阶段高频信息易丢失
        super(MixFreFeature, self).__init__()

        



    def forward(self,x,alpha):

        return x



class Basic_Net(nn.Module):
    def __init__(self, size,inplanes, outplanes,
                 ksize = 1, stride = 1,depth = 2,heads = 4):
        super(Basic_Net, self).__init__()

        # self.conv = res_basic(inplanes = inplanes,
        #                       outplanes = outplanes,
        #                       ksize = ksize,
        #                       stride = stride)
        self.l = VisionFormer(size = size,
                              inplanes = inplanes,
                              outplanes = outplanes,
                              k = ksize,
                              stride = stride,
                              depth = depth,
                              heads = heads
                              )

        #self.out = CBA(in_planes = 2 * outplanes,out_planes = outplanes)
    def forward(self,x):

        #x1 = self.conv(x)

        x2 = self.l(x).contiguous().transpose(2,1)
        B, C, N = x2.shape
        n = int(math.sqrt(N))
        x2 = x2.view(B,C,n,n)

        #out = torch.cat((x1,x2),dim = 1)

        return x2#self.out(out)


class FTMNet(nn.Module):
    def __init__(self,size,num_classes):
        super(FTMNet, self).__init__()

        heads = [1,2,4,8]
        depths = [3, 6, 40, 3 ]
        inplanes = [ 64, 128, 320, 512 ]
        patch_sizes = [ 5, 3, 3, 3 ]
        strides = [ 2, 2, 2, 2 ]

        self.hidden_dim = 738 #TODO

        self.net = DF(inplanes = inplanes[0])


        self.layer1 = Basic_Net(size,inplanes = 3,
                                outplanes = inplanes[0],
                                ksize = patch_sizes[0],
                                stride = strides[0],
                                depth = depths[0],
                                heads = heads[0])


        self.layer2 = Basic_Net(size // 2, inplanes = inplanes[ 0 ],
                                outplanes = inplanes[ 1 ],
                                ksize = patch_sizes[ 1 ],
                                stride = strides[ 1 ],
                                depth = depths[ 1 ],
                                heads = heads[ 1 ])
        self.freq1 = MixFreFeature(inplanes[ 1 ])
        self.layer3 = Basic_Net(size // 4, inplanes = inplanes[ 1 ],
                                outplanes = inplanes[ 2 ],
                                ksize = patch_sizes[ 2],
                                stride = strides[ 2 ],
                                depth = depths[ 2],
                                heads = heads[ 2 ])
        self.freq2 = MixFreFeature(inplanes[ 2 ])
        self.layer4 = Basic_Net(size, inplanes = inplanes[ 2 ],
                                outplanes = inplanes[ 3],
                                ksize = patch_sizes[ 3 ],
                                stride = strides[ 3 ],
                                depth = depths[ 3 ],
                                heads = heads[ 3 ])

        self.UP = nn.Upsample(scale_factor = 2.,mode = 'bilinear',align_corners = True)
        self.fusion1 = FeatureFusionModule(in_chan1 = inplanes[-1],
                                          in_chan2 = inplanes[-2],
                                          out_chan = self.hidden_dim)

        self.fusion2 = FeatureFusionModule(in_chan1 = self.hidden_dim,
                                       in_chan2 = inplanes[ 1 ],
                                       out_chan = self.hidden_dim)

        self.fusion3 = FeatureFusionModule(in_chan1 = self.hidden_dim,
                                       in_chan2 = inplanes[ 0 ],
                                       out_chan = self.hidden_dim)

        # self.layer = nn.Conv2d(in_channels = inplanes[0],
        #                                     out_channels = inplanes[1],
        #                                     kernel_size = 3,
        #                                     padding = 1)

        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor = 2),
            nn.Conv2d(in_channels = 738, out_channels = 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
            nn.ReLU()
        )

        self.conv = nn.Conv2d( inplanes[0] + inplanes[3], inplanes[3], kernel_size = 3, padding = 1, bias = False)

        self.out = nn.Conv2d(64, num_classes, kernel_size = 3, padding = 1, bias = False)

        self.points = nn.Sequential(

            nn.Conv2d(in_channels = 738, out_channels = 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 3, padding = 1),
            nn.ReLU()
        )
        self.boundary = nn.Conv2d(64, 1, kernel_size = 3, padding = 1, bias = False)

    def forward(self,x):

        x1 = self.layer1(x)

        out = self.net(x1)

        x2 = self.layer2(x1)
        _,c,h,w = x2.shape
        # out1 = F.interpolate(x1,(h,w),mode = 'bilinear',align_corners = True)

        # x2 = x2 + self.layer(out1)

        x2 = self.freq1(x2)

        x3 = self.layer3(x2)
        x3 = self.freq2(x3)

        x4 = self.layer4(x3)

        b,c,h,w = x4.shape
        out = F.interpolate(out,(h,w),mode = 'bilinear',
                            align_corners = True)
        out = torch.cat((out,x4),dim = 1)
        x4 = self.conv(out)

        up1 = self.UP(x4)
        up1 = self.fusion1(up1,x3)
        up2 = self.UP(up1)
        up2 = self.fusion2(up2,x2)

        up3 = self.UP(up2)
        up3 = self.fusion3(up3,x1)

        out = self.up_conv(up3)
        pred = self.out(out)
        points = self.points(up3)
        boundary = self.boundary(out)
        return pred,boundary,points




if __name__ == '__main__':

    # x = torch.randn((1,32,64,64))
    #
    # model = MixFreFeature(inplanes = 32)
    #
    # out = model(x)
    #
    # print(out.shape)

    # x1 = torch.randn((1,32,64,64))
    # x2 = torch.randn((1,16,64,64))
    #
    # model = FeatureFusionModule(in_chan1 = 32,in_chan2 = 16,out_chan = 64)
    #
    # out = model(x1,x2)
    #
    # print(out.shape)

    x = torch.randn((1,3,128,128))
    model = FTMNet(size = 128,num_classes = 2)
    outs = model(x)

    for o in outs:
     print(o.shape)
    #torch.save(model.state_dict(),'last.pth')