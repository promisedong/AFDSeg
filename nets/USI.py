import torch
import torch.nn as nn
from torch.nn import init
import math
def fuse(conv, bn):
    fused = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    fused = fused.to(device)

    # setting weights
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.size()))

    # setting bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.weight.size(0)).to(device)

   # print(w_bn.device,b_conv.device)
    b_conv = torch.mm(w_bn, b_conv.view(-1, 1)).view(-1)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fused.bias.copy_(b_conv + b_bn)

    return fused

class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward = 'slicing'):
        super().__init__()

        #
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias = False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[ :, :self.dim_conv3, :, : ] = self.partial_conv3(x[ :, :self.dim_conv3, :, : ])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [ self.dim_conv3, self.dim_untouched ], dim = 1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class ResCBA(nn.Module):
    def __init__(self,
                 inc,
                 outc,
                 ksize = 3,
                 stride = 1,
                 is_use_bn = True
                 ):
        super(ResCBA, self).__init__()

        # hidden_dim = int(inc * width) #2 times

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = inc,
                      out_channels = outc,
                      kernel_size = 1,
                      stride = stride,
                      bias = False),
            nn.BatchNorm2d(outc) if is_use_bn else nn.Identity(),
            nn.ReLU(inplace = True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = inc,
                      out_channels = outc,
                      kernel_size = ksize,
                      stride = stride,
                      padding = ksize // 2,
                      bias = False),
            nn.BatchNorm2d(outc) if is_use_bn else nn.Identity(),
            nn.ReLU(inplace = True)
        )

        self.pw1 = PW(in_planes = inc, div = 4)
        self.conv1 = nn.Conv2d(in_channels = inc,
                               out_channels = outc,
                               kernel_size = ksize,
                               stride = stride,
                               padding = ksize // 2,
                               bias = False)

        self.conv2 = nn.Conv2d(in_channels = inc,
                               out_channels = outc,
                               kernel_size = ksize,
                               stride = stride,
                               padding = (ksize // 2),
                               bias = False)

        self.out = nn.Conv2d(in_channels = 2 * outc,
                             out_channels = outc,
                             kernel_size = 1,
                             bias = False)

    def forward(self, x):
        # through
        out = self.layer1(x)
        #out = nn.functional.relu(fuse(self.layer1[0],self.layer1[1])(x),inplace=True)
        #out = nn.functional.relu(self.layer1[0](x),inplace=True)

        out1 = self.layer2(x)
        #out1 = nn.functional.relu(fuse(self.layer2[0], self.layer2[1])(x), inplace=True)
        #out1 = nn.functional.relu(self.layer2[0](x),inplace=True)

        out = (out + out1)
        # right
        rout1 = (self.conv1(self.pw1(x)))

        rout2 = self.conv2(x)

        rout = self.out(torch.cat((rout1, rout2), dim = 1))
        out = (out + rout)
        return out


# ------------------------------------------------#
#               处理冗余特征
# ------------------------------------------------#
class PW(nn.Module):
    def __init__(self, in_planes, div, kernel = 3, stride = 1, width = 1.5):
        super(PW, self).__init__()

        assert (div) > 0, 'in_planes > 1!!!'

        self.in_planes = in_planes // 4

        self.conv1 = nn.Conv2d(in_channels = self.in_planes, out_channels = self.in_planes, kernel_size = kernel,
                               stride = stride, padding = kernel // 2, bias = False)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = in_planes,
                      out_channels = int(width * in_planes),
                      kernel_size = 1),
            nn.BatchNorm2d(int(width * in_planes)),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = int(width * in_planes),
                      out_channels = in_planes,
                      kernel_size = 1)
        )

        self.conv3 = nn.Conv2d(in_channels = in_planes, out_channels = in_planes, kernel_size = kernel,
                               stride = stride, padding = kernel // 2, bias = False)

        self.out = nn.Parameter(torch.randn(in_planes, ))

    def forward(self, x):
        # _,c,_,_ = x.shape
        # out = (torch.randn(c, ))
        index = torch.argsort(self.out, dim = 0, descending = True).contiguous()

        gt_index = index[ :self.in_planes ]

        umask = index[ self.in_planes: ]

        x1 = x[ :, gt_index, :, : ]  # 获取需要进行修改的特征

        unmask = x[ :, umask, :, : ]

        x1 = self.conv1(x1)


        # TODO
        x2 = torch.cat((x1, unmask), dim = 1)
        x2 = self.conv2(x2)


        # x2 = nn.functional.relu(fuse(self.conv2[0],self.conv2[1])(x2),inplace=True)
        # x2 = self.conv2[-1].forward(x2)

        # x2 = nn.functional.relu(self.conv2[0](x2), inplace=True)
        # x2 = self.conv2[-1](x2)

        x2 = self.conv3(x2) + x

        return x2


class CBA(nn.Module):
    def __init__(self, in_planes, out_planes, kernel = 3, stride = 1):
        super(CBA, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel, stride = stride,
                      padding = kernel // 2, bias = False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace = True)
        )

        # self.pw = nn.Sequential(Partial_conv3(out_planes, n_div = 4),
        #                         nn.BatchNorm2d(out_planes),
        #                         nn.ReLU(inplace = True)
        #                         )

    def forward(self, x):
        # x = fuse(self.layer[0],self.layer[1])(x)
        # return nn.functional.relu(x,inplace=True)
        return (self.layer(x))

        # x = nn.functional.relu(self.layer[0](x))
        # return  x



class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num = 3, stride = 1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size = 3, stride = 2, padding = 1,
                          groups = out_planes // 2, bias = False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size = 3, stride = 2, padding = 1, groups = in_planes,
                          bias = False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size = 1, bias = False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(CBA(in_planes, out_planes // 2, kernel = 1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(CBA(out_planes // 2, out_planes // 2, stride = stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(CBA(out_planes // 2, out_planes // 4, stride = stride))
            elif idx < block_num - 1:
                self.conv_list.append(CBA(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(CBA(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = [ ]
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim = 1) + x


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num = 3, stride = 1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            # 分组卷积 下采样
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size = 3, stride = 2, padding = 1,
                          groups = out_planes // 2, bias = False),
                nn.BatchNorm2d(out_planes // 2),
            )

            # 跳跃连接
            self.skip = nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1)
            stride = 1

        #
        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(CBA(in_planes, out_planes // 2, kernel = 1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(CBA(out_planes // 2, out_planes // 2, stride = stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(CBA(out_planes // 2, out_planes // 4, stride = stride))

            # 不同的通道输出
            elif idx < block_num - 1:
                self.conv_list.append(CBA(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(CBA(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = [ ]
        out1 = self.conv_list[ 0 ](x)

        for idx, conv in enumerate(self.conv_list[ 1: ]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim = 1)
        return out


# STDC2Net
class SCOT(nn.Module):
    # 2 2 2
    # 4 5 3
    #
    def __init__(self, base = 32, layers = [4,5,1], block_num = 4, type = "cat", num_classes = 1000, dropout = 0.20,
                 pretrain_model = '', use_conv_last = False):
        super(SCOT, self).__init__()

        # 类型
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck

        # 最后是否使用卷积
        self.use_conv_last = use_conv_last

        # build
        self.features = self._make_layers(base, layers, block_num, block)

        # 分类识别
        self.conv_last = ResCBA(base * 16, max(1024, base * 16), ksize = 1, stride = 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base * 16), max(1024, base * 16), bias = False)
        self.bn = nn.BatchNorm1d(max(1024, base * 16))
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(p = dropout)
        self.linear = nn.Linear(max(1024, base * 16), num_classes, bias = False)

        # 直接下采样
        self.x2 = nn.Sequential(self.features[ :1 ])
        self.x4 = nn.Sequential(self.features[ 1:2 ])

        self.x8 = nn.Sequential(self.features[ 2:6 ])
        self.x16 = nn.Sequential(self.features[ 6:11 ])
        #self.x32 = nn.Sequential(self.features[ 11: ])

    # 构造模型 resnet
    def _make_layers(self, base, layers, block_num, block):
        features = [ ]

        # downsample
        features += [ CBA(3, base // 2, 3, 2) ]
        features += [ ResCBA(base // 2, base, 3, 2) ]

        #
        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(block(base * int(math.pow(2, i + 1)), base * int(math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(block(base * int(math.pow(2, i + 2)), base * int(math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x):

        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        #feat32 = self.x32(feat16)

        # if self.use_conv_last:
        #     feat32 = self.conv_last(feat32)

        return feat2, feat4, feat8, feat16#, feat32

    def forward_impl(self, x):
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        # out = self.bn(out)
        out = self.relu(out)
        # out = self.relu(self.bn(self.fc(out)))
        out = self.dropout(out)
        out = self.linear(out)
        return out


def flops_parameters(model = None,
                     input = None,
                     verbose = False):
    from thop import profile
    flops, para = profile(model, input, verbose = verbose)

    f = '{:.3f}G'.format(flops / (1e9))
    p = '{:.3f}M'.format(para / (1e6))

    print(f'FLOPs: {f}')
    print(f'Para: {p}')
    return f, p


if __name__ == "__main__":


    model = SCOT().cuda()
    x = torch.randn((1, 3, 512, 512)).cuda()

    out = model(x)

    for o in out:
        print(o.shape)

    flops_parameters(model, (x,))

