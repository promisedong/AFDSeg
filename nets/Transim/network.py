import sys,os


sys.path.insert(0,os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.Transim.sample import CSDN_Tem,\
                                CSDN_Temd,Hist_adjust


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

class SE(nn.Module):
    def __init__(self,channels,ratio=16):
        super(SE, self).__init__()

        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))


        self.conv = nn.Conv2d(in_channels = channels * 2,
                              out_channels = channels,
                              kernel_size = 1)

        #self.alpha = nn.Parameter(torch.tensor(0.995),requires_grad = True)
        self.fc=nn.Sequential(
            nn.Linear(channels,channels//ratio,False),
            nn.ReLU(),
            nn.Linear(channels//ratio, channels, False),
            nn.Sigmoid()
        )


    def forward(self,x):


        b,c,_,_ = x.size()

        avg = self.avgpool(x)#.view(b,c)
        max = self.maxpool(x)#.view(b,c)

        out = self.conv(torch.cat((avg,max),dim = 1))
       


        out2 = self.fc(out.view(b,c)).view(b,c,1,1)

       

        return x * out2.expand_as(x)  #TODO?





class LowFR(nn.Module):
    def __init__(self,inplanes = None,outplanes = None,ksize = 1,stride = 1):
        super(LowFR, self).__init__()

       


    def forward(self,x):

       

        return x


class DF(nn.Module):

    def __init__(self, inplanes = 64,scale_factor = 4):
        super(DF, self).__init__()

       

    def forward(self, x):
        
        return x


if __name__ == '__main__':
    x = torch.randn((1,3,256,256))

    model = DF()

    out = model(x)

    print(out.shape)