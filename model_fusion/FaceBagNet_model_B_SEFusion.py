import os
from utils import *
from torchvision.models.resnet import BasicBlock
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
BatchNorm2d = nn.BatchNorm2d
from model.FaceBagNet_model_B import Net
from model.backbone.FaceBagNet import SEModule, SEResNeXtBottleneck

###########################################################################################3
class FusionNet(nn.Module):
    def load_pretrain(self, pretrain_file):
        #raise NotImplementedError
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        print('')


    def __init__(self, num_class=2):
        super(FusionNet,self).__init__()

        self.color_moudle  = Net(num_class=num_class,is_first_bn=True)
        self.depth_moudle = Net(num_class=num_class,is_first_bn=True)
        self.ir_moudle = Net(num_class=num_class,is_first_bn=True)

        self.color_SE = SEModule(512, reduction=16)
        self.depth_SE = SEModule(512, reduction=16)
        self.ir_SE = SEModule(512, reduction=16)

        self.bottleneck = nn.Sequential(nn.Conv2d(512*3, 512, kernel_size=1, padding=0),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(inplace=True))

        # res_0的输入是512维，输出是1024维，
        self.res_0 = self._make_layer(
            SEResNeXtBottleneck,
            planes=256,
            blocks=2,
            stride=2,
            groups=32,
            reduction=16,
            downsample_kernel_size=1,
            downsample_padding=0
        )

        # res_1的输入是1024维，输出是2048维，
        self.res_1 = self._make_layer(
            SEResNeXtBottleneck,
            planes=512,
            blocks=2,
            stride=2,
            groups=32,
            reduction=16,
            downsample_kernel_size=1,
            downsample_padding=0
        )

        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(2048, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, num_class))

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):

        downsample = None
        # 以下这行为增加的
        inplanes = planes*2
        # 以下这行是源代码的，这行应该不要的，否则输入和输出维度永远一致，无法和下一层相连接
        # self.inplanes = planes * block.expansion
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )


        layers = []
        layers.append(block(inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)


    def forward(self, x):
        batch_size,C,H,W = x.shape

        color = x[:, 0:3,:,:]
        depth = x[:, 3:6,:,:]
        ir = x[:, 6:9,:,:]
        # 返回shape: torch.Size([batch_size, 512, h/8, w/8])
        color_feas = self.color_moudle.forward_res3(color)
        depth_feas = self.depth_moudle.forward_res3(depth)
        ir_feas = self.ir_moudle.forward_res3(ir)

        # 返回shape: torch.Size([batch_size, 512, h/8, w/8])
        color_feas = self.color_SE(color_feas)
        depth_feas = self.depth_SE(depth_feas)
        ir_feas = self.ir_SE(ir_feas)

        # 返回shape: torch.Size([batch_size, 1536, h/8, w/8])
        fea = torch.cat([color_feas, depth_feas, ir_feas], dim=1)

        # 返回shape: torch.Size([batch_size, 1024, h/8, w/8])
        fea = self.bottleneck(fea)


        # 返回shape: torch.Size([batch_size, 1024, h/16, w/16])
        x = self.res_0(fea)

        # 返回shape: torch.Size([batch_size, 2048, h/32, w/32])
        x = self.res_1(x)

        # 返回shape: torch.Size([batch_size, 2048])
        x = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size, -1)

        # x.shape: torch.Size([batch_size, 2])
        x = self.fc(x)
        return x,None,None

    def set_mode(self, mode, is_freeze_bn=False ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['backup']:
            self.train()
            if is_freeze_bn==True: ##freeze
                for m in self.modules():
                    if isinstance(m, BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad   = False

### run ##############################################################################
def run_check_net():
    num_class = 2
    net = Net(num_class)
    # print(net)

########################################################################################
if __name__ == '__main__':
    # run_check_net()
    # print( 'sucessful!')
    # 在FaceBagNet_model_B_SEFusion.py中from backbone.FaceBagNet前加.
    x = torch.rand((4, 9, 112, 112)).cuda()
    model = FusionNet(num_class=2).cuda()
    y,_,_ = model(x)
    print(y.shape)