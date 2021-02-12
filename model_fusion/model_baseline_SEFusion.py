import os
from utils import *
import torchvision.models as tvm
from torchvision.models.resnet import BasicBlock
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

BatchNorm2d = nn.BatchNorm2d
from model.model_baseline import Net
from model.backbone.FaceBagNet import SEModule

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


    def __init__(self, num_class=2, modality='fusion'):
        super(FusionNet,self).__init__()
        # Net是model_baseline中的net，返回
        # logit.shape: torch.Size([batch_size, 2])
        # logit.shape: torch.Size([batch_size, 300])
        # fea.shape: torch.Size([batch_size, 512])
        self.modality = modality
        if self.modality == 'fusion':
            self.color_moudle  = Net(num_class=num_class,is_first_bn=True)
            self.depth_moudle = Net(num_class=num_class,is_first_bn=True)
            self.ir_moudle = Net(num_class=num_class,is_first_bn=True)

            # SEModule，输入channels和reduction,这个channel要和前一个网络的输出维度一致
            self.color_SE = SEModule(128,reduction=16)
            self.depth_SE = SEModule(128,reduction=16)
            self.ir_SE = SEModule(128,reduction=16)

            # 采用resnet的方式创建两个层
            self.res_0 = self._make_layer(BasicBlock, 384, 256, 2, stride=2)
        else:
            self.color_moudle = Net(num_class=num_class, is_first_bn=True)
            self.color_SE = SEModule(128, reduction=16)
            self.res_0 = self._make_layer(BasicBlock, 128, 256, 2, stride=2)
        self.res_1 = self._make_layer(BasicBlock, 256, 512, 2, stride=2)

        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(512, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, num_class))

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 :
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        batch_size,C,H,W = x.shape
        if self.modality == 'fusion':
            color = x[:, 0:3,:,:]
            depth = x[:, 3:6,:,:]
            ir = x[:, 6:9,:,:]

            # 返回shape: torch.Size([batch_size, 128, h/8, w/8])
            color_feas = self.color_moudle.forward_res3(color)
            depth_feas = self.depth_moudle.forward_res3(depth)
            ir_feas = self.ir_moudle.forward_res3(ir)

            # 返回shape: torch.Size([batch_size, 128, h/8, w/8])
            color_feas = self.color_SE(color_feas)
            depth_feas = self.depth_SE(depth_feas)
            ir_feas = self.ir_SE(ir_feas)

            # 返回fea.shape: torch.Size([batch_size, 384, h/8, w/8])
            fea = torch.cat([color_feas, depth_feas, ir_feas], dim=1)
        else:
            color = x
            color_feas = self.color_moudle.forward_res3(color)
            fea = self.color_SE(color_feas)

        x = self.res_0(fea)
        x = self.res_1(x)
        x = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size, -1)
        x = self.fc(x)
        # x.shape: torch.Size([batch_size, 2])
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
    print(net)

########################################################################################
if __name__ == '__main__':
    # run_check_net()
    # print( 'sucessful!')
    x = torch.rand((4, 3, 112, 112))
    model = FusionNet(num_class=2, modality = 'color')
    y,_,_ = model(x)
    print(y.shape)