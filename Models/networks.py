import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from functools import partial
import math
import torchvision.models as models
import torchvision.models as models
import random

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)
    
def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2),stride=stride, bias=bias)

class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.PReLU(), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            if i == 0:
                m.append(conv(n_feats, 32, kernel_size, bias=bias))
            else:
                m.append(conv(32, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
    
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class CALayer(nn.Module):
    def __init__(self, channel, reduction=2, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y 

class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)

    def forward(self, x, x_img):
        img = self.conv2(x) + x_img
        return img

class selfattention(nn.Module):
    def __init__(self, inplanes):
        super(selfattention, self).__init__()

        self.interchannel = inplanes
        self.inplane = inplanes
        self.g = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(inplanes, self.interchannel, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(inplanes, self.interchannel, kernel_size=1, stride=1, padding=0)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        b, c, h, w = x.size()
        g_y = self.g(x).view(b, c, -1)  # BXcXN
        theta_x = self.theta(x).view(b, self.interchannel, -1)
        theta_x = F.softmax(theta_x, dim=-1)  # softmax on N
        theta_x = theta_x.permute(0, 2, 1).contiguous()  # BXNXC'

        phi_x = self.phi(x).view(b, self.interchannel, -1)  # BXC'XN

        similarity = torch.bmm(phi_x, theta_x)  # BXc'Xc'

        g_y = F.softmax(g_y, dim=1)
        attention = torch.bmm(similarity, g_y)  # BXCXN
        attention = attention.view(b, c, h, w).contiguous()
        y = self.act(x + attention)
        return y

class FaceCycleBackbone(torch.nn.Module):
    def __init__(self, channels, bias=False):
        super(FaceCycleBackbone, self).__init__()
        act = act = nn.PReLU()
        self.channels = channels
        ##############Stage-1###################################################  
        self.alpha_0 = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.matrix_0 = ResBlock(default_conv,3,3)
        self.matrix_T0 = ResBlock(default_conv,3,3)
        self.proxnet_0 = nn.Sequential(conv(3, self.channels, kernel_size=3, bias=True),
                                           CAB(self.channels, kernel_size=3, reduction=2, bias=True, act=act))
        self.rec_0 = SAM(self.channels, kernel_size=1, bias=bias)
        
        ##############Stage-2################################################### 
        self.alpha_1 = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.matrix_1 = ResBlock(default_conv,3,3)
        self.matrix_T1 = ResBlock(default_conv,3,3)
        self.proxnet_1 = nn.Sequential(conv(3, self.channels, kernel_size=3, bias=True),
                                           CAB(self.channels, kernel_size=3, reduction=2, bias=True, act=act))
        self.rec_1 = SAM(self.channels, kernel_size=1, bias=bias)

        ##############Stage-3################################################### 
        self.alpha_2 = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.matrix_2 = ResBlock(default_conv,3,3)
        self.matrix_T2 = ResBlock(default_conv,3,3)
        self.proxnet_2 = nn.Sequential(conv(3, self.channels, kernel_size=3, bias=True),
                                           CAB(self.channels, kernel_size=3, reduction=2, bias=True, act=act))
        self.rec_2 = SAM(self.channels, kernel_size=1, bias=bias)

    def forward(self, input):
        noise_image = []
        clean_image = []
        
        ##-------------- Stage 1---------------------
        x1_m = self.matrix_0(input) - input
        x1_t = input - self.alpha_0 * self.matrix_T0(x1_m)
        x1 = self.proxnet_0(x1_t)
        x1_clean = self.rec_0(x1, x1_t)
        x1_noise = input - x1_clean
        clean_image.append(x1_clean)
        noise_image.append(x1_noise)
        
        ##-------------- Stage 2---------------------
        x2_m = self.matrix_1(x1_clean) - input
        x2_t = x1_clean - self.alpha_1 * self.matrix_T1(x2_m)
        x2 = self.proxnet_1(x2_t)
        x2_clean = self.rec_1(x2, x2_t)
        x2_noise = input - x2_clean
        clean_image.append(x2_clean)
        noise_image.append(x2_noise)

        ##-------------- Stage 3---------------------
        x3_m = self.matrix_2(x2_clean) - input
        x3_t = x2_clean - self.alpha_2 * self.matrix_T2(x3_m)
        x3 = self.proxnet_2(x3_t)
        x3_clean = self.rec_2(x3, x3_t)
        x3_noise = input - x3_clean
        clean_image.append(x3_clean)
        noise_image.append(x3_noise)

        return clean_image, noise_image

class ExpPoseModel(nn.Module):
    def __init__(self,stage, channels):
        super(ExpPoseModel, self).__init__()
        
        self.stage = stage
        self.channels = channels
        self.encoder = FaceCycleBackbone(self.channels)

        self.clean_fc = nn.Sequential(nn.Linear(12288, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048,512),
                                         nn.BatchNorm1d(512))

        self.noise_fc = nn.Sequential(nn.Linear(12288, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048,512),
                                         nn.BatchNorm1d(512))

    def forward(self,exp_img=None, normal_img=None, flip_img=None, exp_image_pose_neg=None, recon_only=False, state='pfe'):
        if not recon_only:
            if state == 'pfe':
                normal_clean_img, normal_noise_img = self.encoder(normal_img)
                flip_clean_img, flip_noise_img = self.encoder(flip_img)

                return normal_clean_img, normal_noise_img, flip_clean_img, flip_noise_img
                
            elif state == 'exp':
                list_exp_fea = []
                exp_image, _ = self.encoder(exp_img)
                for i in range(self.stage):
                    exp_fea = exp_image[i].view(exp_img.size()[0], -1)
                    exp_fea_fc = self.clean_fc(exp_fea)
                    list_exp_fea.append(exp_fea_fc)
                return list_exp_fea
                
            elif state == 'pose':
                list_pose_fea = []
                lis_pose_neg_fea = []
                _, pose_image = self.encoder(exp_img)
                _, pose_neg_image = self.encoder(exp_image_pose_neg)
                for i in range(self.stage):
                    pose_fea = pose_image[i].view(exp_img.size()[0], -1)
                    pose_fea_fc = self.noise_fc(pose_fea)
                    list_pose_fea.append(pose_fea_fc)

                    pose_neg_fea = pose_neg_image[i].view(exp_image_pose_neg.size()[0], -1)
                    pose_neg_fea_fc = self.noise_fc(pose_neg_fea)
                    lis_pose_neg_fea.append(pose_neg_fea_fc)
                return list_pose_fea, lis_pose_neg_fea
        else:
            normal_clean_img, normal_noise_img = self.encoder(normal_img)
            flip_clean_img, flip_noise_img = self.encoder(flip_img)

            return normal_clean_img, normal_noise_img, flip_clean_img, flip_noise_img
        
