# Copyright 2018-2019 Xiao Zhai
#
# This file is part of Adaptive Style Transfer, my own implementation of the 
# ECCV 2018 paper A Style-Aware Content Loss for Real-time HD Style Transfer.
#
# Adaptive Style Transfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Adaptive Style Transfer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from torch import nn
import torch
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self, options):
        super(encoder, self).__init__()
        self.c1 = nn.Conv2d(options.dim, options.gf_dim, 3)
        self.c2 = nn.Conv2d(options.gf_dim, options.gf_dim, 3, stride=2)
        self.c3 = nn.Conv2d(options.gf_dim, options.gf_dim*2, 3, stride=2)
        self.c4 = nn.Conv2d(options.gf_dim*2, options.gf_dim*4, 3, stride=2)
        self.c5 = nn.Conv2d(options.gf_dim*4, options.gf_dim*8, 3, stride=2)        
    def forward(self, x):
        self.eval()
        x = F.pad(F.instance_norm(x), (15,15,15,15), 'reflect')
        x = F.relu(F.instance_norm(self.c1(x)), inplace=True)
        x = F.relu(F.instance_norm(self.c2(x)), inplace=True)
        x = F.relu(F.instance_norm(self.c3(x)), inplace=True)
        x = F.relu(F.instance_norm(self.c4(x)), inplace=True)
        x = F.relu(F.instance_norm(self.c5(x)), inplace=True)
        self.train()
        return x

class residule_block(nn.Module):
    def __init__(self, dim, ks=3, s=1):
        super(residule_block, self).__init__()
        self.c1 = nn.Conv2d(dim, dim, ks, stride=s)
        self.c2 = nn.Conv2d(dim, dim, ks, stride=s)
        self.ks = ks
    def forward(self, x):
        p = int((self.ks - 1) / 2)
        y = F.pad(x, (p,p,p,p), 'reflect')
        y = F.instance_norm(self.c1(y))
        y = F.pad(F.relu(x), (p,p,p,p), 'reflect')
        y = F.instance_norm(self.c2(y))
        return y+x

class upscale_block(nn.Module):
    def __init__(self, in_dim, out_dim, ks=3, s=2):
        super(upscale_block, self).__init__()
        self.c1 = nn.Conv2d(in_dim, out_dim, ks, stride=1, padding=int((ks - 1) / 2))
        self.s = s
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.s, mode='nearest')
        return F.relu(F.instance_norm(self.c1(x)), inplace=True)

class decoder(nn.Module):
    def __init__(self, options):
        super(decoder, self).__init__()
        self.r1 = residule_block(options.gf_dim*8)
        self.r2 = residule_block(options.gf_dim*8)
        self.r3 = residule_block(options.gf_dim*8)
        self.r4 = residule_block(options.gf_dim*8)
        self.r5 = residule_block(options.gf_dim*8)
        self.r6 = residule_block(options.gf_dim*8)
        self.r7 = residule_block(options.gf_dim*8)
        self.r8 = residule_block(options.gf_dim*8)
        self.r9 = residule_block(options.gf_dim*8)
        self.u1 = upscale_block(options.gf_dim*8, options.gf_dim*8, 3,2)
        self.u2 = upscale_block(options.gf_dim*8, options.gf_dim*4, 3,2)
        self.u3 = upscale_block(options.gf_dim*4, options.gf_dim*2, 3,2)
        self.u4 = upscale_block(options.gf_dim*2, options.gf_dim, 3,2)
        self.c1 = nn.Conv2d(options.gf_dim, 3, 7, stride=1)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        self.eval()
        x = self.r3(self.r2(self.r1(x)))
        x = self.r6(self.r5(self.r4(x)))
        x = self.r9(self.r8(self.r7(x)))
        x = self.u4(self.u3(self.u2(self.u1(x))))
        x = F.pad(x, (3,3,3,3), 'reflect')
        x = self.sig(self.c1(x))*2. - 1.
        self.train()
        return x

class transformer_block(nn.Module):
    def __init__(self, options):
        super(transformer_block, self).__init__()
    def forward(self, x):
        self.eval()
        F.avg_pool2d(x,kernel_size=10,stride=1)
        self.train()
        return x

class discriminator(nn.Module):
    def __init__(self, options):
        super(discriminator, self).__init__()
        self.c1 = nn.Conv2d(options.dim, options.df_dim*2, 5, stride=2)
        self.p1 = nn.Conv2d(options.df_dim*2, 1, 5, stride=1)
        self.c2 = nn.Conv2d(options.df_dim*2, options.df_dim*2, 5, stride=2)
        self.p2 = nn.Conv2d(options.df_dim*2, 1, 10, stride=1)
        self.c3 = nn.Conv2d(options.df_dim*2, options.df_dim*4, 5, stride=2)
        self.c4 = nn.Conv2d(options.df_dim*4, options.df_dim*8, 5, stride=2)
        self.p4 = nn.Conv2d(options.df_dim*8, 1, 10, stride=1)
        self.c5 = nn.Conv2d(options.df_dim*8, options.df_dim*8, 5, stride=2)
        self.c6 = nn.Conv2d(options.df_dim*8, options.df_dim*16, 5, stride=2)
        self.p6 = nn.Conv2d(options.df_dim*16, 1, 6, stride=1)
        self.c7 = nn.Conv2d(options.df_dim*16, options.df_dim*16, 5, stride=2)
        self.p7 = nn.Conv2d(options.df_dim*16, 1, 3, stride=1)
    def forward(self, x):
        x = F.leaky_relu(F.instance_norm(self.c1(x)), negative_slope=0.2, inplace=True)
        p1 = self.p1(x)
        x = F.leaky_relu(F.instance_norm(self.c2(x)), negative_slope=0.2, inplace=True)
        p2 = self.p2(x)
        x = F.leaky_relu(F.instance_norm(self.c3(x)), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(F.instance_norm(self.c4(x)), negative_slope=0.2, inplace=True)
        p4 = self.p4(x)
        x = F.leaky_relu(F.instance_norm(self.c5(x)), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(F.instance_norm(self.c6(x)), negative_slope=0.2, inplace=True)
        p6 = self.p6(x)
        x = F.leaky_relu(F.instance_norm(self.c7(x)), negative_slope=0.2, inplace=True)
        p7 = self.p7(x)
        return {"pred_1": p1,
                "pred_2": p2,
                "pred_4": p4,
                "pred_6": p6,
                "pred_7": p7}