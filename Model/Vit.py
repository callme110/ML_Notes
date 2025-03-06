# -*- coding:utf-8 -*-
# @Time : 2025/3/4 11:31
# @Author: Jay
# @File : Vit.py
# @Description :
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    # image size 图像大小，patch size 每个patch大小
    def __init__(self, image_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        '''
                image_size 作为局部变量，在__init__方法执行后会销毁，在其他方法中无法使用
                self.image_size 是实例属性，存储在对象的内部，可以在该类的其他方法中访问
        '''
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])  # 224//16=14   （14*14）
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 图像被分为多少个块

        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size,
                              stride=patch_size)  # 3*224*224 ---> 768*14*14
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        batch_size, channel, height, width = x.shape
        assert height == self.image_size[0] and width == self.image_size[1], \
            f"输入图像大小{height}*{width}与模型期望大小{self.image_size[0]}*{self.image_size[1]}不一致"
        # batch_size,3，224，224 ---> batch_size,768,14*14 ---> batch_size,768,196 ---> batch_size,176,768
        x = self.proj(x).flatten(2).transpose(1, 2)  # flatten(2)将x的第2个张量后所有张量展平
        x = self.norm(x)  # 若有归一化层
        return x
