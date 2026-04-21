#!/usr/bin/env python3
"""
RepViT backbone模块
"""

import torch
import torch.nn as nn
import sys
import os

# 添加repvit.py的路径
sys.path.append('/home/huangzh/myreal/crack_seg_ocr')
from repvit import repvit_m1_1, RepViTBlock

class RepViTBackbone(nn.Module):
    """RepViT backbone，输出多个中间层特征"""
    
    def __init__(self, out_indices=[2, 6, 19, 23]):
        super().__init__()
        self.out_indices = out_indices
        
        # 创建RepViT模型 - 与crack_seg项目保持一致
        self.repvit = repvit_m1_1(pretrained=False, num_classes=1000, distillation=False)
        
        # 获取特征提取器
        self.features = self.repvit.features
        
        # 存储中间层输出
        self.intermediate_features = {}
        
    def forward(self, x):
        """前向传播，返回多个中间层特征"""
        features = []
        
        # 直接遍历RepViT的features层，提取指定索引的特征
        for i, layer in enumerate(self.repvit.features):
            x = layer(x)
            
            # 检查是否是需要的中间层
            if i in self.out_indices:
                features.append(x)
        
        return features

def repvit_m1_1_backbone(out_indices=[2, 6, 19, 23]):
    """创建RepViT M1.1 backbone"""
    return RepViTBackbone(out_indices=out_indices)