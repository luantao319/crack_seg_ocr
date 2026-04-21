#!/usr/bin/env python3
"""
完整的分割模型，结合RepViT backbone和OCR分割头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import repvit_m1_1_backbone
from .ocr_head import OCRSegmentationHead

class CrackSegmentationModel(nn.Module):
    """裂缝分割模型"""
    
    def __init__(self, num_classes=1, ocr_mid_channels=512, ocr_key_channels=256):
        super().__init__()
        self.num_classes = num_classes
        
        # RepViT backbone
        self.backbone = repvit_m1_1_backbone(out_indices=[2, 6, 19, 23])
        
        in_channels_list = [64, 128, 256, 512]
        
        # OCR分割头
        self.segmentation_head = OCRSegmentationHead(
            in_channels_list=in_channels_list,
            num_classes=num_classes,
            ocr_mid_channels=ocr_mid_channels,
            ocr_key_channels=ocr_key_channels
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            out: 分割输出 [B, num_classes, H, W]
        """
        # 提取多尺度特征
        features = self.backbone(x)
        
        # 调试信息
        if len(features) == 0:
            #print("Warning: Backbone returned empty features list!")
            # 返回一个默认的输出
            return torch.zeros(x.size(0), self.num_classes, x.size(2), x.size(3), device=x.device)
        
        #print(f"Backbone features: {len(features)} features with shapes: {[f.shape for f in features]}")
        
        # OCR分割
        out = self.segmentation_head(features)
        
        return out

def get_segmentation_model(num_classes=1, ocr_mid_channels=512, ocr_key_channels=256):
    """获取分割模型"""
    return CrackSegmentationModel(
        num_classes=num_classes,
        ocr_mid_channels=ocr_mid_channels,
        ocr_key_channels=ocr_key_channels
    )