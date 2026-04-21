#!/usr/bin/env python3
"""
小波增强的裂缝分割模型
结合小波滤波和RepViT backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import pywt
import cv2
from typing import List, Tuple
import matplotlib.pyplot as plt

# 添加repvit.py的路径
sys.path.append('/home/huangzh/myreal/crack_seg_ocr')
from repvit import repvit_m1_1
from .ocr_head import OCRSegmentationHead
from .filters import AdvancedWaveletFilter

class LogGaborFilter(nn.Module):
    """LogGabor滤波器 - 专门针对裂缝检测优化"""
    def __init__(self, orientations=6, wavelengths=3, sigma=0.65):
        super(LogGaborFilter, self).__init__()
        
        # 针对裂缝检测优化的参数
        self.orientations = orientations
        self.wavelengths = wavelengths
        self.sigma = sigma
        
        # 预计算LogGabor核并转换为PyTorch张量
        self.loggabor_kernels = self._create_loggabor_kernels()
        
        # 将18层LogGabor特征合并成1层的卷积层
        self.merge_conv = nn.Conv2d(orientations * wavelengths, 1, kernel_size=1)  # 18层 -> 1层
    
    def _create_loggabor_kernels(self):
        """创建LogGabor卷积核"""
        kernels = []
        
        # 角度范围：0到π
        angles = np.linspace(0, np.pi, self.orientations, endpoint=False)
        
        # 波长范围：基于图像尺寸
        min_wavelength = 3.0
        max_wavelength = 15.0
        wavelengths = np.logspace(np.log10(min_wavelength), np.log10(max_wavelength), self.wavelengths)
        
        for wavelength in wavelengths:
            for angle in angles:
                # 创建LogGabor核
                kernel = self._create_single_loggabor_kernel(wavelength, angle)
                kernels.append(kernel)
        
        return nn.ParameterList(kernels)
    
    def _create_single_loggabor_kernel(self, wavelength, angle, size=21):
        """创建单个LogGabor核"""
        # 创建网格
        x, y = np.meshgrid(np.arange(-size//2, size//2), 
                           np.arange(-size//2, size//2))
        #print(x.shape)
        
        # 旋转坐标
        x_rot = x * np.cos(angle) + y * np.sin(angle)
        y_rot = -x * np.sin(angle) + y * np.cos(angle)
        
        # LogGabor函数
        r = np.sqrt(x_rot**2 + y_rot**2)
        r[r == 0] = 1e-10  # 避免除零
        
        # 径向函数（LogGabor）
        radial = np.exp(-(np.log(r / wavelength))**2 / (2 * np.log(self.sigma)**2))
        
        # 角度函数
        angular = np.exp(-y_rot**2 / (2 * (wavelength / 3)**2))
        
        # 组合
        kernel = radial * angular
        
        # 归一化
        kernel = kernel / np.sum(np.abs(kernel))
        
        # 转换为PyTorch张量 [1, 1, size, size]
        kernel_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
        
        return kernel_tensor
    
    def forward(self, x):
        """
        x: [B, 3, H, W] 在GPU上
        返回: [B, 3, H, W] 在GPU上
        """
        batch_size, _, height, width = x.shape
        device = x.device
        
        # 确保所有核都在正确的设备上
        if len(self.loggabor_kernels) > 0 and self.loggabor_kernels[0].device != device:
            for i, kernel in enumerate(self.loggabor_kernels):
                self.loggabor_kernels[i] = kernel.to(device)
        
        responses = []
        
        # 对每个LogGabor核进行卷积
        for kernel in self.loggabor_kernels:
            # 对每个通道分别卷积，保持3通道
            channel_responses = []
            for c in range(3):
                # [B, 1, H, W] -> [B, 1, H, W]
                response = F.conv2d(x[:, c:c+1], kernel, padding=10)
                channel_responses.append(response)
            
            # 合并通道响应，保持3通道
            response_3ch = torch.cat(channel_responses, dim=1)  # [B, 3, H, W]
            responses.append(response_3ch)  # 保持3通道
        
        # 堆叠所有响应 [B, 18, 3, H, W]
        if responses:
            responses_stack = torch.stack(responses, dim=1)  # [B, 18, 3, H, W]
            
            # 重塑为4D张量，以便卷积处理
            B, N, C, H, W = responses_stack.shape  # [B, 18, 3, H, W]
            # 重塑为 [B*3, 18, H, W] 以便用18通道->1通道的卷积
            responses_reshaped = responses_stack.permute(0, 2, 1, 3, 4).reshape(B*C, N, H, W)  # [B*3, 18, H, W]
            
            # 用1x1卷积：18通道 -> 1通道
            merged = self.merge_conv(responses_reshaped)  # [B*3, 1, H, W]
            
            # 重塑回 [B, 3, H, W]
            combined = merged.reshape(B, C, H, W)  # [B, 3, H, W]
            
            return combined
        else:
            # 如果没有响应，返回零张量
            return torch.zeros(batch_size, 3, height, width, device=device)

class ResNetBlock(nn.Module):
    """ResNet Block用于特征融合，参考原始项目的实现"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class WTLowGaborHighFusionBranch(nn.Module):
    """融合WT低频 + Gabor高频的分支"""
    
    def __init__(self, wt_type='db1', orientations=6, wavelengths=3, sigma=0.65):
        super().__init__()
        
        # 小波滤波器（只用于提取LL低频）
        self.wavelet_filter = AdvancedWaveletFilter(wt_type=wt_type, wt_levels=1, kernel_size=3)
        self.wt_function = self.wavelet_filter.wt_function
        
        # LogGabor滤波器（提供高频信息）
        self.loggabor_filter = LogGaborFilter(orientations=orientations, wavelengths=wavelengths, sigma=sigma)
        
        # 映射卷积层: 6通道 -> 3通道
        self.mapping_conv = nn.Conv2d(6, 3, kernel_size=1)
        
        # 融合后的RepViT backbone
        self.fusion_repvit = repvit_m1_1(pretrained=False, num_classes=1000, distillation=False)
        self.fusion_features = self.fusion_repvit.features
        
    def forward(self, x):
        """前向传播"""
        # 融合WT低频 + Gabor高频，并映射到3通道
        fused_input = self.fuse_and_map(x)  # [B, 3, H, W]
        
        # 直接返回融合后的3通道特征，不要送入RepViT
        return fused_input  # [B, 3, H, W]
    
    def fuse_and_map(self, x):
        """融合WT低频 + Gabor高频，并映射到3通道"""
        # WT低频: 结构信息（小波LL分量）
        wt_low = self.extract_wavelet_low_freq(x)  # [B, 3, H, W]
        
        # Gabor高频: 纹理细节（LogGabor响应 - 原图）
        gabor_high = self.get_gabor_high_freq(x)  # [B, 3, H, W]
        
        # 拼接: [B, 6, H, W]
        concat_features = torch.cat([wt_low, gabor_high], dim=1)
        
        # 通过1x1卷积映射到3通道
        mapped_features = self.mapping_conv(concat_features)
        
        return mapped_features  # [B, 3, H, W]
    
    def extract_wavelet_low_freq(self, x):
        """提取小波低频分量LL"""
        curr_x = self.wt_function(x)  # [B, 3, 4, H//2, W//2]
        low_freq = curr_x[:, :, 0, :, :]  # [B, 3, H//2, W//2] - LL分量
        
        # 上采样回原始尺寸
        low_freq = F.interpolate(low_freq, size=x.shape[2:], mode='bilinear', align_corners=False)
        return low_freq  # [B, 3, H, W]
    
    def get_gabor_high_freq(self, x):
        """获取Gabor高频分量"""
        # 获取LogGabor滤波结果
        gabor_response = self.loggabor_filter(x)  # [B, 3, H, W]
        
        # 提取高频分量：与原图差分
        gabor_high = gabor_response - x  # [B, 3, H, W]
        
        return gabor_high
    
    def get_fusion_visualization(self, x):
        """获取融合分支的可视化结果"""
        with torch.no_grad():
            # 获取中间结果
            wt_low = self.extract_wavelet_low_freq(x)      # [B, 3, H, W] - 小波LL低频分量
            gabor_high = self.get_gabor_high_freq(x)      # [B, 3, H, W] - Gabor高频分量（滤波响应-原图）
            fused = self.fuse_and_map(x)                  # [B, 3, H, W] - 融合后的3通道特征
            
            # 返回可视化结果
            return {
                'wt_low': wt_low,           # WT低频分量（结构信息）
                'gabor_high': gabor_high,   # Gabor高频分量（纹理细节）
                'fused': fused              # 融合后的3通道特征
            }

class WaveletEnhancedBackbone(nn.Module):
    """小波增强的backbone，支持RGB、小波、LogGabor和融合分支"""
    
    def __init__(self, out_indices=[2, 6, 19, 23], use_wavelet=True, use_loggabor=True, use_fusion=True):
        super().__init__()
        self.out_indices = out_indices
        self.use_wavelet = use_wavelet
        self.use_loggabor = use_loggabor
        self.use_fusion = use_fusion
        
        # 小波滤波器
        if use_wavelet:
            self.wavelet_filter = AdvancedWaveletFilter(wt_type='db1', wt_levels=1, kernel_size=3)
        
        # LogGabor滤波器
        if use_loggabor:
            self.loggabor_filter = LogGaborFilter(orientations=6, wavelengths=3, sigma=0.65)
        
        # 融合分支（WT低频 + Gabor高频）
        if use_fusion:
            self.fusion_branch = WTLowGaborHighFusionBranch()
        
        # 输入融合层 - 在进入backbone之前融合3通道特征
        if use_wavelet or use_loggabor or use_fusion:
            # 计算融合后的通道数
            total_channels = 3  # RGB特征通道数
            if use_wavelet:
                total_channels += 3  # 小波特征通道数
            if use_loggabor:
                total_channels += 3  # LogGabor特征通道数
            if use_fusion:
                total_channels += 3  # 融合分支特征通道数
            
            # 融合层：将多分支3通道特征融合成3通道
            self.input_fusion_block = nn.Sequential(
                nn.Conv2d(total_channels, 64, 3, padding=1),  # 3x3卷积扩展通道
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                ResNetBlock(64, 64), 
                ResNetBlock(64, 16),                                         # 第1层ResNetBlock
                ResNetBlock(16, 3),                           # 第2层ResNetBlock，输出3通道
            )
        
        # RepViT模型 - 使用融合后的输入
        self.repvit = repvit_m1_1(pretrained=False, num_classes=1000, distillation=False)
        self.features = self.repvit.features
        
    def forward(self, x):
        """前向传播"""
        features = []
        
        if self.use_wavelet or self.use_loggabor or self.use_fusion:
            # 输入融合：在进入backbone之前融合多分支特征
            concat_feats = [x]  # RGB原图
            
            if self.use_wavelet:
                wavelet_x = self.wavelet_filter(x)  # [B, 3, H, W]
                concat_feats.append(wavelet_x)
            
            if self.use_loggabor:
                loggabor_x = self.loggabor_filter(x)  # [B, 3, H, W]
                concat_feats.append(loggabor_x)
            
            if self.use_fusion:
                fusion_x = self.fusion_branch.fuse_and_map(x)  # [B, 3, H, W]
                concat_feats.append(fusion_x)
            
            # 拼接所有特征
            concat_feat = torch.cat(concat_feats, dim=1)  # [B, total_channels, H, W]
            
            # 通过输入融合层融合成3通道
            fused_input = self.input_fusion_block(concat_feat)  # [B, 3, H, W]
            
            # 使用融合后的输入送入RepViT backbone
            for i, layer in enumerate(self.features):
                fused_input = layer(fused_input)
                if i in self.out_indices:
                    features.append(fused_input)
        else:
            # 原始RGB路径
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in self.out_indices:
                    features.append(x)
        
        return features
    
    def get_wavelet_visualization(self, x):
        """获取小波可视化结果"""
        if not self.use_wavelet:
            return None
        
        with torch.no_grad():
            wavelet_x = self.wavelet_filter(x)
            return wavelet_x
    
    def get_loggabor_visualization(self, x):
        """获取LogGabor可视化结果"""
        if not self.use_loggabor:
            return None
        
        with torch.no_grad():
            loggabor_x = self.loggabor_filter(x)
            return loggabor_x
    
    def get_fusion_visualization(self, x):
        """获取融合分支可视化结果"""
        if not self.use_fusion:
            return None
        
        with torch.no_grad():
            return self.fusion_branch.get_fusion_visualization(x)

class WaveletEnhancedSegmentationModel(nn.Module):
    """小波增强的裂缝分割模型，支持RGB、小波、LogGabor和融合分支"""
    
    def __init__(self, num_classes=1, ocr_mid_channels=512, ocr_key_channels=256, use_wavelet=True, use_loggabor=True, use_fusion=True):
        super().__init__()
        self.num_classes = num_classes
        self.use_wavelet = use_wavelet
        self.use_loggabor = use_loggabor
        self.use_fusion = use_fusion
        
        # 多分支增强的backbone
        self.backbone = WaveletEnhancedBackbone(
            out_indices=[2, 6, 19, 23], 
            use_wavelet=use_wavelet, 
            use_loggabor=use_loggabor,
            use_fusion=use_fusion
        )
        
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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        out = self.segmentation_head(features)
        return out
    
    def get_visualization(self, x):
        """获取可视化结果"""
        vis_results = {}
        
        # RGB原图
        vis_results['rgb'] = x
        
        # 小波结果
        if self.use_wavelet:
            vis_results['wavelet'] = self.backbone.get_wavelet_visualization(x)
        
        # LogGabor结果
        if self.use_loggabor:
            vis_results['loggabor'] = self.backbone.get_loggabor_visualization(x)
        
        # 融合分支结果
        if self.use_fusion:
            fusion_vis = self.backbone.get_fusion_visualization(x)
            # 将RGB转换为灰度图用于可视化
            if fusion_vis is not None:
                vis_results['fusion_wt_low'] = torch.mean(fusion_vis['wt_low'], dim=1, keepdim=True)  # WT低频灰度
                vis_results['fusion_gabor_high'] = torch.mean(fusion_vis['gabor_high'], dim=1, keepdim=True)  # Gabor高频灰度
                vis_results['fusion_combined'] = torch.mean(fusion_vis['fused'], dim=1, keepdim=True)  # 融合结果灰度
        
        return vis_results

def get_wavelet_enhanced_model(num_classes=1, ocr_mid_channels=512, ocr_key_channels=256, use_wavelet=True, use_loggabor=True, use_fusion=True):
    """获取小波增强的裂缝分割模型"""
    return WaveletEnhancedSegmentationModel(
        num_classes=num_classes,
        ocr_mid_channels=ocr_mid_channels,
        ocr_key_channels=ocr_key_channels,
        use_wavelet=use_wavelet,
        use_loggabor=use_loggabor,
        use_fusion=use_fusion
    ) 