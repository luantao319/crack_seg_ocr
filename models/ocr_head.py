#!/usr/bin/env python3
"""
OCR分割头模块 - 基于完整OCR实现，适配单帧图像
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def label_to_onehot(gt, num_classes, ignore_index=-1):
    """
    gt: ground truth with size (N, H, W)
    num_classes: the number of classes of different label
    """
    N, H, W = gt.size()
    x = gt
    x[x == ignore_index] = num_classes
    # convert label into onehot format
    onehot = torch.zeros(N, x.size(1), x.size(2), num_classes + 1).cuda()
    onehot = onehot.scatter_(-1, x.unsqueeze(-1), 1)
    return onehot.permute(0, 3, 1, 2)

class SpatialGather_Module(nn.Module):
    """
    Aggregate the context features according to the initial predicted probability distribution.
    Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=1, scale=1, use_gt=False):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale
        self.use_gt = use_gt
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats, probs, gt_probs=None):
        if self.use_gt and gt_probs is not None:
            gt_probs = label_to_onehot(
                gt_probs.squeeze(1).type(torch.cuda.LongTensor), probs.size(1)
            )
            batch_size, c, h, w = (
                gt_probs.size(0),
                gt_probs.size(1),
                gt_probs.size(2),
                gt_probs.size(3),
            )
            gt_probs = gt_probs.view(batch_size, c, -1)
            feats = feats.view(batch_size, feats.size(1), -1)
            feats = feats.permute(0, 2, 1)  # batch x hw x c
            gt_probs = F.normalize(gt_probs, p=1, dim=2)  # batch x k x hw
            ocr_context = (
                torch.matmul(gt_probs, feats).permute(0, 2, 1).unsqueeze(3)
            )  # batch x k x c
            return ocr_context
        else:
            batch_size, c, h, w = (
                probs.size(0),
                probs.size(1),
                probs.size(2),
                probs.size(3),
            )
            probs = probs.view(batch_size, c, -1)
            feats = feats.view(batch_size, feats.size(1), -1)
            feats = feats.permute(0, 2, 1)  # batch x hw x c
            probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
            ocr_context = (
                torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)
            )  # batch x k x c
            return ocr_context

class PyramidSpatialGather_Module(nn.Module):
    """
    Aggregate the context features according to the initial predicted probability distribution.
    Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=1, scales=[1, 2, 4]):
        super(PyramidSpatialGather_Module, self).__init__()
        self.scales = scales
        self.relu = nn.ReLU(inplace=True)

    def _compute_single_scale(self, feats, probs, dh, dw):
        batch_size, k, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        c = feats.size(1)

        out_h, out_w = math.ceil(h / dh), math.ceil(w / dw)
        pad_h, pad_w = out_h * dh - h, out_w * dw - w
        if pad_h > 0 or pad_w > 0:  # padding in both left&right sides
            feats = F.pad(
                feats, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            )
            probs = F.pad(
                probs, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            )

        feats = feats.view(batch_size, c, out_h, dh, out_w, dw).permute(
            0, 3, 5, 1, 2, 4
        )
        feats = feats.contiguous().view(batch_size, dh * dw, c, out_h, out_w)

        probs = probs.view(batch_size, k, out_h, dh, out_w, dw).permute(
            0, 3, 5, 1, 2, 4
        )
        probs = probs.contiguous().view(batch_size, dh * dw, k, out_h, out_w)

        feats = feats.view(batch_size, dh * dw, c, -1)
        probs = probs.view(batch_size, dh * dw, k, -1)
        feats = feats.permute(0, 1, 3, 2)

        probs = F.softmax(probs, dim=3)  # batch x k x hw
        cc = torch.matmul(probs, feats).view(batch_size, -1, c)  # batch x k x c

        return cc.permute(0, 2, 1).unsqueeze(3)

    def forward(self, feats, probs):
        ocr_list = []
        for scale in self.scales:
            ocr_tmp = self._compute_single_scale(feats, probs, scale, scale)
            ocr_list.append(ocr_tmp)
        pyramid_ocr = torch.cat(ocr_list, 2)
        return pyramid_ocr

class ObjectAttentionBlock(nn.Module):
    """
    The basic implementation for object context block
    """
    def __init__(self, in_channels, key_channels, scale=1, use_gt=False, use_bg=False, fetch_attention=False):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.fetch_attention = fetch_attention
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels), nn.ReLU(),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels), nn.ReLU(),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels), nn.ReLU(),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels), nn.ReLU(),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels), nn.ReLU(),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels), nn.ReLU(),
        )

    def forward(self, x, proxy, gt_label=None):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        if self.use_gt and gt_label is not None:
            gt_label = label_to_onehot(
                gt_label.squeeze(1).type(torch.cuda.LongTensor), proxy.size(2) - 1
            )
            sim_map = (
                gt_label[:, :, :, :].permute(0, 2, 3, 1).view(batch_size, h * w, -1)
            )
            if self.use_bg:
                bg_sim_map = 1.0 - sim_map
                bg_sim_map = F.normalize(bg_sim_map, p=1, dim=-1)
            sim_map = F.normalize(sim_map, p=1, dim=-1)
        else:
            sim_map = torch.matmul(query, key)
            sim_map = (self.key_channels ** -0.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)  # hw x k x k x c
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(
                input=context, size=(h, w), mode="bilinear", align_corners=True
            )

        if self.use_bg:
            bg_context = torch.matmul(bg_sim_map, value)
            bg_context = bg_context.permute(0, 2, 1).contiguous()
            bg_context = bg_context.view(batch_size, self.key_channels, *x.size()[2:])
            bg_context = self.f_up(bg_context)
            bg_context = F.interpolate(
                input=bg_context, size=(h, w), mode="bilinear", align_corners=True
            )
            return context, bg_context
        else:
            if self.fetch_attention:
                return context, sim_map
            else:
                return context

class ObjectAttentionBlock2D(ObjectAttentionBlock):
    def __init__(self, in_channels, key_channels, scale=1, use_gt=False, use_bg=False, fetch_attention=False, bn_type="torchbn"):
        super(ObjectAttentionBlock2D, self).__init__(
            in_channels, key_channels, scale, use_gt, use_bg, fetch_attention
        )

class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.1, use_gt=False, use_bg=False, use_oc=True, fetch_attention=False, bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.use_oc = use_oc
        self.fetch_attention = fetch_attention
        self.object_context_block = ObjectAttentionBlock2D(
            in_channels, key_channels, scale, use_gt, use_bg, fetch_attention, bn_type
        )
        if self.use_bg:
            if self.use_oc:
                _in_channels = 3 * in_channels
            else:
                _in_channels = 2 * in_channels
        else:
            _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, feats, proxy_feats, gt_label=None):
        if self.use_gt and gt_label is not None:
            if self.use_bg:
                context, bg_context = self.object_context_block(feats, proxy_feats, gt_label)
            else:
                context = self.object_context_block(feats, proxy_feats, gt_label)
        else:
            if self.fetch_attention:
                context, sim_map = self.object_context_block(feats, proxy_feats)
            else:
                context = self.object_context_block(feats, proxy_feats)

        if self.use_bg:
            if self.use_oc:
                output = self.conv_bn_dropout(torch.cat([context, bg_context, feats], 1))
            else:
                output = self.conv_bn_dropout(torch.cat([bg_context, feats], 1))
        else:
            output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        if self.fetch_attention:
            return output, sim_map
        else:
            return output

class OCRSegmentationHead(nn.Module):
    """
    OCR分割头 - 基于完整OCR实现，适配单帧图像
    """
    
    def __init__(self, in_channels_list, num_classes=1, ocr_mid_channels=512, ocr_key_channels=256):
        super().__init__()
        self.num_classes = num_classes
        
        # 计算总输入通道数
        total_channels = sum(in_channels_list)
        
        # 特征融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, ocr_mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 辅助预测头
        self.aux_head = nn.Sequential(
            nn.Conv2d(total_channels, ocr_mid_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(ocr_mid_channels), nn.GELU(),
            nn.Conv2d(ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
        # OCR模块
        self.ocr_gather_head = PyramidSpatialGather_Module(cls_num=num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=ocr_mid_channels, 
            key_channels=ocr_key_channels,
            out_channels=ocr_mid_channels, 
            scale=1, 
            dropout=0.05
        )
        
        # 最终预测头
        self.cls_head = nn.Sequential(
            nn.Conv2d(ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_classes)
        )
        # 上采样层 - 将输出上采样到原始输入尺寸
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
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
    
    def forward(self, features):
        """
        Args:
            features: 多尺度特征列表 [feat1, feat2, feat3, feat4]
        Returns:
            out: 最终分割输出 [B, num_classes, H, W]
        """
        # 获取第一个特征的尺寸作为目标尺寸
        target_size = features[0].shape[2:]
        
        # 将所有特征双线性插值到相同尺寸
        aligned_features = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        # 拼接所有特征
        concat_feat = torch.cat(aligned_features, dim=1)
        
        # 特征融合
        fused_feat = self.fusion_conv(concat_feat)
        
        # 辅助预测
        out_aux = self.aux_head(concat_feat)
        
        # OCR处理
        # 1. 聚集上下文
        context = self.ocr_gather_head(fused_feat, out_aux)
        
        # 2. 分布上下文
        enhanced_feat = self.ocr_distri_head(fused_feat, context)
        
        # 3. 最终预测
        out = self.cls_head(enhanced_feat)
        
        # 4. 上采样到原始输入尺寸 (512x512)
        out = self.upsample(out)
        
        return out