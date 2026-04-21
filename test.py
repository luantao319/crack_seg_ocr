#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Crack Segmentation Testing  (OCR + 小波/LogGabor/融合分支 + Otsu/固定阈值)
与 train.py 完全对齐
增加功能：每个批次保存一个可视化图像
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import math
import pandas as pd
from PIL import Image

# 设置matplotlib不显示图形界面
plt.switch_backend('Agg')

from train import FocalDiceLoss
def set_seed(seed=42):
    """设置随机种子确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
def worker_init_fn(worker_id):
    """Worker初始化函数，确保每个worker的随机性一致"""
    # 设置worker特定的种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# 添加项目路径
sys.path.append('/home/huangzh/myreal')

# 导入项目模块
from dataset import create_crack_datasets, orginal_collate_fn
from models.segmentation_model import get_segmentation_model
from models.wavelet_enhanced_model import get_wavelet_enhanced_model
from utils import _otsu_threshold, calculate_metrics, plot_training_curves, visualize_predictions
# 导入可能需要的自适应阈值函数
try:
    from utils import binary_threshold_adaptive
except ImportError:
    # 如果没有该函数，使用_otsu_threshold实现
    def binary_threshold_adaptive(prob_map, method='otsu'):
        if method == 'otsu':
            thr = _otsu_threshold(prob_map)
            return (prob_map > thr).astype(np.float32), thr
        else:
            raise NotImplementedError(f"Method {method} not implemented")

def parse_args():
    parser = argparse.ArgumentParser(description='Crack Segmentation Testing')
    parser.add_argument('--data_path', type=str, default='CFD', help='path to dataset')
    parser.add_argument('--weights', type=str, default='best_model.pth', help='path to best_model.pth')
    parser.add_argument('--save_dir', type=str, default='test_results', help='where to save imgs + csv')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--image_size', nargs=2, type=int, default=[512, 512])

    # 与训练对齐的增强开关
    parser.add_argument('--use_wavelet', action='store_true', help='Use wavelet enhancement')
    parser.add_argument('--use_loggabor', action='store_true', help='Use LogGabor enhancement')
    parser.add_argument('--use_fusion', action='store_true', help='Use WT Low + Gabor High fusion branch')

    # 与训练对齐的阈值策略
    parser.add_argument('--use_otsu', action='store_true', default=False)
    parser.add_argument('--fixed_threshold', type=float, default=0.109804, help='fixed threshold for binarization')
    parser.add_argument('--no-otsu', dest='use_otsu', action='store_false')

    # OCR 通道数
    parser.add_argument('--ocr_mid_channels', type=int, default=512)
    parser.add_argument('--ocr_key_channels', type=int, default=256)
    
    # 可视化相关参数
    parser.add_argument('--viz_samples', type=int, default=4, help='number of samples to visualize per batch')
    parser.add_argument('--viz_dpi', type=int, default=300, help='dpi for saved visualization images')

    return parser.parse_args()


def build_model(args):
    if args.use_wavelet or args.use_loggabor or args.use_fusion:
        model = get_wavelet_enhanced_model(
            num_classes=1,
            ocr_mid_channels=args.ocr_mid_channels,
            ocr_key_channels=args.ocr_key_channels,
            use_wavelet=args.use_wavelet,
            use_loggabor=args.use_loggabor,
            use_fusion=args.use_fusion
        )
    else:
        model = get_segmentation_model(
            num_classes=1,
            ocr_mid_channels=args.ocr_mid_channels,
            ocr_key_channels=args.ocr_key_channels
        )
    return model


def threshold_mask(prob_map, use_otsu, fixed_thr):
    """
    prob_map: np.array [H,W] 0~1
    return: binary mask [H,W] uint8
    """
    if use_otsu:
        thr = _otsu_threshold(prob_map)
    else:
        thr = fixed_thr
    return (prob_map > thr).astype(np.uint8)

criterion = FocalDiceLoss(alpha=0.25, gamma=2.0, dice_weight=0.6)


def visualize_batch(args, batch_idx, images, masks, probabilities, enhancement_viz=None):
    """
    可视化一个批次的结果并保存
    args: 命令行参数
    batch_idx: 批次索引
    images: 输入图像 [B,3,H,W]
    masks: 真实掩码 [B,1,H,W]
    probabilities: 预测概率图 [B,1,H,W]
    enhancement_viz: 增强分支的可视化结果（如果有）
    """
    num_samples = min(args.viz_samples, len(images))
    has_enhancement = args.use_wavelet or args.use_loggabor or args.use_fusion
    fixed_threshold = args.fixed_threshold
    
    # 计算列数
    if has_enhancement and enhancement_viz is not None:
        num_cols = 2  # 原图 + 真实mask
        if args.use_wavelet and enhancement_viz.get('wavelet') is not None:
            num_cols += 1
        if args.use_loggabor and enhancement_viz.get('loggabor') is not None:
            num_cols += 1
        if args.use_fusion and enhancement_viz.get('fusion_combined') is not None:
            num_cols += 3  # WT低频 + Gabor高频 + 融合结果
        num_cols += 2  # 预测概率图 + 预测二值图
    else:
        num_cols = 4  # 原图 + 真实mask + 预测概率图 + 预测二值图
    
    # 创建画布
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(5*num_cols, 5*num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]  # 确保是二维数组
    
    # 设置总标题（保持不变）
    threshold_method = "Otsu" if args.use_otsu else f"fixed_threshold: ({fixed_threshold:.3f})"
    enhancement_types = []
    if args.use_wavelet:
        enhancement_types.append("Wavelet")
    if args.use_loggabor:
        enhancement_types.append("LogGabor")
    if args.use_fusion:
        enhancement_types.append("Fusion")
    enhancement_str = " + ".join(enhancement_types) if enhancement_types else "None"
    fig.suptitle(f'Batch {batch_idx:03d} | {enhancement_str} | {threshold_method}\n\n', fontsize=16)
    
    
    for i in range(num_samples):
        col_idx = 0
        
        # 1. 原图（反归一化）
        img_path = images[i]  # images_path 是图片路径列表，每个元素是单个图片路径
        image = Image.open(img_path)
        image = image.resize((512, 512), Image.Resampling.LANCZOS)  # Python3.9+
        axes[i, col_idx].imshow(image)
        # 只在第一行设置标题，增大字体到14
        if i == 0:
            axes[i, col_idx].set_title('Original', fontsize=48)
        axes[i, col_idx].axis('off')
        col_idx += 1
        
        # 2. 增强分支可视化（如果有）
        if has_enhancement and enhancement_viz is not None:
            # 小波处理图
            if args.use_wavelet and enhancement_viz.get('wavelet') is not None:
                wavelet_img = enhancement_viz['wavelet'][i]
                if len(wavelet_img.shape) == 3:
                    if wavelet_img.shape[0] == 3:
                        # 3通道转灰度
                        wavelet_img_gray = np.mean(wavelet_img.cpu().numpy(), axis=0)
                    else:
                        wavelet_img_gray = wavelet_img.cpu().numpy().squeeze()
                else:
                    wavelet_img_gray = wavelet_img.cpu().numpy().squeeze()
                # 归一化到0-1
                wavelet_img_gray = (wavelet_img_gray - wavelet_img_gray.min()) / (wavelet_img_gray.max() - wavelet_img_gray.min() + 1e-8)
                axes[i, col_idx].imshow(wavelet_img_gray, cmap='gray')
                # 只在第一行设置标题，增大字体到14
                if i == 0:
                    axes[i, col_idx].set_title('Wavelet', fontsize=48)
                axes[i, col_idx].axis('off')
                col_idx += 1
            
            # LogGabor处理图
            if args.use_loggabor and enhancement_viz.get('loggabor') is not None:
                loggabor_img = enhancement_viz['loggabor'][i]
                if len(loggabor_img.shape) == 3:
                    if loggabor_img.shape[0] == 3:
                        # 3通道转灰度
                        loggabor_img_gray = np.mean(loggabor_img.cpu().numpy(), axis=0)
                    else:
                        loggabor_img_gray = loggabor_img.cpu().numpy().squeeze()
                else:
                    loggabor_img_gray = loggabor_img.cpu().numpy().squeeze()
                # 归一化到0-1
                loggabor_img_gray = (loggabor_img_gray - loggabor_img_gray.min()) / (loggabor_img_gray.max() - loggabor_img_gray.min() + 1e-8)
                axes[i, col_idx].imshow(loggabor_img_gray, cmap='gray')
                # 只在第一行设置标题，增大字体到14
                if i == 0:
                    axes[i, col_idx].set_title('LogGabor', fontsize=48)
                axes[i, col_idx].axis('off')
                col_idx += 1
            
            # 融合分支可视化
            if args.use_fusion and enhancement_viz.get('fusion_combined') is not None:
                # WT低频分量
                if 'fusion_wt_low' in enhancement_viz:
                    wt_low_img = enhancement_viz['fusion_wt_low'][i]
                    wt_low_gray = wt_low_img.cpu().numpy().squeeze()
                    # 归一化到0-1
                    wt_low_gray = (wt_low_gray - wt_low_gray.min()) / (wt_low_gray.max() - wt_low_gray.min() + 1e-8)
                    axes[i, col_idx].imshow(wt_low_gray, cmap='gray')
                    # 只在第一行设置标题，增大字体到14
                    if i == 0:
                        axes[i, col_idx].set_title('WT (Low)', fontsize=48)
                    axes[i, col_idx].axis('off')
                    col_idx += 1
                
                # Gabor高频分量
                if 'fusion_gabor_high' in enhancement_viz:
                    gabor_high_img = enhancement_viz['fusion_gabor_high'][i]
                    gabor_high_gray = gabor_high_img.cpu().numpy().squeeze()
                    # 归一化到0-1
                    gabor_high_gray = (gabor_high_gray - gabor_high_gray.min()) / (gabor_high_gray.max() - gabor_high_gray.min() + 1e-8)
                    axes[i, col_idx].imshow(gabor_high_gray, cmap='gray')
                    # 只在第一行设置标题，增大字体到14
                    if i == 0:
                        axes[i, col_idx].set_title('Gabor (High)', fontsize=48)
                    axes[i, col_idx].axis('off')
                    col_idx += 1
                
                # 融合结果
                fusion_img = enhancement_viz['fusion_combined'][i]
                fusion_gray = fusion_img.cpu().numpy().squeeze()
                # 归一化到0-1
                fusion_gray = (fusion_gray - fusion_gray.min()) / (fusion_gray.max() - fusion_gray.min() + 1e-8)
                axes[i, col_idx].imshow(fusion_gray, cmap='gray')
                # 只在第一行设置标题，增大字体到14
                if i == 0:
                    axes[i, col_idx].set_title('Fusion', fontsize=48)
                axes[i, col_idx].axis('off')
                col_idx += 1
        
        # 3. 真实mask（二值）
        mask_np = masks[i].cpu().numpy().squeeze()  # [H,W]
        axes[i, col_idx].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        # 只在第一行设置标题，增大字体到14
        if i == 0:
            axes[i, col_idx].set_title('Ground Truth', fontsize=48)
        axes[i, col_idx].axis('off')
        col_idx += 1
        
        # 4. 预测概率图（0-1连续值）
        pred_prob = probabilities[i, 0].cpu().numpy()  # [H,W]
        im_prob = axes[i, col_idx].imshow(pred_prob, cmap='gray', vmin=0, vmax=1)
        if args.use_otsu:
            # 使用Otsu自适应阈值
            pred_binary, adaptive_threshold = binary_threshold_adaptive(pred_prob, method='otsu')
            axes[i, col_idx].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
            # 只在第一行设置标题，增大字体到14  (Otsu{adaptive_threshold:.3f})
            if i == 0:
                axes[i, col_idx].set_title(f'Prediction', fontsize=48)
        else:
            # 使用固定阈值
            pred_binary = (pred_prob > fixed_threshold).astype(np.float32)
            axes[i, col_idx].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
            # 只在第一行设置标题，增大字体到14 ({fixed_threshold:.3f})
            if i == 0:
                axes[i, col_idx].set_title(f'Prediction', fontsize=48)
        axes[i, col_idx].axis('off')
        col_idx += 1
        
        # 5. 预测二值图
        # 只在第一行设置标题，增大字体到14
        if i == 0:
            axes[i, col_idx].set_title('Prediction', fontsize=48)
        axes[i, col_idx].axis('off')
        # 添加颜色条
        plt.colorbar(im_prob, ax=axes[i, col_idx], fraction=0.046, pad=0.04)

        
    
    # 调整布局并保存
    plt.tight_layout()
    save_path = os.path.join(args.save_dir, f'visualization_batch_{batch_idx:03d}.png')
    plt.savefig(save_path, dpi=args.viz_dpi, bbox_inches='tight')
    plt.close()
    print(f"Batch {batch_idx:03d} visualization saved to: {save_path}")


@torch.no_grad()
def main():
    args = parse_args()
    enhancement_types = []
    if args.use_wavelet:
        enhancement_types.append("小波")
    if args.use_loggabor:
        enhancement_types.append("LogGabor")
    if args.use_fusion:
        enhancement_types.append("融合分支")
    print(f"使用{' + '.join(enhancement_types)}增强模型")
    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    # ---------- 数据集 ----------
    test_img_dir = os.path.join(args.data_path, 'images', 'validation')
    test_mask_dir = os.path.join(args.data_path, 'annotations', 'validation')
    from dataset import CrackDataset
    test_set = CrackDataset(
        images_dir=test_img_dir,
        masks_dir=test_mask_dir,
        image_size=tuple(args.image_size),
        is_training=False,  # 无增强
        seed=42
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=orginal_collate_fn,
        worker_init_fn=worker_init_fn
    )

    # ---------- 模型 ----------
    model = build_model(args)
    ckpt = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(args.device)
    model.eval()
    print(f'Loaded weights from {args.weights}  (best {ckpt.get("best_metric","?")}: {ckpt.get("best_metric_value","?")})')

    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_thresholds = []  # 记录所有阈值

    # ---------- 测试 ----------
    for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
        images = batch['images'].to(args.device)
        masks  = batch['masks'].to(args.device).unsqueeze(1).float()  # (B,1,H,W)
        images_path = batch['image_path']
        # 前向传播
        logits = model(images)
        enhancement_viz = model.get_visualization(images)
        
        # 计算损失 —— 使用 FocalDiceLoss
        loss = criterion(logits, masks)

        probabilities = torch.tanh(logits)  # logits -> 概率（0-1）
        
        # 生成二值预测
        if args.use_otsu:
            # Otsu阈值需要逐样本处理
            pred_binary_list = []
            batch_thresholds = []
            for i in range(len(probabilities)):
                prob_np = probabilities[i, 0].cpu().numpy()
                pred_binary, thr = binary_threshold_adaptive(prob_np, method='otsu')
                pred_binary_list.append(torch.from_numpy(pred_binary).unsqueeze(0).to(args.device))
                batch_thresholds.append(thr)
            pred_binary = torch.cat(pred_binary_list, dim=0)
            all_thresholds.extend(batch_thresholds)
        else:
            # 固定阈值
            pred_binary = (probabilities > args.fixed_threshold).float().squeeze(1)
            all_thresholds.extend([args.fixed_threshold] * len(images))
        
        # 记录结果
        all_predictions.append(pred_binary)
        all_targets.append(masks.squeeze(1))  # 保持维度一致
        total_loss += loss.item()
        
        # 可视化当前批次并保存
        visualize_batch(args, batch_idx, images_path, masks, probabilities, enhancement_viz)

    # 计算整体指标
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    if all_predictions and all_targets:
        predictions_tensor = torch.cat(all_predictions, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(predictions_tensor, targets_tensor)
        metrics['avg_loss'] = avg_loss
    else:
        metrics = {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 
            'f1': 0.0, 'iou': 0.0, 'dice': 0.0, 'avg_loss': avg_loss
        }
    
    # 打印结果
    print(f"\nTest Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 保存指标到CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(args.save_dir, 'test_metrics.csv'), index=False)
    print(f"\nMetrics saved to: {os.path.join(args.save_dir, 'test_metrics.csv')}")


if __name__ == '__main__':
    main()