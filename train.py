#!/usr/bin/env python3
"""
裂缝分割训练脚本 - OCR版本
"""

import os
import sys
import matplotlib
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
os.environ.pop("DISPLAY", None)      # 保险起见
matplotlib.use("Agg") 
# 添加项目路径
sys.path.append('/home/huangzh/myreal')

# 导入项目模块
from dataset import create_crack_datasets, simple_collate_fn
from models.segmentation_model import get_segmentation_model
from models.wavelet_enhanced_model import get_wavelet_enhanced_model
from utils import calculate_metrics, plot_training_curves, visualize_predictions

from monai.losses import DiceLoss, FocalLoss


def worker_init_fn(worker_id):
    """Worker初始化函数，确保每个worker的随机性一致"""
    # 设置worker特定的种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 设置随机种子
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

# 日志和输出目录管理
def get_new_output_dir(base_dir="outputs"):
    now = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, now)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def visualize_and_save(model, val_loader, device, save_path, epoch, use_wavelet=False, use_loggabor=False, use_fusion=False, use_otsu=True, fixed_threshold=0.0):
    """可视化并保存结果"""
    
    def save_probability_table(pred_prob, save_dir, image_id, epoch):
        """保存概率图的像素值到CSV表格"""
        # 获取图像尺寸
        height, width = pred_prob.shape
        
        # 创建坐标网格
        y_coords, x_coords = np.meshgrid(range(height), range(width), indexing='ij')
        
        # 展平数据
        data = {
            'row': y_coords.flatten(),
            'col': x_coords.flatten(),
            'probability': pred_prob.flatten()
        }
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 保存到CSV
        csv_filename = f'probability_table_epoch_{epoch}_image_{image_id}.csv'
        csv_path = os.path.join(save_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        
        # 显示统计信息
        logging.info(f"图像{image_id}概率表已保存到: {csv_path}")
        logging.info(f"概率值统计 - 最小值: {df['probability'].min():.4f}, 最大值: {df['probability'].max():.4f}, 平均值: {df['probability'].mean():.4f}")
        
        return df
    
    model.eval()
    
    # 获取一个批次的数据
    batch = next(iter(val_loader))
    images = batch['images'].to(device)
    masks = batch['masks'].to(device)
    
    with torch.no_grad():
        # 模型输出logits，然后转换为概率
        logits = model(images)
        probabilities = torch.tanh(logits) # logits -> 概率
    
    # 根据是否使用增强调整可视化布局
    has_enhancement = use_wavelet or use_loggabor or use_fusion
    if has_enhancement and hasattr(model, 'get_visualization'):
        # 获取增强处理的可视化结果
        enhancement_viz = model.get_visualization(images)
        
        # 计算可视化列数
        num_cols = 2  # 原图 + 真实mask
        if use_wavelet and enhancement_viz.get('wavelet') is not None:
            num_cols += 1
        if use_loggabor and enhancement_viz.get('loggabor') is not None:
            num_cols += 1
        if use_fusion and enhancement_viz.get('fusion_combined') is not None:
            num_cols += 3  # WT低频 + Gabor高频 + 融合结果
        num_cols += 2  # 预测概率图 + 预测二值图
        
        # 可视化前4个样本
        fig, axes = plt.subplots(4, num_cols, figsize=(5*num_cols, 20))
        
        # 设置标题
        enhancement_types = []
        if use_wavelet:
            enhancement_types.append("Wavelet")
        if use_loggabor:
            enhancement_types.append("LogGabor")
        if use_fusion:
            enhancement_types.append("Fusion")
        
        # 添加阈值方式信息
        threshold_method = "Otsu自适应阈值" if use_otsu else f"固定阈值({fixed_threshold})"
        
        title = "" 
        fig.suptitle(title, fontsize=16)
        
        for i in range(min(4, len(images))):
            col_idx = 0
            
            # 原图
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            axes[i, col_idx].imshow(img_np)
            axes[i, col_idx].set_title('Original Image')
            axes[i, col_idx].axis('off')
            col_idx += 1
            
            # 小波处理图
            if use_wavelet and enhancement_viz.get('wavelet') is not None:
                wavelet_img = enhancement_viz['wavelet'][i]
                if len(wavelet_img.shape) == 3 and wavelet_img.shape[0] == 3:
                    # 如果是3通道，转换为灰度
                    wavelet_img_gray = np.mean(wavelet_img.cpu().numpy(), axis=0)
                else:
                    wavelet_img_gray = wavelet_img.cpu().numpy().squeeze()
                axes[i, col_idx].imshow(wavelet_img_gray, cmap='gray')
                axes[i, col_idx].set_title('Wavelet Processed')
                axes[i, col_idx].axis('off')
                col_idx += 1
            
            # LogGabor处理图
            if use_loggabor and enhancement_viz.get('loggabor') is not None:
                loggabor_img = enhancement_viz['loggabor'][i]
                if len(loggabor_img.shape) == 3 and loggabor_img.shape[0] == 3:
                    # 如果是3通道，转换为灰度
                    loggabor_img_gray = np.mean(loggabor_img.cpu().numpy(), axis=0)
                else:
                    loggabor_img_gray = loggabor_img.cpu().numpy().squeeze()
                axes[i, col_idx].imshow(loggabor_img_gray, cmap='gray')
                axes[i, col_idx].set_title('LogGabor Processed')
                axes[i, col_idx].axis('off')
                col_idx += 1
            
            # 融合分支可视化
            if use_fusion and enhancement_viz.get('fusion_combined') is not None:
                # WT低频分量
                wt_low_img = enhancement_viz['fusion_wt_low'][i]
                wt_low_gray = wt_low_img.cpu().numpy().squeeze()
                axes[i, col_idx].imshow(wt_low_gray, cmap='gray')
                axes[i, col_idx].set_title('WT Low Freq (LL)')
                axes[i, col_idx].axis('off')
                col_idx += 1
                
                # Gabor高频分量
                gabor_high_img = enhancement_viz['fusion_gabor_high'][i]
                gabor_high_gray = gabor_high_img.cpu().numpy().squeeze()
                axes[i, col_idx].imshow(gabor_high_gray, cmap='gray')
                axes[i, col_idx].set_title('Gabor High Freq (Response-Input)')
                axes[i, col_idx].axis('off')
                col_idx += 1
                
                # 融合结果
                fusion_img = enhancement_viz['fusion_combined'][i]
                fusion_gray = fusion_img.cpu().numpy().squeeze()
                axes[i, col_idx].imshow(fusion_gray, cmap='gray')
                axes[i, col_idx].set_title('Fusion Combined')
                axes[i, col_idx].axis('off')
                col_idx += 1
            
            # 真实mask（二值）
            mask_np = masks[i].cpu().numpy()
            axes[i, col_idx].imshow(mask_np, cmap='gray')
            axes[i, col_idx].set_title('Ground Truth (Binary)')
            axes[i, col_idx].axis('off')
            col_idx += 1
            
            # 预测概率图（0-1连续值）
            pred_prob = probabilities[i, 0].cpu().numpy()
            im_prob = axes[i, col_idx].imshow(pred_prob, cmap='gray', vmin=0, vmax=1)
            axes[i, col_idx].set_title('Prediction (Probability)')
            axes[i, col_idx].axis('off')
            plt.colorbar(im_prob, ax=axes[i, col_idx], fraction=0.046, pad=0.04)
            col_idx += 1
            
            # 预测二值图（根据阈值方式选择）
            if use_otsu:
                # 使用Otsu自适应阈值
                from utils import binary_threshold_adaptive
                pred_binary, adaptive_threshold = binary_threshold_adaptive(pred_prob, method='otsu')
                axes[i, col_idx].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
                axes[i, col_idx].set_title(f'Prediction (Binary, Otsu thresh={adaptive_threshold:.3f})')
            else:
                # 使用固定阈值
                pred_binary = (pred_prob > fixed_threshold).astype(np.float32)
                axes[i, col_idx].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
                axes[i, col_idx].set_title(f'Prediction (Binary, Fixed thresh={fixed_threshold:.3f})')
            axes[i, col_idx].axis('off')
    else:
        logging.info("without enhancement visualization")
        # 原始布局
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        
        # 添加阈值方式信息
        threshold_method = "Otsu自适应阈值" if use_otsu else f"固定阈值({fixed_threshold})"
        
        fig.suptitle(f'', fontsize=16)
        
        for i in range(min(4, len(images))):
            # 原图
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # 真实mask（二值）
            mask_np = masks[i].cpu().numpy()
            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title('Ground Truth (Binary)')
            axes[i, 1].axis('off')
            
            # 预测概率图（0-1连续值）
            pred_prob = probabilities[i, 0].cpu().numpy()
            im_prob = axes[i, 2].imshow(pred_prob, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title('Prediction (Probability)')
            axes[i, 2].axis('off')
            plt.colorbar(im_prob, ax=axes[i, 2], fraction=0.046, pad=0.04)
            
            # 预测二值图（根据阈值方式选择）
            if use_otsu:
                # 使用Otsu自适应阈值
                from utils import binary_threshold_adaptive
                pred_binary, adaptive_threshold = binary_threshold_adaptive(pred_prob, method='otsu')
                axes[i, 3].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
                axes[i, 3].set_title(f'Prediction (Binary, Otsu thresh={adaptive_threshold:.3f})')
            else:
                # 使用固定阈值
                pred_binary = (pred_prob > fixed_threshold).astype(np.float32)
                axes[i, 3].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
                axes[i, 3].set_title(f'Prediction (Binary, Fixed thresh={fixed_threshold:.3f})')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'visualization_epoch_{epoch}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"可视化结果已保存到: {os.path.join(save_path, f'visualization_epoch_{epoch}.png')}")


class CosineScheduler:
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0):
        self.base_value = base_value
        self.final_value = final_value
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.start_warmup_value = start_warmup_value
        self.schedule = self._compute_schedule()
    
    def _compute_schedule(self):
        """计算学习率调度表"""
        schedule = []
        
        for i in range(self.total_iters):
            if i < self.warmup_iters:
                # Warmup阶段：线性增加
                lr = self.start_warmup_value + (self.base_value - self.start_warmup_value) * i / self.warmup_iters
            else:
                # Cosine衰减阶段
                progress = (i - self.warmup_iters) / (self.total_iters - self.warmup_iters)
                lr = self.final_value + 0.5 * (self.base_value - self.final_value) * (1 + math.cos(math.pi * progress))
            
            schedule.append(lr)
        
        return schedule
    
    def __getitem__(self, index):
        return self.schedule[index]

def build_dinov2_schedulers(cfg):
    """
    Args:
        cfg: 配置对象，包含训练参数
    
    Returns:
        tuple: (lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule)
    """
    OFFICIAL_EPOCH_LENGTH = cfg.get('OFFICIAL_EPOCH_LENGTH', 100)  # 每epoch的迭代次数
    
    # 学习率调度
    lr = dict(
        base_value=cfg.get('lr', 0.001),
        final_value=cfg.get('min_lr', 0.00001),
        total_iters=cfg.get('epochs', 200) * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.get('warmup_epochs', 1) * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    
    # 权重衰减调度
    wd = dict(
        base_value=cfg.get('weight_decay', 0.04),
        final_value=cfg.get('weight_decay_end', 0.4),
        total_iters=cfg.get('epochs', 200) * OFFICIAL_EPOCH_LENGTH,
    )
    
    # 动量调度
    momentum = dict(
        base_value=cfg.get('momentum_teacher', 0.996),
        final_value=cfg.get('final_momentum_teacher', 1.0),
        total_iters=cfg.get('epochs', 200) * OFFICIAL_EPOCH_LENGTH,
    )
    
    # 教师温度调度
    teacher_temp = dict(
        base_value=cfg.get('teacher_temp', 0.04),
        final_value=cfg.get('teacher_temp', 0.04),
        total_iters=cfg.get('warmup_teacher_temp_epochs', 30) * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.get('warmup_teacher_temp_epochs', 30) * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.get('warmup_teacher_temp', 0.04),
    )
    
    # 创建调度器
    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)
    
    # 冻结最后一层的调度
    freeze_epochs = cfg.get('freeze_last_layer_epochs', 0)
    if freeze_epochs > 0:
        freeze_iters = freeze_epochs * OFFICIAL_EPOCH_LENGTH
        last_layer_lr_schedule.schedule[:freeze_iters] = [0] * freeze_iters
    
    logging.info("DINOv2 schedulers ready.")
    
    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )

def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    """
    应用优化器调度
    
    Args:
        optimizer: 优化器
        lr: 学习率
        wd: 权重衰减
        last_layer_lr: 最后一层学习率
    """
    for param_group in optimizer.param_groups:
        is_last_layer = param_group.get("is_last_layer", False)
        lr_multiplier = param_group.get("lr_multiplier", 1.0)
        wd_multiplier = param_group.get("wd_multiplier", 1.0)
        
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier

def get_param_groups(model, base_lr, base_wd):
    """
    获取参数组，用于分层学习率
    
    Args:
        model: 模型
        base_lr: 基础学习率
        base_wd: 基础权重衰减
    
    Returns:
        list: 参数组列表
    """
    param_groups = []
    
    # 获取模型参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 判断是否为最后一层
        is_last_layer = "cls_head" in name or "head" in name
        
        param_groups.append({
            "params": param,
            "lr": base_lr,
            "weight_decay": base_wd,
            "is_last_layer": is_last_layer,
            "lr_multiplier": 1.0,
            "wd_multiplier": 1.0,
        })
    
    return param_groups

def get_cosine_warmup_scheduler(optimizer, total_epochs, warmup_epochs=1, T_0=50, T_mult=2, min_lr_factor=0.01):
    """
    带预热的余弦退火学习率调度器
    
    Args:
        optimizer: 优化器
        total_epochs: 总训练轮数
        warmup_epochs: 预热轮数
        T_0: 第一个周期长度
        T_mult: 周期倍数
        min_lr_factor: 最小学习率因子（相对于初始学习率）
    
    Returns:
        学习率调度器
    """
    # 创建一个自定义的LambdaLR调度器，手动实现预热+余弦退火
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 预热阶段：从0.1倍学习率线性增加到1倍学习率
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        else:
            # 余弦退火阶段：使用余弦函数从1.0降到min_lr_factor
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr_factor + (1.0 - min_lr_factor) * 0.5 * (1 + math.cos(math.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def plot_lr_schedule(total_epochs, warmup_epochs, base_lr, min_lr_factor=0.01, save_path=None):
    """
    绘制学习率调度曲线
    
    Args:
        total_epochs: 总训练轮数
        warmup_epochs: 预热轮数
        base_lr: 基础学习率
        min_lr_factor: 最小学习率因子
        save_path: 保存路径
    """
    epochs = range(total_epochs)
    lrs = []
    
    for epoch in epochs:
        if epoch < warmup_epochs:
            # 预热阶段：从0.1倍学习率线性增加到1倍学习率
            lr = base_lr * (0.1 + 0.9 * (epoch / warmup_epochs))
        else:
            # 余弦退火阶段：使用余弦函数从1.0降到min_lr_factor
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            lr = base_lr * (min_lr_factor + (1.0 - min_lr_factor) * 0.5 * (1 + math.cos(math.pi * progress)))
        lrs.append(lr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lrs, linewidth=2)
    plt.title('Fixed Learning Rate Schedule (Warmup + Cosine Annealing)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 标记关键点
    plt.axvline(x=warmup_epochs, color='red', linestyle='--', alpha=0.7, label=f'Warmup End (Epoch {warmup_epochs})')
    plt.axhline(y=base_lr, color='green', linestyle='--', alpha=0.7, label=f'Base LR ({base_lr})')
    plt.axhline(y=base_lr * min_lr_factor, color='orange', linestyle='--', alpha=0.7, label=f'Min LR ({base_lr * min_lr_factor:.6f})')
    
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"学习率调度曲线已保存到: {save_path}")
    else:
        plt.show()


def plot_dinov2_lr_schedule(cfg, save_path=None):
    """
    Args:
        cfg: 配置参数
        save_path: 保存路径
    """
    # 构建调度器
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_dinov2_schedulers(cfg)
    
    total_epochs = cfg['epochs']
    OFFICIAL_EPOCH_LENGTH = cfg['OFFICIAL_EPOCH_LENGTH']
    
    # 计算每个epoch的平均学习率
    epoch_lrs = []
    for epoch in range(total_epochs):
        start_iter = epoch * OFFICIAL_EPOCH_LENGTH
        end_iter = min((epoch + 1) * OFFICIAL_EPOCH_LENGTH, total_epochs * OFFICIAL_EPOCH_LENGTH)
        avg_lr = np.mean(lr_schedule.schedule[start_iter:end_iter])
        epoch_lrs.append(avg_lr)
    
    plt.figure(figsize=(12, 8))
    
    # 主图 - 学习率调度
    plt.subplot(2, 2, 1)
    plt.plot(range(total_epochs), epoch_lrs, linewidth=2, color='blue')
    plt.title('DINOv2 Learning Rate Schedule', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Learning Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 标记关键点
    plt.axvline(x=cfg['warmup_epochs'], color='red', linestyle='--', alpha=0.7, label=f'Warmup End')
    plt.axhline(y=cfg['lr'], color='green', linestyle='--', alpha=0.7, label=f'Base LR')
    plt.axhline(y=cfg['min_lr'], color='orange', linestyle='--', alpha=0.7, label=f'Min LR')
    plt.legend()
    
    # 权重衰减调度
    plt.subplot(2, 2, 2)
    epoch_wds = []
    for epoch in range(total_epochs):
        start_iter = epoch * OFFICIAL_EPOCH_LENGTH
        end_iter = min((epoch + 1) * OFFICIAL_EPOCH_LENGTH, total_epochs * OFFICIAL_EPOCH_LENGTH)
        avg_wd = np.mean(wd_schedule.schedule[start_iter:end_iter])
        epoch_wds.append(avg_wd)
    
    plt.plot(range(total_epochs), epoch_wds, linewidth=2, color='green')
    plt.title('Weight Decay Schedule', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Weight Decay', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 迭代级别的学习率变化
    plt.subplot(2, 2, 3)
    early_iters = min(1000, total_epochs * OFFICIAL_EPOCH_LENGTH)
    plt.plot(range(early_iters), lr_schedule.schedule[:early_iters], linewidth=1, color='purple')
    plt.title('LR at Iteration Level (First 1000 iters)', fontsize=12)
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Learning Rate', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 学习率变化率
    plt.subplot(2, 2, 4)
    lr_changes = np.diff(lr_schedule.schedule[:early_iters])
    plt.plot(range(1, early_iters), lr_changes, linewidth=1, color='orange')
    plt.title('Learning Rate Change Rate', fontsize=12)
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('LR Change', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"DINOv2学习率调度曲线已保存到: {save_path}")
    else:
        plt.show()


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.6):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice  = DiceLoss(to_onehot_y=False, sigmoid=True)  # 内部已做 sigmoid
        self.w     = dice_weight

    def forward(self, pred, mask):
        return (1 - self.w) * self.focal(pred, mask) + self.w * self.dice(pred, mask)

criterion = FocalDiceLoss(alpha=0.25, gamma=2.0, dice_weight=0.6)

# ---------- 训练一个 epoch ----------
def train_one_epoch(model, optimizer, train_loader, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0

    # 实例化损失函数（可放到外面，避免每个 batch 都 new）

    train_loader_tqdm = tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{num_epochs}', leave=False)
    for batch_idx, batch in enumerate(train_loader_tqdm):
        images = batch['images'].to(device)
        masks  = batch['masks'].to(device).unsqueeze(1).float()  # (B,1,H,W)

        # 前向
        logits = model(images)          # (B,1,H,W)

        # 计算损失
        loss = criterion(logits, masks)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_loader_tqdm.set_postfix({'loss': f'{loss.item():.4f}'})
        logging.info(f"Epoch {epoch+1}/{num_epochs} 训练损失 {loss.item():.7f}")

    avg_loss = total_loss / len(train_loader)
    logging.info(f"Epoch {epoch+1}/{num_epochs} 平均损失: {avg_loss:.4f}")
    return avg_loss

def validate_one_epoch(model, val_loader, device, epoch, num_epochs,
                       use_otsu=True, fixed_threshold=0.0):
    """验证一个 epoch，使用 FocalDiceLoss 计算验证损失"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_thresholds = []  # 记录所有阈值

    # 验证阶段同样使用与训练阶段相同的 criterion
    # （确保外部已经实例化：criterion = FocalDiceLoss(...)）
    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader, desc=f'Val Epoch {epoch+1}', leave=False)
        for batch in val_loader_tqdm:
            images = batch['images'].to(device)
            masks  = batch['masks'].to(device).unsqueeze(1).float()  # (B,1,H,W)

            # 前向
            logits = model(images)  # (B,1,H,W)

            # 计算损失 —— 使用 FocalDiceLoss
            loss = criterion(logits, masks)
            total_loss += loss.item()

            # 概率 & 二值化
            probabilities = torch.sigmoid(logits)  # logits -> 概率

            if use_otsu:
                from utils import binary_threshold_adaptive
                pred_binary, adaptive_threshold = binary_threshold_adaptive(
                    probabilities, method='otsu')
                pred_binary = pred_binary.squeeze(1)  # (B,H,W)
                all_thresholds.append(adaptive_threshold)
                val_loader_tqdm.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'threshold': f'{adaptive_threshold:.4f}'
                })
            else:
                pred_binary = (probabilities > fixed_threshold).float().squeeze(1)
                all_thresholds.append(fixed_threshold)
                val_loader_tqdm.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'threshold': f'{fixed_threshold:.4f}'
                })

            all_predictions.append(pred_binary)
            all_targets.append(masks.squeeze(1))  # 与 pred_binary 对齐

    # 指标计算
    if all_predictions and all_targets:
        predictions_tensor = torch.cat(all_predictions, dim=0)  # (N,H,W)
        targets_tensor     = torch.cat(all_targets, dim=0)      # (N,H,W)
        metrics = calculate_metrics(predictions_tensor, targets_tensor)
    else:
        metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                   'f1': 0.0, 'iou': 0.0, 'dice': 0.0}

    avg_loss = total_loss / len(val_loader)

    # 日志
    if use_otsu:
        thresholds_np = np.array(all_thresholds)
        logging.info(f"Epoch {epoch+1}/{num_epochs} 验证损失: {avg_loss:.4f}")
        logging.info(f"Otsu阈值统计 - 均值: {thresholds_np.mean():.6f}, "
                     f"标准差: {thresholds_np.std():.6f}, "
                     f"范围: [{thresholds_np.min():.6f}, {thresholds_np.max():.6f}], "
                     f"中位数: {np.median(thresholds_np):.6f}")
    else:
        logging.info(f"Epoch {epoch+1}/{num_epochs} 验证损失: {avg_loss:.4f}, "
                     f"固定阈值: {fixed_threshold:.6f}")
    logging.info(f"验证指标: {metrics}")

    return avg_loss, metrics

def main():
    parser = argparse.ArgumentParser(description='Crack Segmentation Training - OCR Version')
    parser.add_argument('--data_path', type=str, default='crack500', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--save_interval', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--visualize_interval', type=int, default=2, help='Visualize results every N epochs')
    parser.add_argument('--best_metric', type=str, default='iou', choices=['iou', 'f1', 'dice', 'accuracy'], help='Metric to select best model')
    parser.add_argument('--ocr_mid_channels', type=int, default=512, help='OCR mid channels')
    parser.add_argument('--ocr_key_channels', type=int, default=256, help='OCR key channels')
    parser.add_argument('--use_wavelet', action='store_true', help='Use wavelet enhancement')
    parser.add_argument('--use_loggabor', action='store_true', help='Use LogGabor enhancement')
    parser.add_argument('--use_fusion', action='store_true', help='Use WT Low + Gabor High fusion branch')
    parser.add_argument('--use_otsu', action='store_true', default=True, help='Use Otsu adaptive threshold (default: True, use --no-otsu to disable)')
    parser.add_argument('--no-otsu', dest='use_otsu', action='store_false', help='Use fixed threshold instead of Otsu')
    parser.add_argument('--fixed_threshold', type=float, default=0.0, help='Fixed threshold value when not using Otsu (default: 0.0)')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained model weights (.pth file)')

    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(42)
    
    # 创建输出目录
    output_dir = get_new_output_dir(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    log_path = os.path.join(output_dir, 'training.log')
    setup_logging(log_path)
    
    logging.info(f"Starting training with args: {args}")
    
    # 显示阈值方法信息
    if args.use_otsu:
        threshold_method = "Otsu自适应阈值"
    else:
        threshold_method = f"固定阈值({args.fixed_threshold})"
    logging.info(f"阈值方法: {threshold_method}")
    
    # 显示数据集配置信息
    logging.info(f"训练集: 启用数据增强，种子=42")
    logging.info(f"验证集: 启用数据增强，种子=42（确保确定性）")
    
    # 创建数据集
    train_dataset, val_dataset = create_crack_datasets(
        data_root=args.data_path,
        image_size=(512, 512),
        batch_size=args.batch_size,
        seed=42  # 传递种子参数
    )
    
    # 确保验证集也使用种子和数据增强（确定性）
    # 重新创建验证集，启用数据增强但使用种子
    from dataset import CrackDataset
    
    # 定义验证集路径
    val_images_dir = os.path.join(args.data_path, 'images', 'validation')
    val_masks_dir = os.path.join(args.data_path, 'annotations', 'validation')
    
    # 重新创建验证集，启用数据增强但使用种子
    val_dataset = CrackDataset(
        images_dir=val_images_dir,
        masks_dir=val_masks_dir,
        image_size=(512, 512),
        is_training=True,  # 启用数据增强
        seed=42  # 使用种子
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=simple_collate_fn,
        worker_init_fn=worker_init_fn,  # 添加worker初始化函数
        generator=torch.Generator().manual_seed(42)  # 设置确定性生成器
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=simple_collate_fn,
        worker_init_fn=worker_init_fn  # 添加worker初始化函数
    )
    
    # 模型
    if args.use_wavelet or args.use_loggabor or args.use_fusion:
        model = get_wavelet_enhanced_model(
            num_classes=1,
            ocr_mid_channels=args.ocr_mid_channels,
            ocr_key_channels=args.ocr_key_channels,
            use_wavelet=args.use_wavelet,
            use_loggabor=args.use_loggabor,
            use_fusion=args.use_fusion
        )
        enhancement_types = []
        if args.use_wavelet:
            enhancement_types.append("小波")
        if args.use_loggabor:
            enhancement_types.append("LogGabor")
        if args.use_fusion:
            enhancement_types.append("融合分支")
        logging.info(f"使用{' + '.join(enhancement_types)}增强模型")
    else:
        model = get_segmentation_model(
            num_classes=1,
            ocr_mid_channels=args.ocr_mid_channels,
            ocr_key_channels=args.ocr_key_channels
        )
        logging.info("使用标准模型")
    if os.path.exists(args.pretrained):
        model.load_state_dict(torch.load(args.pretrained, map_location=args.device)['model_state_dict'])
        logging.info(f"成功加载预训练模型: {args.pretrained}")
    model.to(args.device)
    
    # 配置参数
    cfg = {
        'lr': args.learning_rate,
        'min_lr': args.learning_rate * 0.01,  # 最小学习率为基础学习率的1%
        'epochs': args.num_epochs,
        'warmup_epochs': 0,
        'OFFICIAL_EPOCH_LENGTH': 100,  # 每epoch的迭代次数
        'weight_decay': 0.04,
        'weight_decay_end': 0.4,
        'momentum_teacher': 0.996,
        'final_momentum_teacher': 1.0,
        'teacher_temp': 0.04,
        'warmup_teacher_temp': 0.04,
        'warmup_teacher_temp_epochs': 30,
        'freeze_last_layer_epochs': 0
    }
    
    # 获取参数组
    param_groups = get_param_groups(model, args.learning_rate, cfg['weight_decay'])
    
    # 优化器
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))
    
    # 使用修复后的余弦退火和预热调度器
    scheduler = get_cosine_warmup_scheduler(optimizer, args.num_epochs, warmup_epochs=0, T_0=25, T_mult=2)
    
    # 绘制学习率调度曲线
    plot_lr_schedule(args.num_epochs, 1, args.learning_rate, save_path=os.path.join(output_dir, 'lr_schedule.png'))
    
    # 训练循环
    best_metric_value = 0.0
    train_losses = []
    val_losses = []
    train_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'iou': [], 'dice': []}
    val_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'iou': [], 'dice': []}
    learning_rates = []  # 记录学习率变化
    
    for epoch in range(args.num_epochs):
        # 训练
        train_loss = train_one_epoch(model, optimizer, train_loader, args.device, epoch, args.num_epochs)
        
        # 验证
        val_loss, metrics = validate_one_epoch(model, val_loader, args.device, epoch, args.num_epochs, use_otsu=args.use_otsu, fixed_threshold=args.fixed_threshold)
        
        # 更新学习率
        scheduler.step(epoch)
        
        # 记录当前学习率（在scheduler.step之后）
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 打印学习率信息
        if (epoch + 1) % 10 == 0 or epoch < 5:
            logging.info(f"Epoch {epoch+1}/{args.num_epochs} - Learning Rate: {current_lr:.6f}")
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        for key in val_metrics:
            val_metrics[key].append(metrics[key])
            train_metrics[key].append(0.0)  # 训练时不计算指标，设为0
        
        # 保存最佳模型
        current_metric = metrics[args.best_metric]
        if current_metric > best_metric_value:
            best_metric_value = current_metric
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': args.best_metric,
                'best_metric_value': best_metric_value,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, os.path.join(output_dir, 'best_model.pth'))
            logging.info(f"新的最佳模型已保存，{args.best_metric}: {best_metric_value:.4f}")
            
            # 输出当前epoch的详细验证指标
            logging.info(f"Epoch {epoch+1} 验证指标详情:")
            logging.info(f"  损失: {val_loss:.4f}")
            logging.info(f"  准确率: {metrics['accuracy']:.4f}")
            logging.info(f"  精确率: {metrics['precision']:.4f}")
            logging.info(f"  召回率: {metrics['recall']:.4f}")
            logging.info(f"  F1分数: {metrics['f1']:.4f}")
            logging.info(f"  IoU: {metrics['iou']:.4f}")
            logging.info(f"  Dice系数: {metrics['dice']:.4f}")
        
        # 定期可视化（可配置间隔）
        if (epoch + 1) % args.visualize_interval == 0:
            visualize_and_save(model, val_loader, args.device, output_dir, epoch + 1, use_wavelet=args.use_wavelet, use_loggabor=args.use_loggabor, use_fusion=args.use_fusion, use_otsu=args.use_otsu, fixed_threshold=args.fixed_threshold)
        
        # 定期保存模型
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': args.best_metric,
            'best_metric_value': best_metric_value,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, os.path.join(output_dir, 'training_curves.png'))
    
    # 绘制学习率变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, linewidth=2)
    plt.title('Learning Rate During Training', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("学习率变化曲线已保存")
    
    # 输出最佳模型的验证数据
    logging.info("=" * 80)
    logging.info("最佳模型验证数据总结")
    logging.info("=" * 80)
    
    # 找到最佳模型的epoch
    best_epoch = None
    best_metrics = None
    for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        current_metric = val_metrics[args.best_metric][i]
        if current_metric == best_metric_value:
            best_epoch = i + 1
            best_metrics = {
                'epoch': best_epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'accuracy': val_metrics['accuracy'][i],
                'precision': val_metrics['precision'][i],
                'recall': val_metrics['recall'][i],
                'f1': val_metrics['f1'][i],
                'iou': val_metrics['iou'][i],
                'dice': val_metrics['dice'][i]
            }
            break
    
    if best_metrics:
        logging.info(f"最佳模型出现在第 {best_metrics['epoch']} 个epoch")
        logging.info(f"最佳指标 ({args.best_metric}): {best_metric_value:.4f}")
        logging.info("-" * 50)
        logging.info("训练损失: {:.4f}".format(best_metrics['train_loss']))
        logging.info("验证损失: {:.4f}".format(best_metrics['val_loss']))
        logging.info("-" * 50)
        logging.info("验证指标详情:")
        logging.info("  准确率 (Accuracy): {:.4f}".format(best_metrics['accuracy']))
        logging.info("  精确率 (Precision): {:.4f}".format(best_metrics['precision']))
        logging.info("  召回率 (Recall): {:.4f}".format(best_metrics['recall']))
        logging.info("  F1分数: {:.4f}".format(best_metrics['f1']))
        logging.info("  IoU: {:.4f}".format(best_metrics['iou']))
        logging.info("  Dice系数: {:.4f}".format(best_metrics['dice']))
        logging.info("-" * 50)
        
        # 保存最佳模型验证数据到CSV文件
        import csv
        best_model_csv = os.path.join(output_dir, 'best_model_validation.csv')
        with open(best_model_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['epoch', 'best_metric', 'best_metric_value', 'train_loss', 'val_loss', 
                         'accuracy', 'precision', 'recall', 'f1', 'iou', 'dice']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                'epoch': best_metrics['epoch'],
                'best_metric': args.best_metric,
                'best_metric_value': best_metric_value,
                'train_loss': best_metrics['train_loss'],
                'val_loss': best_metrics['val_loss'],
                'accuracy': best_metrics['accuracy'],
                'precision': best_metrics['precision'],
                'recall': best_metrics['recall'],
                'f1': best_metrics['f1'],
                'iou': best_metrics['iou'],
                'dice': best_metrics['dice']
            })
        logging.info(f"最佳模型验证数据已保存到: {best_model_csv}")
        
        # 输出训练过程中的最佳指标变化
        logging.info("-" * 50)
        logging.info("训练过程中最佳指标变化:")
        best_so_far = 0.0
        for i, current_metric in enumerate(val_metrics[args.best_metric]):
            if current_metric > best_so_far:
                best_so_far = current_metric
                logging.info(f"  Epoch {i+1}: {args.best_metric} = {current_metric:.4f} (新的最佳)")
            elif (i + 1) % 50 == 0:  # 每50个epoch输出一次当前最佳
                logging.info(f"  Epoch {i+1}: 当前最佳 {args.best_metric} = {best_so_far:.4f}")
        
        # 输出训练统计信息
        logging.info("-" * 50)
        logging.info("训练统计信息:")
        logging.info(f"  总训练轮数: {args.num_epochs}")
        logging.info(f"  最终训练损失: {train_losses[-1]:.4f}")
        logging.info(f"  最终验证损失: {val_losses[-1]:.4f}")
        logging.info(f"  最终{args.best_metric}: {val_metrics[args.best_metric][-1]:.4f}")
        
        # 计算指标的平均值和标准差
        final_metrics = {
            'accuracy': val_metrics['accuracy'][-10:],  # 最后10个epoch
            'precision': val_metrics['precision'][-10:],
            'recall': val_metrics['recall'][-10:],
            'f1': val_metrics['f1'][-10:],
            'iou': val_metrics['iou'][-10:],
            'dice': val_metrics['dice'][-10:]
        }
        
        logging.info("-" * 50)
        logging.info("最后10个epoch的平均指标:")
        for metric_name, values in final_metrics.items():
            if values:
                avg_value = sum(values) / len(values)
                logging.info(f"  {metric_name}: {avg_value:.4f}")
    else:
        logging.warning("未找到最佳模型信息")
    
    logging.info("=" * 80)
    logging.info("训练完成！")

if __name__ == '__main__':
    main()