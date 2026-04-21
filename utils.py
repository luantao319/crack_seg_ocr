#!/usr/bin/env python3
"""
工具函数
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from typing import Union, Tuple
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'WenQuanYi Zen Hei']
matplotlib.rcParams['axes.unicode_minus'] = False

def calculate_adaptive_threshold(probabilities: Union[torch.Tensor, np.ndarray], 
                               method: str = 'otsu', 
                               **kwargs) -> float:
    """
    计算自适应阈值
    
    Args:
        probabilities: 概率张量或数组 [B, C, H, W] 或 [H, W]
        method: 阈值计算方法
            - 'otsu': Otsu方法（基于直方图）
            - 'triangle': 三角形方法
            - 'mean': 均值
            - 'median': 中位数
            - 'percentile': 百分位数
            - 'entropy': 熵最大化
        **kwargs: 额外参数
            - percentile: 百分位数（当method='percentile'时）
    
    Returns:
        float: 计算得到的阈值
    """
    if isinstance(probabilities, torch.Tensor):
        probs_np = probabilities.detach().cpu().numpy()
    else:
        probs_np = probabilities
    
    # 确保是2D数组
    if len(probs_np.shape) > 2:
        probs_np = probs_np.flatten()
    
    if method == 'otsu':
        return _otsu_threshold(probs_np)
    elif method == 'triangle':
        return _triangle_threshold(probs_np)
    elif method == 'mean':
        return float(np.mean(probs_np))
    elif method == 'median':
        return float(np.median(probs_np))
    elif method == 'percentile':
        percentile = kwargs.get('percentile', 75)
        return float(np.percentile(probs_np, percentile))
    elif method == 'entropy':
        return _entropy_threshold(probs_np)
    else:
        raise ValueError(f"不支持的阈值方法: {method}")

def _otsu_threshold(probs: np.ndarray) -> float:
    """Otsu阈值计算方法"""
    # 将概率值转换为0-255的整数
    probs_255 = (probs * 255).astype(np.uint8)
    
    # 计算直方图
    hist, bins = np.histogram(probs_255, bins=256, range=(0, 256))
    
    # 计算累积直方图
    total_pixels = np.sum(hist)
    if total_pixels == 0:
        return 0.5
    
    # 计算累积和
    cumsum = np.cumsum(hist)
    cumsum_norm = cumsum / total_pixels
    
    # 计算均值
    mean_total = np.sum(np.arange(256) * hist) / total_pixels
    
    # 寻找最佳阈值
    best_threshold = 0
    best_variance = 0
    
    for threshold in range(1, 256):
        # 背景类
        w0 = cumsum_norm[threshold - 1]
        if w0 == 0:
            continue
        
        # 前景类
        w1 = 1 - w0
        if w1 == 0:
            continue
        
        # 背景均值
        mean0 = np.sum(np.arange(threshold) * hist[:threshold]) / (total_pixels * w0)
        
        # 前景均值
        mean1 = np.sum(np.arange(threshold, 256) * hist[threshold:]) / (total_pixels * w1)
        
        # 类间方差
        variance = w0 * w1 * (mean0 - mean1) ** 2
        
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    
    return best_threshold / 255.0

def _triangle_threshold(probs: np.ndarray) -> float:
    """三角形阈值计算方法"""
    # 将概率值转换为0-255的整数
    probs_255 = (probs * 255).astype(np.uint8)
    
    # 计算直方图
    hist, bins = np.histogram(probs_255, bins=256, range=(0, 256))
    
    # 找到直方图的峰值
    peak_idx = np.argmax(hist)
    
    # 找到直方图的边界
    left_bound = 0
    for i in range(peak_idx):
        if hist[i] > 0:
            left_bound = i
            break
    
    right_bound = 255
    for i in range(peak_idx, 256):
        if hist[i] == 0:
            right_bound = i - 1
            break
    
    # 计算三角形阈值
    # 在峰值和边界之间找到距离最远的点
    max_distance = 0
    best_threshold = peak_idx
    
    for i in range(left_bound, right_bound + 1):
        if hist[i] > 0:
            # 计算到直线的距离
            if right_bound - left_bound > 0:
                distance = hist[i] * (right_bound - i) / (right_bound - left_bound)
                if distance > max_distance:
                    max_distance = distance
                    best_threshold = i
    
    return best_threshold / 255.0

def _entropy_threshold(probs: np.ndarray) -> float:
    """基于熵的阈值计算方法"""
    # 将概率值转换为0-255的整数
    probs_255 = (probs * 255).astype(np.uint8)
    
    # 计算直方图
    hist, bins = np.histogram(probs_255, bins=256, range=(0, 256))
    
    # 归一化直方图
    hist_norm = hist / np.sum(hist)
    
    # 计算累积熵
    best_threshold = 0
    max_entropy = 0
    
    for threshold in range(1, 256):
        # 背景类
        p0 = hist_norm[:threshold]
        p0 = p0[p0 > 0]  # 移除零概率
        
        # 前景类
        p1 = hist_norm[threshold:]
        p1 = p1[p1 > 0]  # 移除零概率
        
        if len(p0) == 0 or len(p1) == 0:
            continue
        
        # 计算熵
        entropy0 = -np.sum(p0 * np.log2(p0))
        entropy1 = -np.sum(p1 * np.log2(p1))
        
        # 加权熵
        w0 = np.sum(hist_norm[:threshold])
        w1 = np.sum(hist_norm[threshold:])
        
        total_entropy = w0 * entropy0 + w1 * entropy1
        
        if total_entropy > max_entropy:
            max_entropy = total_entropy
            best_threshold = threshold
    
    return best_threshold / 255.0

def binary_threshold_adaptive(probabilities: Union[torch.Tensor, np.ndarray], 
                            method: str = 'otsu',
                            fallback_threshold: float = 0.5,
                            **kwargs) -> Tuple[Union[torch.Tensor, np.ndarray], float]:
    """
    使用自适应阈值进行二值化
    
    Args:
        probabilities: 概率张量或数组
        method: 阈值计算方法
        fallback_threshold: 备用阈值（当自适应方法失败时使用）
        **kwargs: 传递给calculate_adaptive_threshold的参数
    
    Returns:
        Tuple: (二值化结果, 使用的阈值)
    """
    try:
        # 计算自适应阈值
        threshold = calculate_adaptive_threshold(probabilities, method, **kwargs)
        
        # 确保阈值在合理范围内
        if not (0.0 <= threshold <= 1.0):
            print(f"警告: 自适应阈值 {threshold:.4f} 超出范围[0,1]，使用备用阈值 {fallback_threshold}")
            threshold = fallback_threshold
        
        # 应用阈值
        if isinstance(probabilities, torch.Tensor):
            binary_result = (probabilities > threshold).float()
        else:
            binary_result = (probabilities > threshold).astype(np.float32)
        
        return binary_result, threshold
        
    except Exception as e:
        print(f"自适应阈值计算失败: {e}，使用备用阈值 {fallback_threshold}")
        if isinstance(probabilities, torch.Tensor):
            binary_result = (probabilities > fallback_threshold).float()
        else:
            binary_result = (probabilities > fallback_threshold).astype(np.float32)
        
        return binary_result, fallback_threshold

def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    计算分割指标
    
    Args:
        predictions: 预测掩码 [B, H, W]
        targets: 真实掩码 [B, H, W]
    
    Returns:
        包含各种指标的字典
    """
    # 转换为numpy数组
    pred_np = predictions.cpu().numpy().flatten()
    target_np = targets.cpu().numpy().flatten()
    print(pred_np.shape, target_np.shape)
    print(pred_np)
    print(target_np)
    # 计算准确率
    accuracy = np.mean(pred_np == target_np)
    
    # 计算精确率、召回率、F1分数
    tp = np.sum((pred_np == 1) & (target_np == 1))
    fp = np.sum((pred_np == 1) & (target_np == 0))
    fn = np.sum((pred_np == 0) & (target_np == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 计算IoU
    intersection = np.sum((pred_np == 1) & (target_np == 1))
    union = np.sum((pred_np == 1) | (target_np == 1))
    iou = intersection / union if union > 0 else 0
    
    # 计算Dice系数
    dice = (2 * intersection) / (np.sum(pred_np == 1) + np.sum(target_np == 1)) if (np.sum(pred_np == 1) + np.sum(target_np == 1)) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': dice
    }

def plot_training_curves(train_losses: list, val_losses: list, 
                        train_metrics: dict, val_metrics: dict, 
                        save_path: str = None) -> None:
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(train_losses, label='train_losses')
    axes[0, 0].plot(val_losses, label='val_losses')
    axes[0, 0].set_title('loss curve')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('loss')
    axes[0, 0].legend()
    
    # IoU曲线
    axes[0, 1].plot(train_metrics['iou'], label='train IoU')
    axes[0, 1].plot(val_metrics['iou'], label='val IoU')
    axes[0, 1].set_title('IoU curve')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    
    # Dice曲线
    axes[1, 0].plot(train_metrics['dice'], label='train Dice')
    axes[1, 0].plot(val_metrics['dice'], label='val Dice')
    axes[1, 0].set_title('Dice value curve')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    
    # 准确率曲线
    axes[1, 1].plot(train_metrics['accuracy'], label='train acc')
    axes[1, 1].plot(val_metrics['accuracy'], label='val acc')
    axes[1, 1].set_title('acc curve')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('acc')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存: {save_path}")
    else:
        plt.show()

def visualize_predictions(model, val_loader, device, save_path, num_samples=4):
    """可视化预测结果"""
    model.eval()
    
    # 获取一个batch的数据
    batch = next(iter(val_loader))
    images = batch['images'].to(device)
    masks = batch['masks'].to(device)
    
    with torch.no_grad():
        predictions = model(images)
        predictions = torch.sigmoid(predictions)  # 如果使用BCE损失
        predictions, adaptive_threshold = binary_threshold_adaptive(predictions, method='otsu')
    
    # 可视化
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        # 原图
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # 真实掩码
        mask = masks[i].cpu().numpy()
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # 预测掩码
        pred = predictions[i, 0].cpu().numpy()
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()