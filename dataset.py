#!/usr/bin/env python3
"""
基于base.py的裂缝数据集 - OCR版本
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import random
from typing import Tuple, List, Dict, Optional

def worker_init_fn(worker_id):
    """Worker初始化函数，确保每个worker的随机性一致"""
    # 设置worker特定的种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class CrackDataset(Dataset):
    """基于base.py的裂缝数据集"""
    
    def __init__(self, 
                 images_dir: str,
                 masks_dir: str,
                 image_size: Tuple[int, int] = (512, 512),
                 is_training: bool = True,
                 seed: int = 42):
        """
        Args:
            images_dir: 图像目录
            masks_dir: 掩码目录
            image_size: 目标图像尺寸 (height, width)
            is_training: 是否为训练模式
            seed: 随机种子
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.is_training = is_training
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 获取图像和掩码文件路径
        self.image_files = []
        self.mask_files = []
        
        # 获取所有图像文件
        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(images_dir, filename)
                mask_path = os.path.join(masks_dir, filename.replace('.jpg', '.png').replace('.jpeg', '.png'))
               
                if os.path.exists(mask_path):
                    self.image_files.append(image_path)
                    self.mask_files.append(mask_path)
                elif os.path.exists(os.path.join(masks_dir, filename)):
                    mask_path = os.path.join(masks_dir, filename)
                    self.image_files.append(image_path)
                    self.mask_files.append(mask_path)
        
        # 按文件名排序，确保顺序一致
        self.image_files.sort()
        self.mask_files.sort()
        
        print(f"找到 {len(self.image_files)} 个图像-掩码对")
        
        # 创建确定性操作序列
        self._create_deterministic_ops()
    
    def _create_deterministic_ops(self):
        """创建确定性的操作序列"""
        # 为每个样本预生成操作序列，确保可重现
        # 重置随机种子状态
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.ops_sequence = []
        for i in range(len(self.image_files)):
            # 使用样本索引和种子生成确定性操作
            # 为每个样本设置特定的种子
            sample_seed = self.seed + i
            random.seed(sample_seed)
            np.random.seed(sample_seed)
            
            op = random.randint(-3, 100)
            self.ops_sequence.append(op)
        
        # 验证操作序列的确定性
        print(f"数据集种子: {self.seed}, 操作序列长度: {len(self.ops_sequence)}")
        print(f"前5个操作: {self.ops_sequence[:5]}")
    
    def _to_mask_tensor(self, im, size):
        """将掩码转换为张量"""
        # 转换为灰度图
        if im.mode != 'L':
            im = im.convert('L')
        
        # 调整尺寸
        im = im.resize(size[::-1], Image.NEAREST)  # PIL使用(width, height)
        
        # 转换为张量并二值化
        im_array = np.array(im)
        im_array = (im_array > 128).astype(np.float32)
        
        return torch.from_numpy(im_array)
    
    def _tensor_resize(self, tensor, size):
        """调整张量大小"""
        # 确保tensor是4D格式 [N, C, H, W]
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        # 调整尺寸
        resized = torch.nn.functional.interpolate(
            tensor, 
            size=size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # 返回3D格式 [C, H, W]
        return resized.squeeze(0)
    
    def read_image(self, _f, mask=False, op=0):
        """读取图像或掩码 - 基于base.py的read_image"""
        if self.is_training:
            # 训练时的数据增强
            if op % 3 == 1:
                # JPEG压缩
                frame = cv2.imread(_f)
                # 使用确定性质量
                quality = 60 + (op % 41)  # 60-100之间的确定性值
                c_ratio = [cv2.IMWRITE_JPEG_QUALITY, quality]
                msg = cv2.imencode(".jpg", frame, c_ratio)[1]
                msg = (np.array(msg)).tobytes()
                frame = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
                im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                im = Image.open(_f)
                
            # 确定性翻转
            if op % 5 == 1:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                
            # 滤镜效果（只对图像应用）
            if not mask:
                if op % 9 == 5:
                    im = im.filter(ImageFilter.DETAIL)
                elif op % 9 == 1:
                    im = im.filter(ImageFilter.GaussianBlur)
                elif op % 9 == 2:
                    im = im.filter(ImageFilter.BLUR)
                elif op % 9 == 3:
                    im = im.filter(ImageFilter.MedianFilter)
        else:
            # 验证时直接读取
            im = Image.open(_f)
        
        # 转换为张量
        if mask:
            # 掩码处理
            tensor = self._to_mask_tensor(im, self.image_size)
        else:
            # 图像处理
            img = np.array(im)
            if img.ndim == 2:  # 灰度图
                img = np.stack([img] * 3, axis=-1)
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensor = self._tensor_resize(tensor, self.image_size)
            
        im.close()
        return tensor
    
    def __getitem__(self, index):
        """获取数据项"""
        # 获取文件路径
        image_path = self.image_files[index]
        mask_path = self.mask_files[index]
        
        # 使用预生成的操作序列
        op = self.ops_sequence[index] if self.is_training else 0
        
        # 读取掩码和图像
        mask_data = self.read_image(mask_path, mask=True, op=op)
        image_data = self.read_image(image_path, mask=False, op=op)
        
        return {
            'images': image_data,
            'masks': mask_data,
            'image_path': image_path,
            'mask_path': mask_path
        }
    
    def __len__(self):
        return len(self.image_files)

def create_crack_datasets(data_root: str, 
                         image_size: Tuple[int, int] = (512, 512),
                         batch_size: int = 16,
                         seed: int = 42) -> Tuple[CrackDataset, CrackDataset]:
    """创建训练和验证数据集"""
    
    # 定义路径
    train_images_dir = os.path.join(data_root, 'images', 'training')
    train_masks_dir = os.path.join(data_root, 'annotations', 'training')
    val_images_dir = os.path.join(data_root, 'images', 'validation')
    val_masks_dir = os.path.join(data_root, 'annotations', 'validation')
    
    # 创建数据集
    train_dataset = CrackDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        image_size=image_size,
        is_training=True,
        seed=seed  # 传递种子参数
    )
    
    val_dataset = CrackDataset(
        images_dir=val_images_dir,
        masks_dir=val_masks_dir,
        image_size=image_size,
        is_training=False,
        seed=seed  # 传递种子参数
    )
    
    return train_dataset, val_dataset

def simple_collate_fn(batch):
    """简化的数据整理函数"""
    images = torch.stack([item['images'] for item in batch])
    masks = torch.stack([item['masks'] for item in batch])
    
    return {
        'images': images,
        'masks': masks
    }

def orginal_collate_fn(batch):
    return {
        # 批量 tensor（满足模型输入）
        'images': torch.stack([item['images'] for item in batch]),
        'masks': torch.stack([item['masks'] for item in batch]),
        # 原始路径列表（保留溯源能力，用于调试）
        'image_path': [item['image_path'] for item in batch],
        'mask_path': [item['mask_path'] for item in batch]
    }
