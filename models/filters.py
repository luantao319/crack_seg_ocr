import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
import cv2





class CrackOptimizedGaborFilter(nn.Module):
    """专门针对裂缝检测优化的Gabor滤波器 - GPU版本 + FC融合"""
    def __init__(self):
        super(CrackOptimizedGaborFilter, self).__init__()
        
        # 针对裂缝检测优化的参数
        orientations = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
        wavelengths = [6.0, 10.0, 14.0]
        
        # 预计算Gabor核并转换为PyTorch张量
        self.gabor_kernels = self._create_gabor_kernels(orientations, wavelengths)
        
        # FC融合层 - 处理每个空间位置
        self.fc = nn.Linear(18, 1)
    
    def _create_gabor_kernels(self, orientations, wavelengths):
        """创建Gabor卷积核"""
        kernels = []
        for wavelength in wavelengths:
            for orientation in orientations:
                # 使用OpenCV创建Gabor核
                kernel = cv2.getGaborKernel((21, 21), 2.0, orientation, wavelength, 0.3, 0)
                # 转换为PyTorch张量 [1, 1, 21, 21]
                kernel_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
                kernels.append(kernel_tensor)
        
        return nn.ParameterList(kernels)
    
    def forward(self, x):
        """
        x: [B, 3, H, W] 在GPU上
        返回: [B, 1, H, W] 在GPU上
        """
        batch_size, _, height, width = x.shape
        device = x.device
        
        # 确保所有核都在正确的设备上
        if len(self.gabor_kernels) > 0 and self.gabor_kernels[0].device != device:
            for i, kernel in enumerate(self.gabor_kernels):
                self.gabor_kernels[i] = kernel.to(device)
        
        responses = []
        
        # 对每个Gabor核进行卷积
        for kernel in self.gabor_kernels:
            # 对每个通道分别卷积
            channel_responses = []
            for c in range(3):
                # [B, 1, H, W] -> [B, 1, H, W]
                response = F.conv2d(x[:, c:c+1], kernel, padding=10)
                channel_responses.append(response)
                #print(response.shape)
            
            # 合并通道响应 [B, 3, H, W]
            response_3ch = torch.cat(channel_responses, dim=1)
            #print(response_3ch.shape)
            # 转换为灰度 [B, 1, H, W]
            response_gray = torch.mean(response_3ch, dim=1, keepdim=True)
            responses.append(response_gray)
        
        # 堆叠所有响应 [B, 18, H, W]
        if responses:
            responses_stack = torch.stack(responses, dim=1).squeeze(2)  # [B, 18, H, W]
            #print(responses_stack.shape)
            
            # 重塑为 [B*H*W, 18] 以便FC处理
            B, C, H, W = responses_stack.shape
            responses_reshaped = responses_stack.permute(0, 2, 3, 1).reshape(B*H*W, C)  # [B*H*W, 18]
            
            # FC融合 [B*H*W, 1]
            fc_output = self.fc(responses_reshaped)
            
            # 重塑回 [B, 1, H, W]
            combined = fc_output.reshape(B, H, W, 1).permute(0, 3, 1, 2)  # [B, 1, H, W]
            
            return combined
        else:
            # 如果没有响应，返回零张量
            return torch.zeros(batch_size, 1, height, width, device=device)


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    """创建小波滤波器"""
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    """小波变换"""
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    """逆小波变换"""
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class _ScaleModule(nn.Module):
    """缩放模块"""
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class AdvancedWaveletFilter(nn.Module):
    """基于MobileMamba的高级小波滤波器"""
    def __init__(self, wt_type='db1', wt_levels=1, kernel_size=3):
        super(AdvancedWaveletFilter, self).__init__()
        
        self.wt_levels = wt_levels
        self.wt_type = wt_type
        
        # 创建小波滤波器
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, 3, 3, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        
        # 小波变换函数 - 使用lambda来确保设备正确
        self.wt_function = lambda x: wavelet_transform(x, self.wt_filter)
        self.iwt_function = lambda x: inverse_wavelet_transform(x, self.iwt_filter)
        
        # 小波卷积层
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(12, 12, kernel_size, padding='same', stride=1, dilation=1,
                      groups=12, bias=False) for _ in range(self.wt_levels)
        ])
        
        # 小波缩放层
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, 12, 1, 1], init_scale=0.5) for _ in range(self.wt_levels)
        ])
        
        # 基础缩放
        self.base_scale = _ScaleModule([1, 3, 1, 1])

    def forward(self, x):
        """前向传播
        Args:
            x: (B, 3, H, W) - 输入图像
        Returns:
            (B, 3, H, W) - 小波处理后的图像
        """
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        # 小波分解
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            
            # 处理奇数尺寸
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            # 小波变换
            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]  # 低频分量

            # 对高频分量进行处理
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        # 小波重构
        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            # 恢复原始尺寸
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        # 应用基础缩放
        x = self.base_scale(x)
        x = x + x_tag

        return x 