import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_gabor_kernel(kernel_size, sigma, theta, lambd, gamma, psi=0):
    """
    创建Gabor滤波器核
    :param kernel_size: 滤波器尺寸 (奇数)
    :param sigma: 高斯标准差
    :param theta: 滤波器方向 (弧度)
    :param lambd: 正弦波波长
    :param gamma: 空间纵横比
    :param psi: 相位偏移 (0或np.pi/2)
    :return: Gabor核
    """
    sigma_x = sigma
    sigma_y = sigma / gamma
    
    # 创建网格
    half_size = kernel_size // 2
    x, y = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]
    
    # 旋转坐标
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    
    # Gabor函数公式
    gabor = np.exp(-(x_rot**2 + sigma_y**2 * y_rot**2) / (2 * sigma_x**2))
    gabor *= np.cos(2 * np.pi * x_rot / lambd + psi)
    
    # 归一化（均值为0，方差为1）
    gabor = (gabor - np.mean(gabor)) / np.std(gabor)
    return gabor

def apply_gabor_filters(image, kernel_params_list):
    """
    应用多个Gabor滤波器到图像
    :param image: 输入灰度图
    :param kernel_params_list: Gabor核参数列表
    :return: 滤波结果列表
    """
    results = []
    for params in kernel_params_list:
        kernel = create_gabor_kernel(**params)
        # 卷积操作（使用CV_64F避免溢出，再归一化到0-255）
        filtered = cv2.filter2D(image, cv2.CV_64F, kernel)
        # 归一化到0-255
        filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min()) * 255
        filtered = filtered.astype(np.uint8)
        results.append(filtered)
    return results

# 主程序
if __name__ == "__main__":
    # 1. 读取输入图像（转为灰度图）
    input_path = "1.jpg"
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"无法找到图像文件: {input_path}")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 定义Gabor滤波器参数（多方向、多尺度）
    kernel_size = 31  # 滤波器尺寸（奇数）
    sigmas = [4, 8]   # 不同尺度（高斯标准差）
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 4个方向（0°,45°,90°,135°）
    lambd = 10        # 波长
    gamma = 0.5       # 纵横比
    psi = 0           # 相位偏移
    
    # 生成参数列表
    kernel_params = []
    for sigma in sigmas:
        for theta in thetas:
            kernel_params.append({
                "kernel_size": kernel_size,
                "sigma": sigma,
                "theta": theta,
                "lambd": lambd,
                "gamma": gamma,
                "psi": psi
            })
    
    # 3. 应用Gabor滤波
    filtered_images = apply_gabor_filters(gray_image, kernel_params)
    
    # 4. 可视化结果
    plt.figure(figsize=(16, 10))
    
    # 显示原始图像
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("原始图像", fontsize=12)
    plt.axis("off")
    
    # 显示灰度图像
    plt.subplot(3, 4, 2)
    plt.imshow(gray_image, cmap="gray")
    plt.title("灰度图像", fontsize=12)
    plt.axis("off")
    
    # 显示所有滤波结果
    for i, (filtered_img, params) in enumerate(zip(filtered_images, kernel_params), start=3):
        theta_deg = params["theta"] * 180 / np.pi
        sigma = params["sigma"]
        plt.subplot(3, 4, i)
        plt.imshow(filtered_img, cmap="gray")
        plt.title(f"σ={sigma}, θ={theta_deg:.0f}°", fontsize=10)
        plt.axis("off")
    
    # 调整布局
    plt.tight_layout()
    
    # 5. 保存结果为1.png
    output_path = "1.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Gabor滤波可视化结果已保存到: {output_path}")