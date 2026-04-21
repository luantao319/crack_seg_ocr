import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- 1. 路径与参数 ----------
generic_dir = r'C:\Users\H3C\Desktop\coco\coco500'      # 普通图像
pavement_dir = r'C:\Users\H3C\Desktop\coco\crack500'    # 路面图像
save_root = r'C:\Users\H3C\Desktop'                     # 输出文件夹
os.makedirs(save_root, exist_ok=True)

image_formats = ('.jpg', '.png', '.jpeg')
sample_size = 100
random_seed = 42
np.random.seed(random_seed)

# ---------- 2. 工具函数 ----------
def list_images(folder):
    """返回文件夹下所有符合格式的图片路径"""
    return [str(p) for p in Path(folder).rglob('*') 
            if p.suffix.lower() in image_formats]

def sample_images(folder, k):
    """随机采样 k 张图像"""
    imgs = list_images(folder)
    if len(imgs) < k:
        raise ValueError(f'{folder} 下只有 {len(imgs)} 张图，不足 {k} 张')
    return np.random.choice(imgs, k, replace=False)

def accumulate_rgb(paths):
    """遍历路径列表，累积所有像素的 RGB 值"""
    r, g, b = [], [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r.extend(img[..., 0].flatten())
        g.extend(img[..., 1].flatten())
        b.extend(img[..., 2].flatten())
    return np.array(r), np.array(g), np.array(b)

# ---------- 3. 采样并累积像素 ----------
print('Sampling generic images...')
generic_paths = sample_images(generic_dir, sample_size)
R_gen, G_gen, B_gen = accumulate_rgb(generic_paths)

print('Sampling pavement images...')
pavement_paths = sample_images(pavement_dir, sample_size)
R_pav, G_pav, B_pav = accumulate_rgb(pavement_paths)

# ---------- 4. 计算直方图 ----------
bins = np.linspace(0, 255, 256)
hist_R_gen, _ = np.histogram(R_gen, bins=bins, density=True)
hist_G_gen, _ = np.histogram(G_gen, bins=bins, density=True)
hist_B_gen, _ = np.histogram(B_gen, bins=bins, density=True)

hist_R_pav, _ = np.histogram(R_pav, bins=bins, density=True)
hist_G_pav, _ = np.histogram(G_pav, bins=bins, density=True)
hist_B_pav, _ = np.histogram(B_pav, bins=bins, density=True)

# ---------- 5. 可视化差异 ----------
x = bins[:-1]
plt.figure(figsize=(15, 5))

for idx, (ch, gen, pav, color) in enumerate(
        zip(('R', 'G', 'B'),
            (hist_R_gen, hist_G_gen, hist_B_gen),
            (hist_R_pav, hist_G_pav, hist_B_pav),
            ('red', 'green', 'blue'))):
    plt.subplot(1, 3, idx+1)
    plt.plot(x, gen, color=color, linestyle='--', label='Generic')
    plt.plot(x, pav, color=color, linestyle='-', label='Pavement')
    plt.title(f'{ch} Channel PDF')
    plt.xlabel('Pixel value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(save_root, 'rgb_pdf_comparison.png')
plt.savefig(out_path, dpi=300)
print('Saved ->', out_path)
plt.show()