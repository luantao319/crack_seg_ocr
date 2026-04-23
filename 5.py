"""
普通图像 vs 路面破损图像
RGB 像素分布差异（单张 3D 直方图，含图例）
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------ 1. 路径与参数 ------------------
generic_dir = r'C:\Users\H3C\Desktop\coco\coco500'
pavement_dir = r'C:\Users\H3C\Desktop\coco\crack500'
save_root = r'C:\Users\H3C\Desktop'
os.makedirs(save_root, exist_ok=True)

sample_size = 100
image_formats = ['.jpg', '.png', '.jpeg']
np.random.seed(42)

gen_color, pav_color = '#2E86AB', '#E63946'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.unicode_minus'] = False

# ------------------ 2. 工具函数 ------------------
def load_rgb_pixels(dir_path, max_count=sample_size):
    pixels = []
    files = [f for f in os.listdir(dir_path)
             if any(f.lower().endswith(fmt) for fmt in image_formats)]
    if len(files) > max_count:
        files = np.random.choice(files, max_count, replace=False)
    for fname in files:
        img = cv2.imread(os.path.join(dir_path, fname))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels.append(img.reshape(-1, 3))
    pixels = np.vstack(pixels)
    return pixels[:, 0], pixels[:, 1], pixels[:, 2]  # R, G, B

# ------------------ 3. 读取像素 ------------------
print('📊 Loading RGB pixels...')
generic_r, generic_g, generic_b = load_rgb_pixels(generic_dir)
pavement_r, pavement_g, pavement_b = load_rgb_pixels(pavement_dir)

# ------------------ 4. 3D 直方图 ------------------
fig = plt.figure(figsize=(18, 14))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=25, azim=45)

bins = 50
x = np.arange(bins)
width = 0.8
offset = 2

# 绘制柱状图
for label, r, g, b, color in [('Generic', generic_r, generic_g, generic_b, gen_color),
                              ('Pavement', pavement_r, pavement_g, pavement_b, pav_color)]:
    hr, _ = np.histogram(r, bins=bins, range=(0, 255), density=True)
    hg, _ = np.histogram(g, bins=bins, range=(0, 255), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(0, 255), density=True)

    ax.bar(x - width, hr, width=width, zs=offset,  zdir='y', color=color, alpha=0.7, label=label)
    ax.bar(x,         hg, width=width, zs=0,       zdir='y', color=color, alpha=0.5)
    ax.bar(x + width, hb, width=width, zs=-offset, zdir='y', color=color, alpha=0.3)

# ========== 关键修改：增加labelpad，拉开文字与坐标轴距离 ==========
ax.set_xlabel('Pixel Value', fontsize=20, fontweight='bold', labelpad=25)
ax.set_ylabel('Channel',     fontsize=20, fontweight='bold', labelpad=25)
ax.set_zlabel('Density',     fontsize=20, fontweight='bold', labelpad=30)

ax.set_yticks([offset, 0, -offset])
ax.set_yticklabels(['R', 'G', 'B'], fontsize=18)

# 标题增加上间距
ax.set_title('RGB Distribution: Generic vs Pavement (3D)', 
             fontsize=24, fontweight='bold', pad=35)

# 刻度文字 + 刻度与轴线间距加大
ax.tick_params(axis='x', labelsize=18, pad=12)
ax.tick_params(axis='y', labelsize=18, pad=12)
ax.tick_params(axis='z', labelsize=18, pad=15)

ax.legend(loc='upper left', fontsize=18)

out_path = os.path.join('RGB_3D_Histogram_Only.png')
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'✅ RGB 对比图已保存: {out_path}')
plt.show()