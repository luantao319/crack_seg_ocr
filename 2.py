import os, glob, cv2, numpy as np
import seaborn as sns, matplotlib.pyplot as plt
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from pywt import dwt2

# 1. 路径：普通图放 generic/，路面图放 pavement/
generic_dir = r'C:\Users\H3C\Desktop\coco\coco500'
pavement_dir = r'C:\Users\H3C\Desktop\coco\crack500'
images = {'Generic': glob.glob(generic_dir+'/*.jpg'),
          'Pavement': glob.glob(pavement_dir+'/*.jpg')}

# 2. 特征提取函数
def extract(img_rgb):
    g = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    # GLCM 特征 (更新了函数名)
    glcm = graycomatrix(g, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]  # ASM 已改为 energy
    homo = graycoprops(glcm, 'homogeneity')[0, 0]
    
    # LBP 熵
    lbp = local_binary_pattern(g, P=8, R=1, method='nri_uniform')
    hist, _ = np.histogram(lbp, bins=59)
    hist = hist.astype(float)
    hist /= hist.sum()  # 归一化
    lbp_ent = -np.sum(hist * np.log2(hist + 1e-12))
    
    # Gabor 均值
    freq = 0.2
    gabor_real, _ = gabor(g, frequency=freq)
    gabor_mean = np.abs(gabor_real).mean()  # 取绝对值
    
    # 小波 HH 能量
    cA, (cH, cV, cD) = dwt2(g, 'haar')
    wavelet_energy = np.sqrt(cD**2).mean()
    
    return [contrast, correlation, energy, homo, lbp_ent, gabor_mean, wavelet_energy]

# 3. 批量计算
feat_df = []
for label, path_list in images.items():
    for p in path_list:
        try:
            feat = extract(cv2.imread(p))
            feat_df.append([label] + feat)
        except Exception as e:
            print(f"处理图片 {p} 时出错: {e}")
            pass

df = pd.DataFrame(feat_df, columns=['Group', 'Contrast', 'Correlation', 'Energy', 'Homogeneity',
                                   'LBP_Entropy', 'Gabor_Mean', 'Wavelet_Energy'])

# 4. 保存结果到CSV
df.to_csv('texture_features.csv', index=False)

# 5. 画图
plt.rcParams['font.size'] = 11
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
sns.boxplot(data=df, x='', y='Contrast', ax=ax[0, 0])
sns.violinplot(data=df, x='', y='LBP_Entropy', ax=ax[0, 1])
sns.kdeplot(data=df, x='Gabor_Mean', hue='', fill=True, ax=ax[1, 0])  # shade 改为 fill
sns.kdeplot(data=df, x='Wavelet_Energy', hue='', fill=True, ax=ax[1, 1])  # shade 改为 fill
plt.tight_layout()
plt.savefig('texture_compare.png', dpi=300)
plt.show()

# 6. 打印统计信息
print("特征统计信息:")
print(df.groupby('Group').mean())