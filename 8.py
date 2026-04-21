import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体样式（放大字体，支持清晰显示）
plt.rcParams.update({
    'font.size': 12,          # 全局基础字体大小
    'axes.titlesize': 14,     # 图表标题字体大小
    'axes.labelsize': 12,     # 坐标轴标签字体大小
    'xtick.labelsize': 10,    # x轴刻度字体大小
    'ytick.labelsize': 10,    # y轴刻度字体大小
    'legend.fontsize': 11,    # 图例字体大小
    'figure.figsize': (14, 8) # 图表默认尺寸
})

def plot_table_data(methods, precision, recall, accuracy, f1, iou, title, save_path):
    """
    绘制单个表格的折线图
    :param methods: 方法名称列表
    :param precision: 精确率数据
    :param recall: 召回率数据
    :param accuracy: 准确率数据
    :param f1: F1分数数据
    :param iou: IoU数据
    :param title: 图表标题
    :param save_path: 图片保存路径
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 定义各指标的颜色和标记（统一风格，便于区分）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'd', '*']
    metrics = ['Precision', 'Recall', 'Accuracy', 'F1-Score', 'IoU']
    data = [precision, recall, accuracy, f1, iou]
    
    # 绘制各指标折线
    for i in range(len(metrics)):
        ax.plot(methods, data[i], 
                marker=markers[i], color=colors[i], 
                label=metrics[i], linewidth=2, markersize=6)
    
    # 定位Ours位置，加粗x轴标签并放大字号
    ours_idx = methods.index('Ours')
    xticks = ax.get_xticklabels()
    xticks[ours_idx].set_weight('bold')  # 加粗Ours标签
    xticks[ours_idx].set_fontsize(11)    # 放大Ours标签字号
    
    # 突出显示Ours的指标点（加粗边框+放大尺寸）
    ours_x = methods[ours_idx]
    for i in range(len(metrics)):
        ax.scatter(ours_x, data[i][ours_idx], 
                   color=colors[i], s=100, marker=markers[i],
                   edgecolor='black', linewidth=2, zorder=5)  # zorder确保点在最上层
    
    # 设置图表标题和坐标轴标签（加粗）
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    # ax.set_xlabel('Methods', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    
    # 旋转x轴标签，避免方法名称重叠
    plt.xticks(rotation=45, ha='right')
    
    # 添加图例（带阴影，增强可读性）
    ax.legend(loc='best', frameon=True, shadow=True, framealpha=0.9)
    
    # 添加网格线（虚线，半透明）
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局，避免标签被截断
    plt.tight_layout()
    
    # 保存图片（高分辨率，避免边框截断）
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ===================== 表格1数据：DeepCrack (Trained on Crack500) =====================
tab1_methods = ['DeepLabv3+', 'ENet', 'PSPNet', 'UperNet', 'SegResNet', 'UNet', 'BiSeNetv2', 
                'DDRNet', 'Lawin', 'Deepcrack19', 'Deepcrack18', 'SCADN', 'RIAD', 'UP-CrackNet', 'Ours']
tab1_precision = [41.07, 63.10, 31.92, 29.16, 50.28, 79.39, 76.20, 38.23, 54.99, 95.82, 90.34, 59.01, 86.54, 72.46, 69.92]
tab1_recall = [32.45, 37.84, 73.21, 66.69, 36.14, 22.93, 20.10, 64.59, 63.05, 48.46, 61.83, 16.97, 16.62, 79.53, 83.28]
tab1_accuracy = [93.71, 93.88, 96.52, 96.28, 93.97, 87.49, 85.79, 96.40, 96.64, 95.38, 97.15, 85.66, 80.52, 97.99, 97.73]
tab1_f1 = [36.25, 47.31, 44.46, 40.57, 42.05, 35.58, 31.81, 48.03, 58.74, 64.37, 73.42, 26.36, 27.88, 75.83, 76.02]
tab1_iou = [22.14, 30.98, 28.58, 25.45, 26.62, 21.64, 18.92, 31.61, 41.59, 47.46, 58.01, 15.18, 16.20, 61.07, 61.31]

# ===================== 表格2数据：Crack500 (Trained on DeepCrack) =====================
tab2_methods = ['DeepLabv3+', 'ENet', 'PSPNet', 'UperNet', 'SegResNet', 'UNet', 'BiSeNetv2', 
                'DDRNet', 'Lawin', 'Deepcrack19', 'Deepcrack18', 'SCADN', 'RIAD', 'UP-CrackNet', 'Ours']
tab2_precision = [33.99, 15.59, 8.28, 46.68, 8.72, 5.65, 20.45, 40.76, 20.27, 61.16, 21.19, 49.13, 67.50, 62.79, 45.65]
tab2_recall = [44.69, 54.30, 85.04, 42.17, 64.04, 47.10, 40.46, 16.88, 76.55, 51.05, 77.55, 14.89, 12.41, 52.20, 42.33]
tab2_accuracy = [93.95, 94.54, 94.79, 93.44, 94.62, 94.37, 93.87, 85.47, 95.19, 94.55, 95.25, 81.46, 71.55, 94.70, 93.51]
tab2_f1 = [38.61, 24.22, 15.09, 44.31, 15.34, 10.09, 27.16, 23.87, 32.05, 55.65, 33.29, 22.86, 20.96, 57.01, 43.93]
tab2_iou = [23.92, 13.78, 8.16, 28.46, 8.31, 5.31, 15.72, 13.55, 19.08, 38.55, 19.97, 12.90, 11.71, 39.87, 28.14]

# ===================== 表格3数据：CFD (Trained on Crack500) =====================
tab3_methods = ['DeepLabv3+', 'ENet', 'PSPNet', 'UperNet', 'SegResNet', 'UNet', 'BiSeNetv2', 
                'DDRNet', 'Lawin', 'Deepcrack19', 'Deepcrack18', 'SCADN', 'RIAD', 'UP-CrackNet', 'Ours']
tab3_precision = [6.73, 0.12, 11.36, 7.17, 0.08, 15.67, 18.85, 0.08, 38.50, 95.21, 77.50, 43.42, 93.61, 63.16, 34.40]
tab3_recall = [54.66, 0.20, 52.25, 55.64, 0.73, 48.02, 21.05, 31.99, 37.25, 29.44, 43.75, 15.34, 11.55, 41.25, 42.62]
tab3_accuracy = [98.27, 98.26, 98.27, 98.28, 98.05, 98.23, 97.35, 98.25, 97.79, 95.94, 97.87, 94.83, 87.39, 97.79, 97.76]
tab3_f1 = [11.99, 0.15, 18.66, 12.70, 0.15, 23.63, 19.89, 0.16, 37.86, 44.98, 55.93, 22.67, 20.56, 49.90, 38.07]
tab3_iou = [6.38, 0.07, 10.29, 6.78, 0.07, 13.40, 11.04, 0.08, 23.35, 29.01, 38.82, 12.78, 11.46, 33.25, 23.51]

# ===================== 表格4数据：CFD (Trained on DeepCrack) =====================
tab4_methods = ['DeepLabv3+', 'ENet', 'PSPNet', 'UperNet', 'SegResNet', 'UNet', 'BiSeNetv2', 
                'DDRNet', 'Lawin', 'Deepcrack19', 'Deepcrack18', 'SCADN', 'RIAD', 'UP-CrackNet', 'Ours']
tab4_precision = [37.29, 17.82, 22.72, 31.49, 16.10, 25.60, 15.70, 9.15, 14.90, 77.20, 35.84, 64.69, 92.26, 52.39, 61.54]
tab4_recall = [48.66, 50.52, 42.55, 51.81, 48.68, 53.28, 49.19, 46.62, 55.15, 38.40, 54.23, 8.63, 10.82, 42.26, 55.89]
tab4_accuracy = [98.22, 98.21, 98.11, 98.29, 98.24, 98.31, 98.24, 98.23, 98.30, 97.44, 98.35, 87.43, 86.61, 97.92, 98.72]
tab4_f1 = [42.23, 26.35, 29.63, 39.18, 24.20, 34.59, 23.81, 15.30, 23.47, 51.29, 43.16, 15.22, 19.37, 46.79, 58.58]
tab4_iou = [26.76, 15.17, 17.39, 24.36, 13.77, 20.91, 13.51, 8.28, 13.29, 34.49, 27.51, 8.24, 10.73, 30.54, 41.42]

# 绘制并保存四个表格的折线图
plot_table_data(tab1_methods, tab1_precision, tab1_recall, tab1_accuracy, tab1_f1, tab1_iou,
                'Generalization Results on DeepCrack (Trained on Crack500)',
                'tab1_deepcrack_crack500.png')

plot_table_data(tab2_methods, tab2_precision, tab2_recall, tab2_accuracy, tab2_f1, tab2_iou,
                'Generalization Results on Crack500 (Trained on DeepCrack)',
                'tab2_crack500_deepcrack.png')

plot_table_data(tab3_methods, tab3_precision, tab3_recall, tab3_accuracy, tab3_f1, tab3_iou,
                'Generalization Results on CFD (Trained on Crack500)',
                'tab3_cfd_crack500.png')

plot_table_data(tab4_methods, tab4_precision, tab4_recall, tab4_accuracy, tab4_f1, tab4_iou,
                'Generalization Results on CFD (Trained on DeepCrack)',
                'tab4_cfd_deepcrack.png')