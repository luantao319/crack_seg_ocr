#!/bin/bash

# 固定阈值配置示例
# 展示如何使用不同的固定阈值值

echo "=== 固定阈值配置示例 ==="
echo ""

# 示例1: 使用固定阈值0.0（默认）
echo "示例1: 固定阈值 0.0"
echo "适用场景: 所有概率值 > 0 的像素都被认为是前景"
echo "配置: USE_OTSU=false, FIXED_THRESHOLD=0.0"
echo "命令: USE_OTSU=false FIXED_THRESHOLD=0.0 ./train.sh"
echo ""

# 示例2: 使用固定阈值0.1
echo "示例2: 固定阈值 0.1"
echo "适用场景: 更保守的分割，只有概率值 > 0.1 的像素被认为是前景"
echo "配置: USE_OTSU=false, FIXED_THRESHOLD=0.1"
echo "命令: USE_OTSU=false FIXED_THRESHOLD=0.1 ./train.sh"
echo ""

# 示例3: 使用固定阈值-0.1
echo "示例3: 固定阈值 -0.1"
echo "适用场景: 更宽松的分割，概率值 > -0.1 的像素都被认为是前景"
echo "配置: USE_OTSU=false, FIXED_THRESHOLD=-0.1"
echo "命令: USE_OTSU=false FIXED_THRESHOLD=-0.1 ./train.sh"
echo ""

# 示例4: 使用固定阈值0.5
echo "示例4: 固定阈值 0.5"
echo "适用场景: 非常保守的分割，只有概率值 > 0.5 的像素被认为是前景"
echo "配置: USE_OTSU=false, FIXED_THRESHOLD=0.5"
echo "命令: USE_OTSU=false FIXED_THRESHOLD=0.5 ./train.sh"
echo ""

# 示例5: 批量实验不同阈值
echo "示例5: 批量实验不同阈值"
echo "对比不同固定阈值的效果:"
echo ""
echo "# 阈值 0.0"
echo "USE_OTSU=false FIXED_THRESHOLD=0.0 ./train.sh"
echo ""
echo "# 阈值 0.1"
echo "USE_OTSU=false FIXED_THRESHOLD=0.1 ./train.sh"
echo ""
echo "# 阈值 0.2"
echo "USE_OTSU=false FIXED_THRESHOLD=0.2 ./train.sh"
echo ""

echo "=== 阈值选择建议 ==="
echo ""
echo "阈值值范围: [-1.0, 1.0] (因为使用tanh激活函数)"
echo ""
echo "推荐阈值:"
echo "  -1.0 到 -0.5: 非常宽松，几乎所有像素都是前景"
echo "  -0.5 到 0.0:  宽松，大部分像素是前景"
echo "  0.0:          默认，平衡的选择"
echo "  0.0 到 0.5:  保守，只有高置信度像素是前景"
echo "  0.5 到 1.0:  非常保守，很少像素是前景"
echo ""
echo "注意: 固定阈值方法确保每次运行结果完全一致！" 