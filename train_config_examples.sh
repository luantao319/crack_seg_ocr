#!/bin/bash

# 裂缝分割训练配置示例
# 展示如何使用不同的阈值方法

echo "=== 裂缝分割训练配置示例 ==="
echo ""

# 示例1: 使用Otsu自适应阈值（追求最佳分割质量）
echo "示例1: Otsu自适应阈值配置"
echo "适用场景: 追求最佳分割质量，可以接受轻微的不一致性"
echo "配置: USE_OTSU=true"
echo "命令: ./train.sh"
echo ""

# 示例2: 使用固定阈值0.0（追求完全可重现）
echo "示例2: 固定阈值配置"
echo "适用场景: 需要完全可重现的结果，用于科学实验对比"
echo "配置: USE_OTSU=false"
echo "命令: USE_OTSU=false ./train.sh"
echo ""

# 示例3: 快速切换配置
echo "示例3: 快速切换配置"
echo "临时使用固定阈值:"
echo "  USE_OTSU=false ./train.sh"
echo ""
echo "临时使用Otsu阈值:"
echo "  USE_OTSU=true ./train.sh"
echo ""

# 示例4: 批量实验配置
echo "示例4: 批量实验配置"
echo "对比两种阈值方法的效果:"
echo ""
echo "# 第一次运行：Otsu方法"
echo "USE_OTSU=true ./train.sh"
echo ""
echo "# 第二次运行：固定阈值方法"
echo "USE_OTSU=false ./train.sh"
echo ""

# 示例5: 环境变量配置
echo "示例5: 环境变量配置"
echo "可以在运行前设置环境变量:"
echo "  export USE_OTSU=false"
echo "  ./train.sh"
echo ""

echo "=== 配置说明 ==="
echo "USE_OTSU=true   : 使用Otsu自适应阈值（默认）"
echo "USE_OTSU=false  : 使用固定阈值0.0"
echo ""
echo "注意: 固定阈值方法确保每次运行结果完全一致！" 