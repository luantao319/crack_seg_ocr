#!/bin/bash

# 裂缝分割训练脚本 - OCR版本

USE_WAVELET="${1:-true}"        # 是否使用小波增强
USE_LOGGABOR="${2:-true}"    # 是否使用LogGabor增强
USE_FUSION="${3:-true}"     # 是否使用融合分支（WT低频+Gabor高频）
CUDA_VISIBLE_DEVICES="${4:-6}"
pretrained="${5:-''}" 
DATA_PATH="${6:-/home/peipengfei/gy/CRACK/crack_seg_ocr/dataset/deepcrack2/}"


# 设置环境变量
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH=$PYTHONPATH:/home/peipengfei/gy/CRACK/crack_seg_ocr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# 训练参数
BASE_OUTPUT_DIR="/home/peipengfei/gy/CRACK/crack_seg_ocr/outputs"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${DATA_LAST_DIR}/"
BATCH_SIZE=32
NUM_EPOCHS=50
LEARNING_RATE=0.0005
DEVICE="cuda"
NUM_WORKERS=2
SAVE_INTERVAL=5
VISUALIZE_INTERVAL=1
OCR_MID_CHANNELS=512
OCR_KEY_CHANNELS=256


# 阈值方法选择
USE_OTSU=true      # 是否使用Otsu自适应阈值 (true: Otsu, false: 固定阈值)
FIXED_THRESHOLD=0.25   # 固定阈值值（当USE_OTSU=false时使用）

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 开始训练
echo "开始训练裂缝分割模型..."
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "批次大小: $BATCH_SIZE"
echo "训练轮数: $NUM_EPOCHS"
echo "学习率: $LEARNING_RATE"
echo "设备: $DEVICE"
echo "小波增强: $USE_WAVELET"
echo "LogGabor增强: $USE_LOGGABOR"
echo "融合分支: $USE_FUSION"

# 显示阈值方法信息
if [ "$USE_OTSU" = true ]; then
    echo "阈值方法: Otsu自适应阈值"
else
    echo "阈值方法: 固定阈值($FIXED_THRESHOLD)"
fi

# 构建训练命令
TRAIN_CMD="python train.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE \
    --num_workers $NUM_WORKERS \
    --save_interval $SAVE_INTERVAL \
    --visualize_interval $VISUALIZE_INTERVAL \
    --ocr_mid_channels $OCR_MID_CHANNELS \
    --pretrained $pretrained \
    --ocr_key_channels $OCR_KEY_CHANNELS"

# 根据启用的增强类型添加参数
if [ "$USE_WAVELET" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use_wavelet"
fi

if [ "$USE_LOGGABOR" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use_loggabor"
fi

if [ "$USE_FUSION" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use_fusion"
fi

# 根据阈值方法选择添加参数
if [ "$USE_OTSU" = false ]; then
    TRAIN_CMD="$TRAIN_CMD --no-otsu --fixed_threshold $FIXED_THRESHOLD"
fi

# 显示最终训练命令
echo "执行训练命令:"
echo "$TRAIN_CMD"

# 运行训练
eval $TRAIN_CMD

echo "训练完成！"