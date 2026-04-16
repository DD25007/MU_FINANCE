#!/bin/bash

DATASETS=("german" "gmsc")
ARCHITECTURES=("ft_transformer" "tab_transformer" "tabddpm")
FORGET_STRATEGIES=("demographic" "random")
MODES=("full")
GPU_ID=1

for DATASET_NAME in "${DATASETS[@]}"; do
    for ARCH in "${ARCHITECTURES[@]}"; do
        for FORGET_STRATEGY in "${FORGET_STRATEGIES[@]}"; do
            for MODE in "${MODES[@]}"; do
                echo "Running: dataset=$DATASET_NAME arch=$ARCH forget_strategy=$FORGET_STRATEGY mode=$MODE"
                CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                    --dataset "$DATASET_NAME" \
                    --arch "$ARCH" \
                    --forget_strategy "$FORGET_STRATEGY" \
                    --mode "$MODE" > "kaustav_${DATASET_NAME}_${ARCH}_${FORGET_STRATEGY}_${MODE}.log" 2>&1 &
            done
        done
    done
done

wait