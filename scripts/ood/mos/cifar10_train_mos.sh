#!/bin/bash
# sh scripts/ood/mos/cifar10_train_mos.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/cifar10/cifar10_double_label.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/train/train_mos.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/mos.yml \
--num_workers 8 \
--network.pretrained False \
--merge_option merge
