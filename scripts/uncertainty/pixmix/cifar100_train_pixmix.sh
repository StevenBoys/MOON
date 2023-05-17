#!/bin/bash
# sh scripts/uncertainty/pixmix/cifar100_train_pixmix.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \
# -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/train/baseline.yml \
configs/preprocessors/pixmix_preprocessor.yml \
--num_workers 8 \
--optimizer.num_epochs 100 \
--mark pixmix
