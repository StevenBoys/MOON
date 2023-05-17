#!/bin/bash
# sh scripts/basics/osr_cifar6/train_cifar6.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
python main.py \
--config configs/datasets/osr_cifar6/cifar6_seed5.yml \
configs/preprocessors/base_preprocessor.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/train/baseline.yml &
