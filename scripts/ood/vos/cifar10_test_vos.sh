#!/bin/bash
# sh scripts/ood/vos/cifar10_test_vos.sh

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/ebo.yml \
--num_workers 8 \
--network.pretrained True \
--network.checkpoint 'results/cifar10_resnet18_32x32_vos/best_epoch92_acc0.9550.ckpt' \
--mark vos
