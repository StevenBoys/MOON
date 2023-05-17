#!/bin/bash
# sh scripts/ood/conf_branch/cifar100_train_conf_branch.sh

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/conf_branch.yml \
configs/pipelines/train/train_conf_branch.yml \
configs/preprocessors/base_preprocessor.yml \
--optimizer.num_epochs 100 \
--network.backbone.name resnet18_32x32 \
--network.backbone.checkpoint 'results/_cifar10_resnet18_32x32_base_e100_lr0.1/best_epoch89_acc0.9430.ckpt' \
--network.pretrained False