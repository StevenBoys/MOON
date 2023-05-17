#!/bin/bash
# sh scripts/ood/msp/cifar10_test_fsood_msp.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_fsood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_fsood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
--num_workers 8 \
--network.checkpoint 'results/checkpoints/cifar10_res18_acc95.24.ckpt' \
--mark 0
