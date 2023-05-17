# python scripts/uncertainty/mixup/sweep.py
import os

config = [
    ['osr_cifar6/cifar6_seed1.yml', 'resnet18_32x32'],
    ['osr_cifar50/cifar50_seed1.yml', 'resnet18_32x32'],
    ['osr_tin20/tin20_seed1.yml', 'resnet18_64x64'],
    ['osr_mnist4/mnist4_seed1.yml', 'lenet'],
    ['mnist/mnist.yml', 'lenet'],
]

for [dataset, network] in config:
    command = (f"PYTHONPATH='.':$PYTHONPATH \
    srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 \
    --cpus-per-task=1 --ntasks-per-node=1 \
    --kill-on-bad-exit=1 --job-name=openood \
    -w SG-IDC1-10-51-2-75 \
    python main.py \
    --config configs/datasets/{dataset} \
    configs/networks/{network}.yml \
    configs/pipelines/train/train_mixup.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained False \
    --optimizer.num_epochs 100 \
    --num_workers 8 &")
    os.system(command)
