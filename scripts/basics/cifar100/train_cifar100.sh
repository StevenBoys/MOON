#!/bin/bash
# sh scripts/basics/cifar100/train_cifar100.sh

GPU=1
CPU=1
#node=73
#jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} \
#-w SG-IDC1-10-51-2-${node} \

CUDA_VISIBLE_DEVICES=2 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.001 --w_out2 0.001 --w_name "1en3" > cifar100_ex_wa_1en3.txt &

CUDA_VISIBLE_DEVICES=2 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.01 --w_name "1en2" > cifar100_ex_wa_1en2.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.1 --w_name "1en1" > cifar100_ex_wa_1en1.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 1.0 --w_out2 1.0 --w_name "1en0" > cifar100_ex_wa_1en0.txt &

##### diff w1 and w2

CUDA_VISIBLE_DEVICES=2 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.0001 --w_out2 0.000001 --w_name "1en4_6" > cifar100_ex_wa_1en4_6.txt &

CUDA_VISIBLE_DEVICES=2 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.0001 --w_out2 0.00001 --w_name "1en4_5" > cifar100_ex_wa_1en4_5.txt &

CUDA_VISIBLE_DEVICES=5 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.0001 --w_out2 0.001 --w_name "1en4_3" > cifar100_ex_wa_1en4_3.txt &

CUDA_VISIBLE_DEVICES=5 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.0001 --w_out2 0.01 --w_name "1en4_2" > cifar100_ex_wa_1en4_2.txt &


##### diff increase w1 and w2

CUDA_VISIBLE_DEVICES=2 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.001 --w_name "1en2_inc_3" > cifar100_ex_wa_1en2_inc_3.txt &

CUDA_VISIBLE_DEVICES=2 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.001 --w_out2 0.001 --w_name "1en3_inc_3" > cifar100_ex_wa_1en3_inc_3.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.0001 --w_out2 0.001 --w_name "1en4_inc_3" > cifar100_ex_wa_1en4_inc_3.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.00001 --w_out2 0.001 --w_name "1en5_inc_3" > cifar100_ex_wa_1en5_inc_3.txt &


##### diff decrease w1 and w2

CUDA_VISIBLE_DEVICES=2 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.001 --w_name "1en2_dec_3" > cifar100_ex_wa_1en2_dec_3.txt &

CUDA_VISIBLE_DEVICES=2 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.001 --w_out2 0.001 --w_name "1en3_dec_3" > cifar100_ex_wa_1en3_dec_3.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.0001 --w_out2 0.001 --w_name "1en4_dec_3" > cifar100_ex_wa_1en4_dec_3.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.00001 --w_out2 0.001 --w_name "1en5_dec_3" > cifar100_ex_wa_1en5_dec_3.txt &

##### diff decrease w1 and w2 tuning seed

CUDA_VISIBLE_DEVICES=2 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.001 --w_name "1en2_dec_3_s7" --seed 7 > cifar100_ex_wa_1en2_dec_3_s7.txt &

CUDA_VISIBLE_DEVICES=2 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.001 --w_out2 0.001 --w_name "1en3_dec_3_s7" --seed 7 > cifar100_ex_wa_1en3_dec_3_s7.txt &


##### diff decrease w1 and w2 tuning lr (2, 0.2)

CUDA_VISIBLE_DEVICES=1 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.001 --w_name "1en2_dec_3_s7" --seed 5 > cifar100_ex_wa_1en2_dec_3_lr_2_02_s5.txt &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.001 --w_out2 0.001 --w_name "1en3_dec_3_s7" --seed 5 > cifar100_ex_wa_1en3_dec_3_lr_2_02_s5.txt &


CUDA_VISIBLE_DEVICES=1 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.001 --w_name "1en2_dec_3_s5" --seed 5 > cifar100_ex_wa_1en2_dec_3_lr_05_005_s5.txt &

CUDA_VISIBLE_DEVICES=4 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.001 --w_out2 0.001 --w_name "1en3_dec_3_s5" --seed 5 > cifar100_ex_wa_1en3_dec_3_lr_05_005_s5.txt &


##### weaker ensemble (turning acc)

CUDA_VISIBLE_DEVICES=2 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_08_002" --seed 5 --lr_ini 0.8 --lr_end 0.02 > cifar100_ex_wa_1en2_dec_3_lr_08_002_s5.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_12_002" --seed 5 --lr_ini 1.2 --lr_end 0.02 > cifar100_ex_wa_1en3_dec_3_lr_12_002_s5.txt &


CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_08_0002" --seed 5 --lr_ini 0.8 --lr_end 0.002 > cifar100_ex_wa_1en2_dec_3_lr_08_0002_s5.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_12_0002" --seed 5 --lr_ini 1.2 --lr_end 0.002 > cifar100_ex_wa_1en3_dec_3_lr_12_0002_s5.txt &


##### weaker ensemble (cifar10)

CUDA_VISIBLE_DEVICES=2 nohup python main.py --config configs/datasets/cifar10/cifar10.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_08_002" --seed 5 --lr_ini 0.8 --lr_end 0.02 > cifar10_ex_wa_1en2_dec_3_lr_08_002_s5.txt &

CUDA_VISIBLE_DEVICES=2 nohup python main.py --config configs/datasets/cifar10/cifar10.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_12_002" --seed 5 --lr_ini 1.2 --lr_end 0.02 > cifar10_ex_wa_1en2_dec_3_lr_12_002_s5.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar10/cifar10.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_2_002" --seed 5 --lr_ini 2 --lr_end 0.02 > cifar10_ex_wa_1en2_3_lr_2_002_s5.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar10/cifar10.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_15_002" --seed 5 --lr_ini 1.5 --lr_end 0.02 > cifar10_ex_wa_1en2_3_lr_15_002_s5.txt &

#CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar10/cifar10.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_18_002" --seed 5 --lr_ini 1.8 --lr_end 0.02 > cifar10_ex_wa_1en2_3_lr_18_002_s5.txt &

##### weaker ensemble (turning acc)

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_15_002" --seed 5 --lr_ini 1.5 --lr_end 0.02 > cifar100_ex_wa_1en2_3_lr_15_002_s5.txt &

CUDA_VISIBLE_DEVICES=2 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.01 --w_name "1en1_2_s5_lr_15_002" --seed 5 --lr_ini 1.5 --lr_end 0.02 > cifar100_ex_wa_1en1_2_lr_15_002_s5.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_20_002" --seed 5 --lr_ini 2 --lr_end 0.02 > cifar100_ex_wa_1en2_3_lr_20_002_s5.txt &

##### weaker ensemble (turning extra class)

CUDA_VISIBLE_DEVICES=4 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.01 --w_name "ex_10_1en1_2_s5_lr_15_002" --seed 5 --lr_ini 1.5 --lr_end 0.02 > cifar100_ex_10_wa_1en1_2_lr_15_002_s5.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.01 --w_name "e_250_50_1en1_2_s5_lr_15_002" --seed 5 --lr_ini 1.5 --lr_end 0.02 --optimizer.num_epochs 250 --optimizer.deduct 50 > cifar100_ex_e_250_50_wa_1en1_2_lr_15_002_s5.txt &


# more average

# c10
CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar10/cifar10.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.001 --w_name "more_1en2_3_s5_lr_15_002" --seed 5 --lr_ini 1.5 --lr_end 0.02 > cifar10_ex_wa_more_1en2_3_lr_15_002_s5.txt &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --config configs/datasets/cifar10/cifar10.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.01 --w_name "more_1en2_2_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 > cifar10_ex_wa_more_1en2_2_lr_15_01_s5.txt & # good

CUDA_VISIBLE_DEVICES=0 nohup python main.py --config configs/datasets/cifar10/cifar10.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.001 --w_out2 0.01 --w_name "more_1en2_3_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 > cifar10_ex_wa_more_1en3_2_lr_15_01_s5.txt & # good

# c100
CUDA_VISIBLE_DEVICES=4 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.01 --w_name "more_1en1_2_s5_lr_15_002" --seed 5 --lr_ini 1.5 --lr_end 0.02 > cifar100_ex_wa_more_1en1_2_lr_15_002_s5.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.01 --w_out2 0.01 --w_name "more_1en2_2_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 > cifar100_ex_wa_more_1en2_2_lr_15_01_s5.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.001 --w_out2 0.01 --w_name "more_1en3_2_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 > cifar100_ex_wa_more_1en3_2_lr_15_01_s5.txt &

## more c100

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.01 --w_name "more_1en1_2_s5_lr_01_001" --seed 5 --lr_ini 0.1 --lr_end 0.01 > cifar100_ex_wa_more_1en1_2_lr_01_001_s5.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 1.0 --w_out2 1.0 --w_name "more_1en0_0_s5_lr_01_001" --seed 5 --lr_ini 0.1 --lr_end 0.01 > cifar100_ex_wa_more_1en0_0_lr_01_001_s5.txt &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.01 --w_name "more_1en1_2_s5_lr_3_005" --seed 5 --lr_ini 3 --lr_end 0.05 > cifar100_ex_wa_more_1en1_2_lr_3_005_s5.txt &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 1.0 --w_out2 1.0 --w_name "more_1en0_0_s5_lr_3_005" --seed 5 --lr_ini 3 --lr_end 0.05 > cifar100_ex_wa_more_1en0_0_lr_3_005_s5.txt &

## more c100

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.01 --w_name "more_1en1_2_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 > cifar100_ex_wa_more_1en1_2_lr_15_01_s5.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 1.0 --w_name "more_1en1_0_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 > cifar100_ex_wa_more_1en1_0_lr_15_01_s5.txt &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.01 --w_name "more_1en1_2_s5_lr_15_0002" --seed 5 --lr_ini 1.5 --lr_end 0.002 > cifar100_ex_wa_more_1en1_2_lr_15_0002_s5.txt &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 1.0 --w_name "more_1en1_0_s5_lr_15_0002" --seed 5 --lr_ini 1.5 --lr_end 0.002 > cifar100_ex_wa_more_1en1_0_lr_15_0002_s5.txt &

## more c100

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.1 --w_name "more_1en1_1_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 > cifar100_ex_wa_more_1en1_2_lr_15_01_s5.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.1 --w_name "more_1en1_1_s5_lr_15_001" --seed 5 --lr_ini 1.5 --lr_end 0.01 > cifar100_ex_wa_more_1en1_0_lr_15_001_s5.txt &

## more c100

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.1 --w_name "more_10_1en1_1_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 > cifar100_ex_wa_more_10_1en1_2_lr_15_01_s5.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.1 --w_name "more_10_1en1_1_s5_lr_15_005" --seed 5 --lr_ini 1.5 --lr_end 0.05 > cifar100_ex_wa_more_10_1en1_2_lr_15_005_s5.txt &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.1 --w_name "more_10_1en1_1_s5_lr_15_0025" --seed 5 --lr_ini 1.5 --lr_end 0.025 > cifar100_ex_wa_more_10_1en1_2_lr_15_0025_s5.txt &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.1 --w_name "more_10_1en1_1_s5_lr_15_02" --seed 5 --lr_ini 1.5 --lr_end 0.2 > cifar100_ex_wa_more_10_1en1_2_lr_15_02_s5.txt &


# c100 ori
CUDA_VISIBLE_DEVICES=4 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --seed 5 --optimizer.num_epochs 250 > cifar100_e_250_ori.txt &


## more c100

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.1 --w_name "c_102_more_10_1en1_1_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 --classes 102 > cifar100_ex_wa_c_102_more_10_1en1_2_lr_15_01_s5.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.1 --w_name "c_103_more_10_1en1_1_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 --classes 103 > cifar100_ex_wa_c_103_more_10_1en1_2_lr_15_01_s5.txt &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.1 --w_name "c_104_more_10_1en1_1_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 --classes 104 > cifar100_ex_wa_c_104_more_10_1en1_2_lr_15_01_s5.txt &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.1 --w_name "c_105_more_10_1en1_1_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 --classes 105 > cifar100_ex_wa_c_105_more_10_1en1_2_lr_15_01_s5.txt &

## more c100

CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml --w_out 0.1 --w_out2 0.1 --w_name "c_101_more_10_1en1_1_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 --classes 101 > cifar100_ex_wa_c_101_more_10_1en1_2_lr_15_01_s5.txt &