#!/bin/bash
# sh scripts/ood/msp/cifar100_test_ood_msp_swa.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en4/model_swa.ckpt' \
--w_out 0.0001 --w_out2 0.0001 --w_name "1en4" \
--mark 0 > cifar100_res18_test_ex_wa_1en4.txt &

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en5/model_swa.ckpt' \
--w_out 0.00001 --w_out2 0.00001 --w_name "1en5" \
--mark 0 > cifar100_res18_test_ex_wa_1en5.txt &

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en6/model_swa.ckpt' \
--w_out 0.000001 --w_out2 0.000001 --w_name "1en6" \
--mark 0 > cifar100_res18_test_ex_wa_1en6.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en7/model_swa.ckpt' \
--w_out 0.0000001 --w_out2 0.0000001 --w_name "1en7" \
--mark 0 > cifar100_res18_test_ex_wa_1en7.txt &


CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en3/model_swa.ckpt' \
--w_out 0.001 --w_out2 0.001 --w_name "1en3" \
--mark 0 > cifar100_res18_test_ex_wa_1en3.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en2/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.01 --w_name "1en2" \
--mark 0 > cifar100_res18_test_ex_wa_1en2.txt &

CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en1/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.1 --w_name "1en1" \
--mark 0 > cifar100_res18_test_ex_wa_1en1.txt &

CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en0/model_swa.ckpt' \
--w_out 1.0 --w_out2 1.0 --w_name "1en0" \
--mark 0 > cifar100_res18_test_ex_wa_1en0.txt &


##### diff w1 and w2

CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en4_6/model_swa.ckpt' \
--w_out 0.0001 --w_out2 0.000001 --w_name "1en4_6" \
--mark 0 > cifar100_res18_test_ex_wa_1en4_6.txt &

CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en4_5/model_swa.ckpt' \
--w_out 0.0001 --w_out2 0.00001 --w_name "1en4_5" \
--mark 0 > cifar100_res18_test_ex_wa_1en4_5.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en4_3/model_swa.ckpt' \
--w_out 0.0001 --w_out2 0.001 --w_name "1en4_3" \
--mark 0 > cifar100_res18_test_ex_wa_1en4_3.txt &

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en4_2/model_swa.ckpt' \
--w_out 0.0001 --w_out2 0.01 --w_name "1en4_2" \
--mark 0 > cifar100_res18_test_ex_wa_1en4_2_n.txt &


##### diff increase w1 and w2

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en2_inc_3/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.001 --w_name "1en2_inc_3" \
--mark 0 > cifar100_res18_test_ex_wa_1en2_inc_3.txt &

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en3_inc_3/model_swa.ckpt' \
--w_out 0.001 --w_out2 0.001 --w_name "1en3_inc_3" \
--mark 0 > cifar100_res18_test_ex_wa_1en3_inc_3.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en4_inc_3/model_swa.ckpt' \
--w_out 0.0001 --w_out2 0.001 --w_name "1en4_inc_3" \
--mark 0 > cifar100_res18_test_ex_wa_1en4_inc_3.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en5_inc_3/model_swa.ckpt' \
--w_out 0.00001 --w_out2 0.001 --w_name "1en5_inc_3" \
--mark 0 > cifar100_res18_test_ex_wa_1en5_inc_3.txt &



##### diff decrease w1 and w2

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en2_dec_3/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.001 --w_name "1en2_dec_3" \
--mark 0 > cifar100_res18_test_ex_wa_1en2_dec_3.txt &

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en3_dec_3/model_swa.ckpt' \
--w_out 0.001 --w_out2 0.001 --w_name "1en3_dec_3" \
--mark 0 > cifar100_res18_test_ex_wa_1en3_dec_3.txt &

CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en4_dec_3/model_swa.ckpt' \
--w_out 0.0001 --w_out2 0.001 --w_name "1en4_dec_3" \
--mark 0 > cifar100_res18_test_ex_wa_1en4_dec_3.txt &

CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en5_dec_3/model_swa.ckpt' \
--w_out 0.00001 --w_out2 0.001 --w_name "1en5_dec_3" \
--mark 0 > cifar100_res18_test_ex_wa_1en5_dec_3.txt &


##### diff decrease w1 and w2 tuning seed

CUDA_VISIBLE_DEVICES=1 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en2_dec_3_s7/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.001 --w_name "1en2_dec_3_s7" --seed 7 \
--mark 0 > cifar100_res18_test_ex_wa_1en2_dec_3_s7.txt &

CUDA_VISIBLE_DEVICES=1 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en3_dec_3_s7/model_swa.ckpt' \
--w_out 0.001 --w_out2 0.001 --w_name "1en3_dec_3_s7" --seed 7 \
--mark 0 > cifar100_res18_test_ex_wa_1en3_dec_3_s7.txt &


##### diff decrease w1 and w2 tuning lr (2, 0.2)

CUDA_VISIBLE_DEVICES=1 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en2_dec_3_s7/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.001 --w_name "1en2_dec_3_s7" --seed 5 \
--mark 0 > cifar100_res18_test_ex_wa_1en2_dec_3_s5.txt &

CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en3_dec_3_s7/model_swa.ckpt' \
--w_out 0.001 --w_out2 0.001 --w_name "1en3_dec_3_s7" --seed 5 \
--mark 0 > cifar100_res18_test_ex_wa_1en3_dec_3_s5.txt &


CUDA_VISIBLE_DEVICES=1 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en2_dec_3_s5/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.001 --w_name "1en2_dec_3_s5" --seed 5 \
--mark 0 > cifar100_res18_test_ex_wa_1en2_dec_3_s5.txt &

CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en3_dec_3_s5/model_swa.ckpt' \
--w_out 0.001 --w_out2 0.001 --w_name "1en3_dec_3_s5" --seed 5 \
--mark 0 > cifar100_res18_test_ex_wa_1en3_dec_3_s5.txt &


CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en4/best.ckpt' \
--w_out 0.0001 --w_out2 0.0001 --w_name "1en4" --seed 5 \
--mark 0 > cifar100_res18_test_ex_wa_1en4_s5.txt &

##### weaker ensemble

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en2_3_s5_lr_08_002/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_08_002" --seed 5 --lr_ini 0.8 --lr_end 0.02 \
--mark 0 > cifar100_res18_test_ex_wa_1en3_dec_3_s5.txt &


CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en2_3_s5_lr_12_002/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_12_002" --seed 5 --lr_ini 1.2 --lr_end 0.02 \
--mark 0 > cifar100_res18_test_ex_wa_1en2_3_s5_lr_12_002.txt &


CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en2_3_s5_lr_08_002/best_epoch87_acc0.7720.ckpt' \
--w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_08_002" --seed 5 --lr_ini 0.8 --lr_end 0.02 \
--mark 0 > cifar100_res18_test_ex_wa_1en2_3_s5_lr_08_002_indi.txt &


CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en2_3_s5_lr_12_002/best_epoch92_acc0.7710.ckpt' \
--w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_12_002" --seed 5 --lr_ini 1.2 --lr_end 0.02 \
--mark 0 > cifar100_res18_test_ex_wa_1en2_3_s5_lr_12_002_indi.txt &

##### weaker ensemble (turning acc)

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en2_3_s5_lr_08_0002/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_08_0002" --seed 5 --lr_ini 0.8 --lr_end 0.002 \
--mark 0 > cifar100_res18_test_ex_wa_1en2_3_s5_lr_08_0002.txt &


CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en2_3_s5_lr_12_0002/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_12_0002" --seed 5 --lr_ini 1.2 --lr_end 0.002 \
--mark 0 > cifar100_res18_test_ex_wa_1en2_3_s5_lr_12_0002.txt &


##### weaker ensemble (cifar10)

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default_1en2_3_s5_lr_08_002/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_08_002" --seed 5 --lr_ini 0.8 --lr_end 0.02 \
--mark 0 > cifar10_res18_test_ex_wa_1en2_3_s5_lr_08_002.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default_1en2_3_s5_lr_12_002/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_12_002" --seed 5 --lr_ini 1.2 --lr_end 0.02 \
--mark 0 > cifar10_res18_test_ex_wa_1en2_3_s5_lr_12_002.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default_1en2_3_s5_lr_2_002/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_2_002" --seed 5 --lr_ini 2 --lr_end 0.02 \
--mark 0 > cifar10_res18_test_ex_wa_1en2_3_s5_lr_2_002.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default_1en2_3_s5_lr_15_002/model_swa.ckpt' \
--w_out2 0.001 --w_name "1en2_3_s5_lr_15_002" --seed 5 --lr_ini 1.5 --lr_end 0.02 \
--mark 0 > cifar10_res18_test_ex_wa_1en2_3_s5_lr_15_002.txt &




# (ori)

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/best.ckpt' \
--mark 0 > cifar10_res18_test_ori.txt &


##### weaker ensemble (turning acc)

CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en2_3_s5_lr_15_002/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_15_002" --seed 5 --lr_ini 1.5 --lr_end 0.02 \
--mark 0 > cifar100_res18_test_ex_wa_1en2_3_s5_lr_15_002.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en1_2_s5_lr_15_002/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.01 --w_name "1en1_2_s5_lr_15_002" --seed 5 --lr_ini 1.5 --lr_end 0.02  \
--mark 0 > cifar100_res18_test_ex_wa_1en1_2_s5_lr_15_002.txt &

CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_1en2_3_s5_lr_20_002/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.001 --w_name "1en2_3_s5_lr_20_002" --seed 5 --lr_ini 2 --lr_end 0.02 \
--mark 0 > cifar100_res18_test_ex_wa_1en2_3_s5_lr_20_002.txt &

CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_ex_10_1en1_2_s5_lr_15_002/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.01 --w_name "ex_10_1en1_2_s5_lr_15_002" --seed 5 --lr_ini 1.5 --lr_end 0.02 \
--mark 0 > cifar100_res18_test_ex_wa_ex_10_1en1_2_s5_lr_15_002.txt &


CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e250_lr0.1_default_e_250_50_1en1_2_s5_lr_15_002/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.01 --w_name "e_250_50_1en1_2_s5_lr_15_002" --seed 5 --lr_ini 1.5 --lr_end 0.02 \
--mark 0 > cifar100_res18_test_ex_wa_e_250_50_1en1_2_s5_lr_15_002.txt &



# more average

# c10

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default_more_1en2_3_s5_lr_15_002/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.001 --w_name "more_1en2_3_s5_lr_15_002" --seed 5 --lr_ini 1.5 --lr_end 0.02 > cifar10_res18_test_ex_wa_more_1en2_3_s5_lr_15_002.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default_more_1en2_2_s5_lr_15_01/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.01 --w_name "more_1en2_2_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 > cifar10_res18_test_ex_wa_more_1en2_2_s5_lr_15_01.txt &

CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default_more_1en2_3_s5_lr_15_01/model_swa.ckpt' \
--w_out 0.001 --w_out2 0.01 --w_name "more_1en2_3_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 > cifar10_res18_test_ex_wa_more_1en2_3_s5_lr_15_01.txt &

# c100

CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_more_1en1_2_s5_lr_15_002/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.01 --w_name "more_1en1_2_s5_lr_15_002" --seed 5 --lr_ini 1.5 --lr_end 0.02 \
--mark 0 > cifar100_res18_test_ex_wa_more_1en1_2_s5_lr_15_002.txt &

CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_more_1en2_2_s5_lr_15_01/model_swa.ckpt' \
--w_out 0.01 --w_out2 0.01 --w_name "more_1en2_2_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 \
--mark 0 > cifar100_res18_test_ex_wa_more_1en2_2_s5_lr_15_01.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_more_1en3_2_s5_lr_15_01/model_swa.ckpt' \
--w_out 0.001 --w_out2 0.01 --w_name "more_1en3_2_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 \
--mark 0 > cifar100_res18_test_ex_wa_more_1en3_2_s5_lr_15_01.txt &

## more c100

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_more_1en1_2_s5_lr_01_001/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.01 --w_name "more_1en1_2_s5_lr_01_001" --seed 5 --lr_ini 0.1 --lr_end 0.01 \
--mark 0 > cifar100_res18_test_ex_wa_more_1en1_2_s5_lr_01_001.txt &

CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_more_1en0_0_s5_lr_01_001/model_swa.ckpt' \
--w_out 1.0 --w_out2 1.0 --w_name "more_1en0_0_s5_lr_01_001" --seed 5 --lr_ini 0.1 --lr_end 0.01 \
--mark 0 > cifar100_res18_test_ex_wa_more_1en0_0_s5_lr_01_001.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_more_1en1_2_s5_lr_3_005/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.01 --w_name "more_1en1_2_s5_lr_3_005" --seed 5 --lr_ini 3 --lr_end 0.05 \
--mark 0 > cifar100_res18_test_ex_wa_more_1en1_2_s5_lr_3_005.txt &

CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_more_1en0_0_s5_lr_3_005/model_swa.ckpt' \
--w_out 1.0 --w_out2 1.0 --w_name "more_1en0_0_s5_lr_3_005" --seed 5 --lr_ini 3 --lr_end 0.05 \
--mark 0 > cifar100_res18_test_ex_wa_more_1en0_0_s5_lr_3_005.txt &


## more c100

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_more_1en1_2_s5_lr_15_01/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.01 --w_name "more_1en1_2_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 \
--mark 0 > cifar100_res18_test_ex_wa_more_1en1_2_s5_lr_15_01.txt &

CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_more_1en1_0_s5_lr_15_01/model_swa.ckpt' \
--w_out 0.1 --w_out2 1.0 --w_name "more_1en1_0_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 \
--mark 0 > cifar100_res18_test_ex_wa_more_1en1_0_s5_lr_15_01.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_more_1en1_2_s5_lr_15_0002/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.01 --w_name "more_1en1_2_s5_lr_15_0002" --seed 5 --lr_ini 1.5 --lr_end 0.002 \
--mark 0 > cifar100_res18_test_ex_wa_more_1en1_2_s5_lr_15_0002.txt &

CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_more_1en1_0_s5_lr_15_0002/model_swa.ckpt' \
--w_out 0.1 --w_out2 1.0 --w_name "more_1en1_0_s5_lr_15_0002" --seed 5 --lr_ini 1.5 --lr_end 0.002 \
--mark 0 > cifar100_res18_test_ex_wa_more_1en1_0_s5_lr_15_0002.txt &

## more c100

CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_more_1en1_1_s5_lr_15_01/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.1 --w_name "more_1en1_1_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 \
--mark 0 > cifar100_res18_test_ex_wa_more_1en1_1_s5_lr_15_01.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_more_1en1_1_s5_lr_15_001/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.1 --w_name "more_1en1_1_s5_lr_15_001" --seed 5 --lr_ini 1.5 --lr_end 0.01 \
--mark 0 > cifar100_res18_test_ex_wa_more_1en1_1_s5_lr_15_001.txt &

## more c100

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_more_10_1en1_1_s5_lr_15_01/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.1 --w_name "more_10_1en1_1_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 \
--mark 0 > cifar100_res18_test_ex_wa_more_10_1en1_1_s5_lr_15_01.txt &

CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_more_10_1en1_1_s5_lr_15_005/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.1 --w_name "more_10_1en1_1_s5_lr_15_005" --seed 5 --lr_ini 1.5 --lr_end 0.05 \
--mark 0 > cifar100_res18_test_ex_wa_more_10_1en1_1_s5_lr_15_005.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_more_10_1en1_1_s5_lr_15_0025/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.1 --w_name "more_10_1en1_1_s5_lr_15_0025" --seed 5 --lr_ini 1.5 --lr_end 0.025 \
--mark 0 > cifar100_res18_test_ex_wa_more_10_1en1_1_s5_lr_15_0025.txt &

CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_more_10_1en1_1_s5_lr_15_02/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.1 --w_name "more_10_1en1_1_s5_lr_15_02" --seed 5 --lr_ini 1.5 --lr_end 0.2 \
--mark 0 > cifar100_res18_test_ex_wa_more_10_1en1_1_s5_lr_15_02.txt &

# c100 ori
CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e250_lr0.1_default/best.ckpt' \
--mark 0 > cifar100_res18_test_ori_e250.txt &


## more c100

CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_c_102_more_10_1en1_1_s5_lr_15_01/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.1 --w_name "c_102_more_10_1en1_1_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 \
--mark 0 > cifar100_res18_test_ex_wa_c_102_more_10_1en1_1_s5_lr_15_01.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_c_103_more_10_1en1_1_s5_lr_15_01/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.1 --w_name "c_103_more_10_1en1_1_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 \
--mark 0 > cifar100_res18_test_ex_wa_c_103_more_10_1en1_1_s5_lr_15_01.txt &


CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_c_104_more_10_1en1_1_s5_lr_15_01/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.1 --w_name "c_104_more_10_1en1_1_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 \
--mark 0 > cifar100_res18_test_ex_wa_c_104_more_10_1en1_1_s5_lr_15_01.txt &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default_c_105_more_10_1en1_1_s5_lr_15_01/model_swa.ckpt' \
--w_out 0.1 --w_out2 0.1 --w_name "c_105_more_10_1en1_1_s5_lr_15_01" --seed 5 --lr_ini 1.5 --lr_end 0.1 \
--mark 0 > cifar100_res18_test_ex_wa_c_105_more_10_1en1_1_s5_lr_15_01.txt &