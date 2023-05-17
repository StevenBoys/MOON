# Towards Reliable Sparse Training in Real World: What's Unknown is What's Undeservedly-ignored

This is the code repository of RigL-based and SET-based models in the following paper: Towards Reliable Sparse Training in Real World: What's Unknown is What's Undeservedly-ignored.

Parts of this code repository is based on the following works by the machine learning community.

* https://github.com/Jingkang50/OpenOOD
* https://github.com/varun19299/rigl-reproducibility

## Get Started

To setup the environment, we use `conda` to manage our dependencies.

Our developers use `CUDA 10.1` to do experiments.

You can specify the appropriate `cudatoolkit` version to install on your machine in the `environment.yml` file, and then run the following to create the `conda` environment:
```bash
conda env create -f environment.yml
conda activate openood
pip install libmr==0.1.9 # if necessary
```

Datasets are provided [here](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/Eso7IDKUKQ9AoY7hm9IU2gIBMWNnWGCYPwClpH0TASRLmg?e=kMrkVQ).
Please unzip the files if necessary.
We also provide an automatic data download script [here](https://github.com/Jingkang50/OpenOOD/blob/main/scripts/download/).

Our codebase accesses the datasets from `./data/` and pretrained models from `./results/checkpoints/` by default.
```
├── ...
├── data
│   ├── benchmark_imglist
│   ├── images_classic
│   ├── images_medical
│   └── images_largescale
├── openood
├── results
│   ├── checkpoints
│   └── ...
├── scripts
├── main.py
├── ...
```

## Example Code

Run ResNet-18 on CIFAR-10 in 90% sparsity using SET.
```
python main.py --config configs/datasets/cifar10/cifar10.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/baseline.yml configs/config.yml configs/masking/SET.yml configs/specific/cifar10.yml --w_out 1.0 --w_out2 64.0 --w_name "c10" --seed 10 --lr_ini 1.2 --lr_end 0.08 --classes 11 --density 0.1 --masking.density 0.1 
```

Evaluate ResNet-18 on CIFAR-10 in 90% sparsity using SET.
```
python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
configs/config.yml \
configs/config.yml configs/masking/SET.yml configs/specific/cifar10.yml \
--num_workers 8 \
--network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default_c10/model_swa.ckpt' \
--w_out 1.0 --w_out2 64.0 --w_name "c10_test_set" --seed 5 --lr_ini 1.2 --lr_end 0.08 \
--classes 11 --density 0.1 --masking.density 0.1 --mark 0 
```


## References

1. OpenOOD: Benchmarking Generalized Out-of-Distribution Detection, (https://arxiv.org/abs/2210.07242).
2. Rigging the Lottery: Making All Tickets Winners, (https://arxiv.org/abs/1911.11134).