from openood.pipelines import get_pipeline
from openood.utils import launch, setup_config

import torch
import numpy as np
import random


def main(config):
    """Main entrance. Config is all you need to provide to run the code. Config
    should be provided in the format of YAML and can be modified with command
    line.

    Example:
        python main.py \
            --config configs/datasets/mnist_datasets.yml \
            configs/train/mnist_baseline.yml \
            --dataset.image_size 32 \
            --network res18

    Note:
        A config file is the minimum requirement.
        You don't need to add "--config_key new_value"
        if you don't have anything to modify.
    """

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)  # if you are using multi-GPU.
    np.random.seed(config.seed)  # Numpy module.
    random.seed(config.seed)  # Python random module.
    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    """
    def _init_fn(worker_id):
        np.random.seed(int(config.seed))
    """
    pipeline = get_pipeline(config)
    pipeline.run()


if __name__ == '__main__':

    config = setup_config(seed=5)
    # generate output directory and save the full config file
    # setup_logger(config)

    # pipeline = get_pipeline(config)
    # pipeline.run()

    launch(
        main,
        config.num_gpus,
        num_machines=config.num_machines,
        machine_rank=config.machine_rank,
        dist_url='auto',
        args=(config, ),
    )
