import openood.utils.comm as comm
from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.utils import setup_logger

from openood.utils import (
    moving_average,
    bn_update,
)
import copy
import os
import torch
from pathlib import Path
#import numpy as np
#import random

#from data import get_dataloaders
#from loss import LabelSmoothingCrossEntropy
from models import registry as model_registry
from sparselearning.core import Masking
from sparselearning.funcs.decay import registry as decay_registry
from sparselearning.utils.accuracy_helper import get_topk_accuracy
from sparselearning.utils.smoothen_value import SmoothenValue
from sparselearning.utils import layer_wise_density
from sparselearning.utils.train_helper import (
    load_weights,
    save_weights,
)
from sparselearning.utils.utils_ens import (
    moving_average,
    moving_average_unbalan,
    scaling_model,
    bn_update,
    mask_moving_average,
)

class TrainPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)
        print("Using seed = {}".format(self.config.seed))
        """
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        np.random.seed(seed=self.config.seed)
        random.seed(self.config.seed)
        """

        # Set device
        if self.config.device == "cuda" and torch.cuda.is_available():
            device = torch.device(self.config.device)
        else:
            device = torch.device("cpu")

        # get dataloader
        loader_dict = get_dataloader(self.config)
        train_loader, val_loader = loader_dict['train'], loader_dict['val']
        if self.config.dataset.name == "imagenet":
            test_loader = val_loader
        else:
            test_loader = loader_dict['test']

        # init network
        net = get_network(self.config.network)
        net = net.to(device)

        # init trainer and evaluator
        trainer = get_trainer(net, train_loader, self.config)
        evaluator = get_evaluator(self.config)

        # Setup mask
        mask = None
        if self.config.density != 1.0:
            max_iter = (
                self.config.masking.end_when
                if self.config.masking.apply_when == "step_end"
                else self.config.masking.end_when * len(train_loader)
            )
            kwargs = {"prune_rate": self.config.masking.prune_rate, "T_max": max_iter}

            if self.config.masking.decay_schedule == "magnitude-prune":
                kwargs = {
                    "final_sparsity": 1 - self.config.masking.final_density,
                    "T_max": max_iter,
                    "T_start": self.config.masking.start_when,
                    "interval": self.config.masking.interval,
                }

            decay = decay_registry[self.config.masking.decay_schedule](**kwargs)

            if self.config.dataset.name == 'imagenet':
                is_img = True
            else:
                is_img = False
            if is_img:
                input_size = (1, 3, 224, 224)
            else:
                input_size = (1, 3, 32, 32)

            mask = Masking(
                trainer.optimizer,
                decay,
                density=self.config.masking.density,
                dense_gradients=self.config.masking.dense_gradients,
                sparse_init=self.config.masking.sparse_init,
                prune_mode=self.config.masking.prune_mode,
                growth_mode=self.config.masking.growth_mode,
                redistribution_mode=self.config.masking.redistribution_mode,
                input_size=input_size,
            )
            # Support for lottery mask
            lottery_mask_path = Path(self.config.masking.get("lottery_mask_path", ""))
            mask.add_module(net, lottery_mask_path)

        # Load from checkpoint
        start_epoch = 1
        if mask != None:
            net, trainer.optimizer, mask, step, start_epoch, best_val_loss = load_weights(
                net, trainer.optimizer, mask, ckpt_dir=self.config.output_dir, resume=self.config.resume
            )

        trainer.mask = mask


        if comm.is_main_process():
            # init recorder
            recorder = get_recorder(self.config)

            print('Start training...', flush=True)

        begin_save = int(self.config.optimizer.num_epochs-self.config.optimizer.deduct)
        not_begin_swa = 1
        val_metrics = evaluator.eval_acc(net, val_loader, None, 1)
        for epoch_idx in range(start_epoch, self.config.optimizer.num_epochs + 1):
            # train and eval the model
            net, train_metrics = trainer.train_epoch(epoch_idx)
            val_metrics = evaluator.eval_acc(net, val_loader, None, epoch_idx)
            comm.synchronize()
            if comm.is_main_process():
                # save model and report the result
                recorder.save_model(net, val_metrics)
                recorder.report(train_metrics, val_metrics)

            if epoch_idx >= begin_save:
                print('begin swa: {}'.format(epoch_idx))
                if trainer.not_begin_swa:
                    trainer.swa_num = 1
                    trainer.model_swa = copy.deepcopy(net)
                    trainer.not_begin_swa = 0
                print('end swa: {}'.format(epoch_idx))
                if mask != None:
                    save_weights(
                        trainer.model_swa,
                        trainer.optimizer,
                        trainer.mask,
                        0.0,
                        step,
                        999,
                        ckpt_dir=self.config.output_dir,
                        is_min=False,
                    )
                else:
                    torch.save(
                        trainer.model_swa.state_dict(),
                        os.path.join(
                            self.config.output_dir,
                            'model_swa.ckpt'))

            if (
                trainer.mask
                and self.config.masking.apply_when == "epoch_end"
                and (epoch_idx-1) < self.config.masking.end_when
            ):
                if (epoch_idx-1) % self.config.masking.interval == 0:
                    trainer.mask.update_connections()


        if comm.is_main_process():
            recorder.summary()
            print(u'\u2500' * 70, flush=True)

            # evaluate on test set
            print('Start testing...', flush=True)

        test_metrics = evaluator.eval_acc(net, test_loader)

        if comm.is_main_process():
            print('\nComplete Evaluation, Last accuracy {:.2f}'.format(
                100.0 * test_metrics['acc']),
                  flush=True)
            print('Completed!', flush=True)
