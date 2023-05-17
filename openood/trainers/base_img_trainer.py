import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.losses.extra_c_loss import extra_c_loss
import openood.utils.comm as comm
from openood.utils import Config
from openood.utils import (
    moving_average,
    bn_update,
)

from .lr_scheduler import cosine_annealing


def lr_tmp(step, tr_len, epochs, lr, lr_min, deduct=20):
    epoch_num = step // tr_len
    #lr_min = 0.12
    #deduct = 50
    if epoch_num < epochs - deduct:
        return cosine_annealing(
            step,
            (epochs - deduct) * tr_len,
            lr,
            lr_min,
        )
    else:
        return lr_min


class BaseTrainer_img:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.model_swa = None
        self.train_loader = train_loader
        self.config = config
        self.ite_num = len(train_loader)
        #print("ite num: {}; loader len: {}".format(self.ite_num, len(train_loader)))
        self.ave_num = int(self.ite_num / 10)
        self.not_begin_swa = 1
        self.swa_num = 0

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: lr_tmp(
                step,
                len(train_loader),config.optimizer.num_epochs,
                config.lr_ini,
                config.lr_end,
                config.optimizer.deduct,
                #0.5,
                #0.05,
            ),
        )
        '''
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1.2,
                0.12
                #3e-2 / config.optimizer.lr,
            ),
        )
        '''
        w_out = self.config.w_out #* 0.95**epoch_idx; 
        w_out2 = self.config.w_out2
        self.criterion = extra_c_loss(model=net, learn_epochs=5, total_epochs=config.optimizer.num_epochs,  
                        use_cuda=True, alpha_final=w_out, alpha_init_factor=w_out2)

    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            """
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()
            """
            data, target = next(train_dataiter)
            data, target = data.cuda(), target.cuda()

            # forward
            logits_classifier = self.net(data)

            loss = self.criterion(logits_classifier, target, epoch_idx)
            #print("losss at {} epoch and {} ite: {}".format(epoch_idx, train_step, loss.item()))
            #print(self.criterion.alpha_thresh)
            #print(self.criterion.alpha_var)

            """
            probs = torch.softmax(logits_classifier, dim=1)
            probs_outside = torch.sum(probs[:, 10:], axis=1)
            probs_outside = probs_outside.view(-1)
            probs_outside = torch.log(probs_outside)
            loss = F.nll_loss(F.log_softmax(logits_classifier, dim=1), target, reduction='none')

            _, pred = logits_classifier.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred)).float()
            correct = correct.view(-1)

            pred_out_bound = (1 + w_out * 1 / (1 + w_out2 * probs_outside)) * (1 - correct) + correct
            pred_out_bound /= pred_out_bound.sum()
            pred_out_bound = pred_out_bound.view(-1)
            loss = (loss * pred_out_bound).sum()
            #loss = F.cross_entropy(logits_classifier, target)
            """

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            """
            if self.not_begin_swa == 0:
                if (train_step-1) // self.ave_num == 0:
                    moving_average(self.model_swa, self.net, 1/(self.swa_num+1))
                    self.swa_num += 1
                    bn_update(self.train_loader, self.model_swa)  
            """              
            if self.not_begin_swa == 0:
                if (train_step-1) // self.ave_num == 0:
                    moving_average(self.model_swa, self.net, 1/(self.swa_num+1))
                    self.swa_num += 1
                    #bn_update(self.train_loader, self.model_swa)  

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()
        if self.not_begin_swa == 0:
            bn_update(self.train_loader, self.model_swa) 
        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced
