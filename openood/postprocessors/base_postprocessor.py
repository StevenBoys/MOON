from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BasePostprocessor:
    def __init__(self, config):
        self.config = config
        self.num_c = config.ood_dataset['num_classes']

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score[:, :self.num_c], dim=1)
        #conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self, net: nn.Module, data_loader: DataLoader, is_id=True):
        if self.config.dataset.name == 'imagenet':
            is_img = True
        else:
            is_img = False
        
        pred_list, conf_list, label_list = [], [], []
        for batch in data_loader:
            if is_img and is_id:
                data, label = batch
                data, label = data.cuda(), label.cuda()
            else:
                data = batch['data'].cuda()
                label = batch['label'].cuda()
            pred, conf = self.postprocess(net, data)
            for idx in range(len(data)):
                pred_list.append(pred[idx].cpu().tolist())
                conf_list.append(conf[idx].cpu().tolist())
                label_list.append(label[idx].cpu().tolist())

        # convert values into numpy array
        pred_list = np.array(pred_list, dtype=int)
        conf_list = np.array(conf_list)
        label_list = np.array(label_list, dtype=int)

        return pred_list, conf_list, label_list
