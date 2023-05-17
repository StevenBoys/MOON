import csv
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .base_evaluator import BaseEvaluator
from .metrics import compute_all_metrics


class OODEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        """OOD Evaluator.

        Args:
            config (Config): Config file from
        """
        super(OODEvaluator, self).__init__(config)
        self.id_pred = None
        self.id_conf = None
        self.id_gt = None
        self.config = config

    def eval_ood(self, net: nn.Module, id_data_loader: DataLoader,
                 ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                 postprocessor: BasePostprocessor):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        # load training in-distribution data
        if self.config.dataset.name == 'imagenet':
            is_img = True
        else:
            is_img = False
            assert 'test' in id_data_loader, \
                'id_data_loaders should have the key: test!'

        if is_img:
            dataset_name = self.config.dataset.name
            print(f'Performing inference on {dataset_name} dataset...', flush=True)
            id_pred, id_conf, id_gt = postprocessor.inference(
                net, id_data_loader['val'])
        else:
            dataset_name = self.config.dataset.name
            print(f'Performing inference on {dataset_name} dataset...', flush=True)
            id_pred, id_conf, id_gt = postprocessor.inference(
                net, id_data_loader['test'])


        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        if self.config.postprocessor.APS_mode:
            self.hyperparam_search(net, [id_pred, id_conf, id_gt],
                                   ood_data_loaders['val'], postprocessor)

        # load nearood data and compute ood metrics
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='nearood')
        # load farood data and compute ood metrics
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='farood')

    def _eval_ood(self,
                  net: nn.Module,
                  id_list: List[np.ndarray],
                  ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                  postprocessor: BasePostprocessor,
                  ood_split: str = 'nearood'):
        print(f'Processing {ood_split}...', flush=True)
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        for dataset_name, ood_dl in ood_data_loaders[ood_split].items():
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)
            ood_pred, ood_conf, ood_gt = postprocessor.inference(net, ood_dl, is_id=False)
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            if self.config.recorder.save_scores:
                self._save_scores(ood_pred, ood_conf, ood_gt, dataset_name)

            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            print(f'Computing metrics on {dataset_name} dataset...')

            ood_metrics = compute_all_metrics(conf, label, pred)
            if self.config.recorder.save_csv:
                self._save_csv(ood_metrics, dataset_name=dataset_name)
            metrics_list.append(ood_metrics)

        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        if self.config.recorder.save_csv:
            self._save_csv(metrics_mean, dataset_name=ood_split)

    def _save_csv(self, metrics, dataset_name):
        [fpr, auroc, aupr_in, aupr_out,
         ccr_4, ccr_3, ccr_2, ccr_1, accuracy] \
         = metrics

        write_content = {
            'dataset': dataset_name,
            'FPR@95': '{:.2f}'.format(100 * fpr),
            'AUROC': '{:.2f}'.format(100 * auroc),
            'AUPR_IN': '{:.2f}'.format(100 * aupr_in),
            'AUPR_OUT': '{:.2f}'.format(100 * aupr_out),
            'CCR_4': '{:.2f}'.format(100 * ccr_4),
            'CCR_3': '{:.2f}'.format(100 * ccr_3),
            'CCR_2': '{:.2f}'.format(100 * ccr_2),
            'CCR_1': '{:.2f}'.format(100 * ccr_1),
            'ACC': '{:.2f}'.format(100 * accuracy)
        }

        fieldnames = list(write_content.keys())

        # print ood metric results
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
            100 * aupr_in, 100 * aupr_out),
              flush=True)
        print('CCR: {:.2f}, {:.2f}, {:.2f}, {:.2f},'.format(
            ccr_4 * 100, ccr_3 * 100, ccr_2 * 100, ccr_1 * 100),
              end=' ',
              flush=True)
        print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
        print(u'\u2500' * 70, flush=True)

        csv_path = os.path.join(self.config.output_dir, 'ood.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)

    def _save_scores(self, pred, conf, gt, save_name):
        save_dir = os.path.join(self.config.output_dir, 'scores')
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, save_name),
                 pred=pred,
                 conf=conf,
                 label=gt)

    def eval_acc_ori(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        """Returns the accuracy score of the labels and predictions.

        :return: float
        """
        if type(net) is dict:
            net['backbone'].eval()
        else:
            net.eval()
        self.id_pred, self.id_conf, self.id_gt = postprocessor.inference(
            net, data_loader)
        metrics = {}
        metrics['acc'] = sum(self.id_pred == self.id_gt) / len(self.id_pred)
        metrics['epoch_idx'] = epoch_idx
        return metrics

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1,
                 num_bins: int = 10):
        net.eval()

        if self.config.dataset.name == 'imagenet':
            is_img = True
        else:
            is_img = False
        """Calculates ECE.
        Args:
          num_bins: the number of bins to partition all samples. we set it as 15.
        Returns:
          ece: the calculated ECE value.
        """

        loss_avg = 0.0
        correct = 0
        total_scores = []
        total_preds = []
        total_labels = []
        idx = 1
        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True):
                # prepare data
                if is_img:
                    data, target = batch
                    data, target = data.cuda(), target.cuda()
                else:
                    data = batch['data'].cuda()
                    target = batch['label'].cuda()

                # forward
                output = net(data)
                loss = F.cross_entropy(output, target)
                output = torch.softmax(output, dim=1)

                # accuracy
                pred = output.data.max(1)[1]
                score = output.data.max(1)[0]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

                if idx == 1:
                    total_preds = list(pred.cpu().numpy().reshape(-1))
                    total_scores = list(score.cpu().numpy().reshape(-1))
                    total_labels = list(target.data.cpu().numpy())
                else:
                    total_preds += list(pred.cpu().numpy().reshape(-1))
                    total_scores += list(score.cpu().numpy().reshape(-1))
                    total_labels += list(target.data.cpu().numpy())

                #total_preds.append(pred.cpu().numpy().reshape(-1))
                #total_scores.append(score.cpu().numpy().reshape(-1))
                #total_labels.append(target.data.cpu().numpy().reshape(-1))

                idx += 1

        scores_np = np.reshape(total_scores, -1)
        preds_np = np.reshape(total_preds, -1)
        labels_np = np.reshape(total_labels, -1)
        acc_tab = np.zeros(num_bins)  # Empirical (true) confidence
        mean_conf = np.zeros(num_bins)  # Predicted confidence
        nb_items_bin = np.zeros(num_bins)  # Number of items in the bins
        tau_tab = np.linspace(0, 1, num_bins + 1)  # Confidence bins
        print("number of bins: {}".format(num_bins))
        #print("mean score: {}".format(np.mean(scores_np)))
        for i in np.arange(num_bins):  # Iterates over the bins
            #print(i)
            # Selects the items where the predicted max probability falls in the bin
            # [tau_tab[i], tau_tab[i + 1)]
            sec = (tau_tab[i + 1] > scores_np) & (scores_np >= tau_tab[i])
            #print("tau i+1: {}, tau i: {}, mean sec: {}".format(tau_tab[i + 1], tau_tab[i], sec))
            #print(sec)
            nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
            # Selects the predicted classes, and the true classes
            class_pred_sec, y_sec = preds_np[sec], labels_np[sec]
            # Averages of the predicted max probabilities
            mean_conf[i] = np.mean(
                scores_np[sec]) if nb_items_bin[i] > 0 else np.nan
            #print(mean_conf[i])
            # Computes the empirical confidence
            acc_tab[i] = np.mean(
                class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan
            #print(acc_tab[i])
        # Cleaning
        mean_conf = mean_conf[nb_items_bin > 0]
        acc_tab = acc_tab[nb_items_bin > 0]
        nb_items_bin = nb_items_bin[nb_items_bin > 0]
        print("mean_conf: {}".format(mean_conf))
        print("acc_tab: {}".format(acc_tab))
        print("nb_items_bin: {}".format(nb_items_bin))
        if sum(nb_items_bin) != 0:
            #print("right place")
            ece = np.average(
                np.absolute(mean_conf - acc_tab),
                weights=nb_items_bin.astype(np.float) / np.sum(nb_items_bin))
        else:
            ece = 0.0
            #print("wrong place")

        loss = loss_avg / len(data_loader)
        acc = correct / len(data_loader.dataset)

        print("ece: {}".format(ece))

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss)
        metrics['acc'] = self.save_metrics(acc)
        metrics['ece'] = self.save_metrics(ece)
        return metrics

    def report(self, test_metrics):
        print('Completed!', flush=True)

    def hyperparam_search(
        self,
        net: nn.Module,
        id_list: List[np.ndarray],
        val_data_loader,
        postprocessor: BasePostprocessor,
    ):
        print('Starting automatic parameter search...')
        aps_dict = {}
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0
        for name in postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1
        for name in hyperparam_names:
            hyperparam_list.append(postprocessor.args_dict[name])
        hyperparam_combination = self.recursive_generator(
            hyperparam_list, count)
        for hyperparam in hyperparam_combination:
            postprocessor.set_hyperparam(hyperparam)
            [id_pred, id_conf, id_gt] = id_list

            ood_pred, ood_conf, ood_gt = postprocessor.inference(
                net, val_data_loader)
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            index = hyperparam_combination.index(hyperparam)
            aps_dict[index] = ood_metrics[1]
            print('Hyperparam:{}, auroc:{}'.format(hyperparam,
                                                   aps_dict[index]))
            if ood_metrics[1] > max_auroc:
                max_auroc = ood_metrics[1]
        for key in aps_dict.keys():
            if aps_dict[key] == max_auroc:
                postprocessor.set_hyperparam(hyperparam_combination[key])
        print('Final hyperparam: {}'.format(postprocessor.get_hyperparam()))

    def recursive_generator(self, list, n):
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results
