# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from copy import deepcopy
from .base import BaseTrainer
from .hooks.base_hook import HookContainer
from .hooks import EvaluationRecorder
from torch.utils.data import SequentialSampler, RandomSampler
from collections import defaultdict, Counter

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
class AdaptTuner(BaseTrainer):
    def __init__(self, conf, collocate_batch_fn, logger):
        assert len(conf.adapt_trn_languages) == 1
        super(AdaptTuner, self).__init__(conf, logger)
        self.log_fn("Init trainer.")
        self.collocate_batch_fn = collocate_batch_fn
        self.model_ptl = conf.ptl
        self.num_shots = conf.adapt_num_shots

        # Set base stats
        self.base_stats = './stats/{}/english.pkl'.format(conf.task)
        if os.path.isfile(self.base_stats):
            print ("Use base (english) stats from {}".format(self.base_stats))
            self.base_stats = load_pickle(self.base_stats)
            self.base_mean = {}
            self.base_cov = {}
            for i in list(self.base_stats.keys()):
                self.base_mean[int(i)] = self.base_stats[i][0]['mean']
                self.base_cov[int(i)] = self.base_stats[i][0]['cov']
        else:
            pass

    def _init_model_opt(self, model):
        model = self._parallel_to_device(model)
        trn_params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trn_params, lr=self.conf.adapt_lr)
        opt.zero_grad()
        model.zero_grad()
        return opt, model

    def _infer_tst_egs(
        self, hook_container, data_iter, metric_name, adapt_loaders, tst_languages
    ):
        assert isinstance(tst_languages, list)
        best_model = deepcopy(
            self._get_eval_recorder_hook(hook_container).best_state["best_state_dict"]
        ).cuda()
        scores = defaultdict(dict)
        for language in tst_languages:
            for split_name in ["tst_egs"]:
                loader = getattr(adapt_loaders[language], split_name)
                if self.conf.dataset_name in ["conll2003", "panx", "udpos"]:
                    eval_res, *_ = self._infer_one_loader_tagging(
                        model=best_model,
                        idx2label=data_iter[language].raw_dataset.idx2label,
                        loader=loader,
                        collocate_batch_fn=self.collocate_batch_fn,
                        metric_name=metric_name,
                    )
                else:
                    eval_res, *_ = self._infer_one_loader(
                        model=best_model,
                        loader=loader,
                        collocate_batch_fn=self.collocate_batch_fn,
                        metric_name=metric_name,
                        # transformation_vector=self.transformation_vector
                    )
                scores[language][split_name] = eval_res
        return scores
    
    def _train(
        self, model, tokenizer, data_iter, metric_name, adapt_loaders, hooks=None
    ):
        opt, model = self._init_model_opt(model)
        self.model = model
        self.model.train()

        hook_container = HookContainer(world_env={"trainer": self}, hooks=hooks)
        hook_container.on_train_begin()

        adapt_language = self.conf.adapt_trn_languages[0]
        learning_curves = {"val_egs": defaultdict(list)}

        for epoch_index in range(1, self.conf.adapt_epochs + 1):
            all_uids, epoch_losses = [], []
            for batched in adapt_loaders[adapt_language].trn_egs:
                batched, golds, uids, _golds_tagging = self.collocate_batch_fn(batched)
                logits, *_ = self._model_forward(self.model, **batched)
                loss = self.criterion(logits, golds).mean()
                epoch_losses.append(loss.item())
                loss.backward()
                opt.step()
                opt.zero_grad()
                all_uids.extend(uids)
                self._batch_step += 1
            epoch_losses_str = "->".join(
                [f"{epoch_loss:.3f}" for epoch_loss in epoch_losses]
            )
            self.log_fn(
                f"epoch loss: {np.mean(epoch_losses):.3f} "
                f"epoch @ {epoch_index} "
                f"detailed loss {epoch_losses_str}"
            )
            self.log_fn(f"{all_uids}")
            self.log_fn("*" * 10)

            scores = defaultdict(dict)
            for language in [adapt_language]:
                for split_name in ["val_egs"]:
                    loader = getattr(adapt_loaders[language], split_name)
                    if self.conf.dataset_name in ["conll2003", "panx", "udpos"]:
                        eval_res, *_ = self._infer_one_loader_tagging(
                            self.model,
                            data_iter[language].raw_dataset.idx2label,
                            loader,
                            self.collocate_batch_fn,
                        )
                    else:
                        eval_res, *_ = self._infer_one_loader(
                            model=self.model,
                            loader=loader,
                            collocate_batch_fn=self.collocate_batch_fn,
                            metric_name=metric_name,
                        )
                    scores[language][split_name] = eval_res
                    learning_curves[split_name][language].append(eval_res)
            eval_score = scores[adapt_language]["val_egs"]
            hook_container.on_validation_end(eval_score=eval_score, all_scores=scores)
            best_epoch_step = self._get_eval_recorder_hook(hook_container).best_epoch
            if (
                self.conf.early_stop
                and epoch_index - best_epoch_step > self.conf.early_stop_patience
            ):
                self.log_fn(
                    f"Early-stopping: current epoch={epoch_index},"
                    f" best_epoch={best_epoch_step + 1}."
                )
                tst_scores = self._infer_tst_egs(
                    hook_container,
                    data_iter,
                    metric_name,
                    adapt_loaders,
                    [self.conf.adapt_trn_languages[0]],
                )
                hook_container.on_train_end(
                    learning_curves=learning_curves, tst_scores=tst_scores,
                )
                return
            self._epoch_step += 1
        tst_scores = self._infer_tst_egs(
            hook_container,
            data_iter,
            metric_name,
            adapt_loaders,
            [self.conf.adapt_trn_languages[0]],
        )
        hook_container.on_train_end(
            learning_curves=learning_curves, tst_scores=tst_scores,
        )
        return
        
        
    def _train_fl(
        self, model, tokenizer, data_iter, metric_name, adapt_loaders, hooks=None
    ):
        opt, model = self._init_model_opt(model)
        self.model = model
        self.model.train()

        hook_container = HookContainer(world_env={"trainer": self}, hooks=hooks)
        hook_container.on_train_begin()

        adapt_language = self.conf.adapt_trn_languages[0]
        learning_curves = {"val_egs": defaultdict(list)}

        # Make transformation vector
        source_mean_vector = torch.tensor(list(self.base_mean.values())).cuda()
        source_mean_vector = torch.mean(source_mean_vector, dim=0)

        # target_mean_vector = torch.zeros(source_mean_vector.shape).cuda()
        # n_samples = 0
        # for batched in adapt_loaders[adapt_language].trn_egs:
        #     batched, golds, uids, _golds_tagging = self.collocate_batch_fn(batched)
        #     with torch.no_grad():
        #         last_hiddens = self.model.get_last_hidden(**batched)
        #         target_mean_vector = target_mean_vector + torch.sum(last_hiddens, dim=0)
        #         n_samples += last_hiddens.shape[0]
        # target_mean_vector = target_mean_vector / n_samples
        # self.transformation_vector = (source_mean_vector - target_mean_vector).float()
        self.transformation_vector = torch.zeros(source_mean_vector.shape).cuda()

        # Generate class (last hiddens)
        aug_num_per_class = 250
        X_aug = []
        Y_aug = []

        self.target_mean = {}
        for k, v in self.base_mean.items():
            self.target_mean[k] = np.zeros(v.shape)

        for batched in adapt_loaders[adapt_language].trn_egs:
            batched, golds, uids, _golds_tagging = self.collocate_batch_fn(batched)
            with torch.no_grad():
                last_hiddens = self.model.get_last_hidden(**batched)
                for last_hidden, gold in list(zip(last_hiddens, golds)):
                    self.target_mean[gold.item()] += last_hidden.cpu().numpy()

        self.target_mean = {k: v/self.num_shots for k, v in self.target_mean.items()}

        classes = list(self.base_mean.keys())
        for k in classes:
            b_mean = self.base_mean[k]
            t_mean = self.target_mean[k]
            
            if k == 0 :
                mean = t_mean
                cov = self.base_cov[k]
            else:
                mean = ((self.target_mean[0] + b_mean - self.base_mean[0]) + t_mean) / 2
                cov = self.base_cov[k]

            X_aug_per_class = np.random.multivariate_normal(mean=mean, cov=cov, size=aug_num_per_class)
            Y_aug_per_class = [k] * aug_num_per_class

            if len(X_aug) == 0:
                X_aug = X_aug_per_class
                Y_aug = np.array(Y_aug_per_class)
            else:
                X_aug = np.concatenate([X_aug, X_aug_per_class])
                Y_aug = np.concatenate([Y_aug, np.array(Y_aug_per_class)])

        # gamma = 0.0
        # mean = gamma * self.base_mean[gold.item()] + (1-gamma) * last_hidden
        # cov = self.base_cov[gold.item()]
        # X_aug_per_sample = np.random.multivariate_normal(mean=mean, cov=cov, size=aug_num_per_class)
        # Y_aug_per_sample = [gold.item()] * aug_num_per_class
        # if len(X_aug) == 0:
        #     X_aug = X_aug_per_sample
        #     Y_aug = np.array(Y_aug_per_sample)
        # else:
        #     X_aug = np.concatenate([X_aug, X_aug_per_sample])
        #     Y_aug = np.concatenate([Y_aug, Y_aug_per_sample])

        for epoch_index in range(1, self.conf.adapt_epochs + 1):
            all_uids, epoch_losses = [], []

            shuffle_idx = np.arange(len(Y_aug))
            np.random.shuffle(shuffle_idx)
            shuffle_X_aug = torch.tensor(X_aug[shuffle_idx]).float()
            shuffle_Y_aug = torch.tensor(Y_aug[shuffle_idx]).long()

            for idx, batched in enumerate(adapt_loaders[adapt_language].trn_egs):
                batched, golds, uids, _golds_tagging = self.collocate_batch_fn(batched)
                # logits, *_ = self._model_forward(self.model, **batched)
                last_hiddens = self.model.get_last_hidden(**batched)

                # last_hiddens = torch.where(last_hiddens > 0, last_hiddens, torch.tensor(0.).cuda())
                # last_hiddens = torch.pow(last_hiddens, beta)
                last_hiddens = torch.cat([last_hiddens,
                                          shuffle_X_aug[idx*len(golds)*aug_num_per_class:(idx+1)*len(golds)*aug_num_per_class,:].cuda()],
                                          dim=0)
                logits = self.model.get_logits_from_last_hidden(last_hiddens)
                golds = torch.cat([golds,
                                   shuffle_Y_aug[idx*len(golds)*aug_num_per_class:(idx+1)*len(golds)*aug_num_per_class].cuda()],
                                   dim=0)
                loss = self.criterion(logits, golds).mean()
                epoch_losses.append(loss.item())
                loss.backward()
                opt.step()
                opt.zero_grad()
                all_uids.extend(uids)
                self._batch_step += 1
            epoch_losses_str = "->".join(
                [f"{epoch_loss:.3f}" for epoch_loss in epoch_losses]
            )
            self.log_fn(
                f"epoch loss: {np.mean(epoch_losses):.3f} "
                f"epoch @ {epoch_index} "
                f"detailed loss {epoch_losses_str}"
            )
            self.log_fn(f"{all_uids}")
            self.log_fn("*" * 10)

            scores = defaultdict(dict)
            for language in [adapt_language]:
                for split_name in ["val_egs"]:
                    loader = getattr(adapt_loaders[language], split_name)
                    if self.conf.dataset_name in ["conll2003", "panx", "udpos"]:
                        eval_res, *_ = self._infer_one_loader_tagging(
                            self.model,
                            data_iter[language].raw_dataset.idx2label,
                            loader,
                            self.collocate_batch_fn,
                        )
                    else:
                        eval_res, *_ = self._infer_one_loader(
                            model=self.model,
                            loader=loader,
                            collocate_batch_fn=self.collocate_batch_fn,
                            metric_name=metric_name,
                            # transformation_vector=self.transformation_vector
                        )
                    scores[language][split_name] = eval_res
                    learning_curves[split_name][language].append(eval_res)
            eval_score = scores[adapt_language]["val_egs"]
            hook_container.on_validation_end(eval_score=eval_score, all_scores=scores)
            best_epoch_step = self._get_eval_recorder_hook(hook_container).best_epoch
            if (
                self.conf.early_stop
                and epoch_index - best_epoch_step > self.conf.early_stop_patience
            ):
                self.log_fn(
                    f"Early-stopping: current epoch={epoch_index},"
                    f" best_epoch={best_epoch_step + 1}."
                )
                tst_scores = self._infer_tst_egs(
                    hook_container,
                    data_iter,
                    metric_name,
                    adapt_loaders,
                    [self.conf.adapt_trn_languages[0]],
                )
                hook_container.on_train_end(
                    learning_curves=learning_curves, tst_scores=tst_scores,
                )
                return
            self._epoch_step += 1
        tst_scores = self._infer_tst_egs(
            hook_container,
            data_iter,
            metric_name,
            adapt_loaders,
            [self.conf.adapt_trn_languages[0]],
        )
        hook_container.on_train_end(
            learning_curves=learning_curves, tst_scores=tst_scores,
        )
        return
        

    def train(
        self, model, tokenizer, data_iter, metric_name, adapt_loaders, 
        hooks=None, use_fl=False
    ):
        if use_fl:
            return self._train_fl(model, tokenizer, data_iter, metric_name,
                                  adapt_loaders, hooks=hooks)
        else:
            return self._train(model, tokenizer, data_iter, metric_name,
                               adapt_loaders, hooks=hooks)