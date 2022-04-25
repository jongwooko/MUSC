# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from .base import BaseTrainer
from .hooks.base_hook import HookContainer
from .hooks import EvaluationRecorder
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict, Counter
from tqdm import tqdm


class BaselineTuner(BaseTrainer):
    def __init__(self, conf, collocate_batch_fn, logger):
        super(BaselineTuner, self).__init__(conf, logger)
        self.log_fn("Init trainer.")
        self.collocate_batch_fn = collocate_batch_fn
        self.model_ptl = conf.ptl

    def _init_model_opt(self, model):
        print ("Initialize Optimizer and Model")
        opt = torch.optim.Adam(model.parameters(), lr=self.conf.finetune_lr)
        opt.zero_grad()
        model.zero_grad()
        return opt, model

    def _infer_tst_egs(self, hook_container, metric_name, adapt_loaders, tst_languages):
        assert isinstance(tst_languages, list)
        best_model = deepcopy(
            self._get_eval_recorder_hook(hook_container).best_state["best_state_dict"]
        ).cuda()
        scores = defaultdict(dict)
        for language in tst_languages:
            for split_name in ["tst_egs"]:
                loader = getattr(adapt_loaders[language], split_name)
                if self.conf.dataset_name in ["conll2003", "panx", "udpos"]:
                    idx2label = adapt_loaders[language].raw_dataset.idx2label
                    eval_res, *_ = self._infer_one_loader_tagging(
                        best_model,
                        idx2label,
                        loader,
                        self.collocate_batch_fn,
                        metric_name=metric_name,
                    )
                else:
                    eval_res, *_ = self._infer_one_loader(
                        best_model,
                        loader,
                        self.collocate_batch_fn,
                        metric_name=metric_name,
                    )
                self.log_fn(f"{language} {split_name} score: {eval_res * 100:.1f}")
                scores[language][split_name] = eval_res
        return scores

    def train(
        self, model, tokenizer, data_iter, metric_name, adapt_loaders, hooks=None
    ):
        print ("first of all")
        opt, model = self._init_model_opt(model)
        self.model = model
        self.model.train()

        print ("self.model.train()")
        
        hook_container = HookContainer(world_env={"trainer": self}, hooks=hooks)
        hook_container.on_train_begin()

        print ("hook_container.on_train_begin()")
        
        for epoch_index in tqdm(range(1, self.conf.finetune_epochs + 1)):
            trn_iters = []
            for language in self.conf.trn_languages:
                try:
                    egs = adapt_loaders[language].trn_egs
                    assert isinstance(egs.sampler, RandomSampler) or isinstance(egs.sampler, DistributedSampler)
                except:
                    egs = adapt_loaders[language].val_egs
                trn_iters.append(iter(egs))
                
            batches_per_epoch = max(len(ti) for ti in trn_iters)
            for batch_index in range(1, batches_per_epoch + 1): # 
                trn_loss = []
                for ti in trn_iters:
                    try:
                        batched = next(ti)
                    except StopIteration:
                        continue
                    batched, golds, uids, _golds_tagging = self.collocate_batch_fn(
                        batched
                    )

                    # if len(golds.size()) == 2:
                    #     for k in batched.keys():
                    #         batched[k] = torch.cat([batched[k][:, 0], batched[k][:, 1]], dim=0)
                    #     golds = torch.cat([golds[:, 0], golds[:, 1]], dim=0)
                    # logits, feats, *_ = self._model_forward(self.model, **batched)
                    # loss = self.criterion(logits, golds).mean()

                    if len(golds.size()) == 2:
                        batched_eng = {}
                        batched_oth = {}
                        for k in batched.keys():
                            batched_eng[k] = batched[k][:, 0]
                            batched_oth[k] = batched[k][:, 1]
                        golds = golds[:, 0]

                    logits_eng, feats_eng, *_ = self._model_forward(self.model, **batched_eng)
                    logits_oth, feats_oth, *_ = self._model_forward(self.model, **batched_oth)

                    # loss = self.criterion(logits_oth, golds).mean()
                    loss = self.criterion(logits_eng, golds).mean() + \
                           self.criterion(logits_oth, golds).mean() + \
                           0.1 * nn.functional.mse_loss(feats_oth, feats_eng)
                    trn_loss.append(loss.item())
                    loss.backward()

                opt.step()
                opt.zero_grad()
                self._batch_step += 1
                self.log_fn(
                    f"Traning loss on {self.conf.trn_languages}: {np.mean(trn_loss):.3f}"
                    f" train batch  @  {batch_index}, epoch @ {epoch_index}"
                    f" global batch @ {self._batch_step}"
                )
                if self._batch_step % self.conf.eval_every_batch == 0: # and \
                    # (self.conf.rank==0 or self.conf.local_rank == -1):
                    if self.conf.dataset_name in ["conll2003", "panx", "udpos"]:
                        eval_score, all_scores = self.plain_eval_tagging(
                            self.model, adapt_loaders, metric_name=metric_name
                        )
                    else:
                        eval_score, all_scores = self.plain_eval(
                            self.model, adapt_loaders, metric_name=metric_name,
                        )
                    self.log_fn("--" * 10)
                    self.log_fn(f"Evaluate @ batch {self._batch_step}:")
                    self.log_fn(f"metrics: {metric_name}")
                    self.log_fn(f"val score: {eval_score}, all: {all_scores.items()}")
                    self.log_fn("--" * 10)
                    hook_container.on_validation_end(
                        eval_score=eval_score, all_scores=all_scores, evaled_model=None,
                    )
        tst_scores = self._infer_tst_egs(
            hook_container, metric_name, adapt_loaders, self.conf.eval_languages,
        )
        hook_container.on_train_end(
            learning_curves=None, tst_scores=tst_scores, tst_learning_curves=None,
        )
        return

    def plain_eval(self, model, adapt_loaders, metric_name):
        all_scores = defaultdict(list)
        val_scores = []
        for val_language in self.conf.eval_languages: # trn 
            val_loaders = adapt_loaders[val_language]
            # for split_ in ["val_egs"]:
            for split_ in ["val_egs", "tst_egs"]:
                val_loader = getattr(val_loaders, split_)
                eval_res, _ = self._infer_one_loader(
                    model, val_loader, self.collocate_batch_fn, metric_name=metric_name
                )
                all_scores[val_language].append((split_, eval_res))
                if split_ == "val_egs":
                    val_scores.append(eval_res)
        assert len(val_scores) == len(self.conf.eval_languages) # trn
        return (np.mean(val_scores), all_scores)

    def plain_eval_tagging(self, model, adapt_loaders, metric_name):
        all_scores = defaultdict(list)
        val_scores = []
        for val_language in self.conf.eval_languages: # trn
            val_loaders = adapt_loaders[val_language]
            idx2label = adapt_loaders[val_language].raw_dataset.idx2label
            # for split_ in ["val_egs"]:
            for split_ in ["val_egs", "tst_egs"]:
                val_loader = getattr(val_loaders, split_)
                eval_res, _ = self._infer_one_loader_tagging(
                    model, idx2label, val_loader, self.collocate_batch_fn, metric_name=metric_name,
                )
                all_scores[val_language].append((split_, eval_res))
                if split_ == "val_egs":
                    val_scores.append(eval_res)
        assert len(val_scores) == len(self.conf.eval_languages) # trn
        return (np.mean(val_scores), all_scores)