from .bert_formatting import tagging_example_to_feature, glue_example_to_feature
import json, pickle
import uuid
import torch
import os

class TaggingDataIter(object):
    def __init__(self, raw_dataset, model, tokenizer, max_seq_len, do_cache=True):
        self.raw_dataset = raw_dataset
        if raw_dataset.trn_egs is not None:
            self.trn_egs = self.wrap_iter(
                task=raw_dataset.name,
                model=model,
                which_split="trn",
                tagged_sents=raw_dataset.trn_egs,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                do_cache=do_cache,
            )
        if raw_dataset.val_egs is not None:
            self.val_egs = self.wrap_iter(
                task=raw_dataset.name,
                model=model,
                which_split="val",
                tagged_sents=raw_dataset.val_egs,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                do_cache=do_cache,
            )
        if raw_dataset.tst_egs is not None:
            self.tst_egs = self.wrap_iter(
                task=raw_dataset.name,
                model=model,
                which_split="tst",
                tagged_sents=raw_dataset.tst_egs,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                do_cache=do_cache,
            )
            
    def wrap_iter(self, task, model, which_split, tagged_sents, tokenizer, max_seq_len, do_cache):
        cached_ = os.path.join(
            "/input/jongwooko/xlt/data", # change this part
            "cached",
            f"{task},{max_seq_len},{model},{which_split},{len(tagged_sents)},cached.pkl",
        )
        os.makedirs(os.path.dirname(cached_), exist_ok=True)
        meta_ = cached_.replace(".pkl", ".metainfo")
        if os.path.exists(cached_) and do_cache:
            print(f"[INFO] loading cached dataset for {task}_{which_split}.")
            with open(meta_, "r") as f:
                meta = json.load(f)
            assert meta["complete"]
            with open(cached_, "rb") as f:
                fts = pickle.load(f)
            if fts["uid"] == meta["uid"]:
                fts = fts["fts"]
            else:
                raise ValueError("uids of data and meta do not match ...")
        else:
            print(f"[INFO] computing fresh dataset for {task}_{which_split}.")
            fts = tagging_example_to_feature(
                which_split, tagged_sents, tokenizer, self.raw_dataset.label2idx, max_seq_len,
            )
            if not do_cache:
                return _TaggingIter(fts)
            uid, complete = str(uuid.uuid4()), True
            try:
                with open(cached_, "wb") as f:
                    pickle.dump({"fts": fts, "uid": uid}, f)
            except:
                complete = False
            with open(meta_, "w") as f:
                json.dump({"complete": complete, "uid": uid}, f)
        return _TaggingIter(fts)
    
    @property
    def name(self):
        return self.raw_dataset.name
    
    @property
    def label_list(self):
        return self.raw_dataset.label_list
    
class _TaggingIter(torch.utils.data.Dataset):
    def __init__(self, fts):
        self.uides = [ft.uid for ft in fts]
        self.input_idses = torch.as_tensor(
            [ft.input_ids for ft in fts], dtype=torch.long
        )
        self.if_tgtes = torch.as_tensor(
            [ft.sent_if_tgt for ft in fts], dtype=torch.bool
        )
        self.attention_maskes = torch.as_tensor(
            [ft.attention_mask for ft in fts], dtype=torch.long
        )
        self.tags_ides = torch.as_tensor([ft.tags_ids for ft in fts], dtype=torch.long)
        
    def __len__(self):
        """ NOTE: size of the dataloader refers to number of sentences,
        rather than tags.
        """
        return self.input_idses.shape[0]
    
    def __getitem__(self, idx):
        return (
            self.uides[idx],
            self.input_idses[idx],
            self.attention_maskes[idx],
            self.tags_ides[idx],
            self.if_tgtes[idx],
        )
    
class SeqClsDataIter(object):
    def __init__(self, raw_dataset, model, tokenizer, max_seq_len, 
                 mislabel_type, mislabel_ratio, do_cache=True):
        self.raw_dataset = raw_dataset
        if raw_dataset.trn_egs is not None:
            self.trn_egs = self.wrap_iter(
                task=raw_dataset.name,
                model=model,
                which_split="trn",
                egs=raw_dataset.trn_egs,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                do_cache=do_cache,
                mislabel_type=mislabel_type,
                mislabel_ratio=mislabel_ratio
            )
        if raw_dataset.val_egs is not None:
            self.val_egs = self.wrap_iter(
                task=raw_dataset.name,
                model=model,
                which_split="val",
                egs=raw_dataset.val_egs,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                do_cache=do_cache,
                mislabel_type="None",
                mislabel_ratio=0.0
            )
        if raw_dataset.tst_egs is not None:
            self.tst_egs = self.wrap_iter(
                task=raw_dataset.name,
                model=model,
                which_split="tst",
                egs=raw_dataset.tst_egs,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                do_cache=do_cache,
                mislabel_type="None",
                mislabel_ratio=0.0
            )
            
    def wrap_iter(self, task, model, which_split, egs, tokenizer, max_seq_len, 
                  do_cache, mislabel_type, mislabel_ratio):
        len_egs = 0 if egs is None else len(egs)
        cached_ = os.path.join(
            "/input/jongwooko/xlt/data", # change this part
            "cached",
            f"{task},{max_seq_len},{model},{which_split},{len_egs},cached.pkl",
        )
        meta_ = cached_.replace(".pkl", ".metainfo")
        if os.path.exists(cached_) and do_cache:
            print("[INFO] loading cached dataset.")
            with open(meta_, "r") as f:
                meta = json.load(f)
            assert meta["complete"]
            with open(cached_, "rb") as f:
                fts = pickle.load(f)
            if fts["uid"] == meta["uid"]:
                fts = fts["fts"]
            else:
                raise ValueError("uids of data and meta do not match ...")
        else:
            print("[INFO] computing fresh dataset.")
            if egs is None or len(egs) == 0:
                fts = []
            else:
                fts = glue_example_to_feature(
                    task, egs, tokenizer, max_seq_len, self.label_list
                )
            if not do_cache:
                return _SeqClsIter(fts)
            uid, complete = str(uuid.uuid4()), True
            try:
                with open(cached_, "wb") as f:
                    pickle.dump({"fts": fts, "uid": uid}, f)
            except:
                complete = False
            with open(meta_, "w") as f:
                json.dump({"complete": complete, "uid": uid}, f)
        return _SeqClsIter(fts)

    @property
    def name(self):
        return self.raw_dataset.name

    @property
    def label_list(self):
        return self.raw_dataset.label_list

class _SeqClsIter(torch.utils.data.Dataset):
    def __init__(self, fts):
        
        fts1, fts2 = [], []
        for ft in fts:
            if 'english' in ft.uid:
                fts1.append(ft)
            else:
                fts2.append(ft)
        
        if len(fts1) == len(fts2):
            self.uides = [(ft1.uid, ft2.uid) for ft1, ft2 in zip(fts1, fts2)]
            self.input_idses = torch.as_tensor(
                [(ft1.input_ids, ft2.input_ids) for ft1, ft2 in zip(fts1, fts2)], dtype=torch.long
            )
            self.golds = torch.as_tensor([(ft1.gold, ft2.gold) for ft1, ft2 in zip(fts1, fts2)], dtype=torch.long)
            self.attention_maskes = torch.as_tensor(
                [(ft1.attention_mask, ft2.attention_mask) for ft1, ft2 in zip(fts1, fts2)], dtype=torch.long
            )
            self.token_type_idses = torch.as_tensor(
                [(ft1.token_type_ids, ft2.token_type_ids) for ft1, ft2 in zip(fts1, fts2)], dtype=torch.long
            )
            
        else:
            self.uides = [ft.uid for ft in fts]
            self.input_idses = torch.as_tensor(
                [ft.input_ids for ft in fts], dtype=torch.long
            )
            self.golds = torch.as_tensor([ft.gold for ft in fts], dtype=torch.long)
            self.attention_maskes = torch.as_tensor(
                [ft.attention_mask for ft in fts], dtype=torch.long
            )
            self.token_type_idses = torch.as_tensor(
                [ft.token_type_ids for ft in fts], dtype=torch.long
            )
        

    def __len__(self):
        return self.golds.shape[0]

    def __getitem__(self, idx):
        return (
            self.uides[idx],
            self.input_idses[idx],
            self.golds[idx],
            self.attention_maskes[idx],
            self.token_type_idses[idx],
        )