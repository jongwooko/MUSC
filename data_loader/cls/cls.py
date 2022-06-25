from ..common import (
    SentencePairExample,
    MultilingualRawDataset,
    RawDataset,
)
import itertools
import os
import json
from collections import OrderedDict, Counter
from ..data_configs import abbre2language

class CLSDataset(MultilingualRawDataset):
    def __init__(self, conf):
        self.name = "cls"
        self.conf = conf
        self.lang_abbres = ["de", "en", "fr", "ja"]
        self.metrics = ["accuracy"]
        self.label_list = ["pos", "neg"]
        self.label2idx = {"pos": 0, "neg": 1}
        self.num_labels = 2
        self.domain = self.conf.domain # "books" "dvd" "music"
        self.contents = OrderedDict()
        self.create_contents()
        
    def get_labels(self):
        return self.label_list
    
    def get_language_data(self, language):
        return self.contents[language]
    
    def create_contents(self):
        cls_ = "/data/FSXLT_dataset/data/download/cls/"
        entries = []
        for lang_abbre in self.lang_abbres:
            for which_split, wsplit in (
                ("train", "trn"),
                ("test", "tst")
            ):
#                 file_ = os.path.join(cls_, f"{self.domain}_{lang_abbre}_{which_split}.tsv")
                if which_split == "test":
                    if self.conf.test_mt and lang_abbre != "en":
                        file_ = os.path.join(cls_, f"{self.domain}_{lang_abbre}_en_{which_split}.tsv")
                        which_split = "tst"
                        entries.extend(self.mt_parse(file_, which_split, lang_abbre))
                    # elif self.conf.test_bt and lang_abbre != "en":
                    #     # back_translate
                    #     file_ = os.path.join(cls_, f"{self.domain}_{lang_abbre}_en_{lang_abbre}_{which_split}.tsv")
                    #     which_split = "tst"
                    #     entries.extend(self.bt_parse(file_, which_split, lang_abbre))
                    else:
                        # original
                        file_ = os.path.join(cls_, f"{self.domain}_{lang_abbre}_{which_split}.tsv")
                        which_split = "tst"
                        entries.extend(self.cls_parse(file_, which_split, lang_abbre))
                elif which_split == "train":
                    if self.conf.train_mt and lang_abbre == "en":
                        continue
                    elif self.conf.train_mt and lang_abbre != "en":
                        file_ = os.path.join(cls_, f"{self.domain}_en_{lang_abbre}_{which_split}.tsv")
                        which_split = "trn"
                        entries.extend(self.mt_parse(file_, which_split, lang_abbre))
                    elif self.conf.train_bt and lang_abbre != "en":
                        file_ = os.path.join(cls_, f"{self.domain}_{lang_abbre}_en_{lang_abbre}_{which_split}.tsv")
                        which_split = "trn"
                        entries.extend(self.bt_parse(file_, which_split, lang_abbre))
                    else:
                        file_ = os.path.join(cls_, f"{self.domain}_{lang_abbre}_{which_split}.tsv")
                        which_split = "trn"
                        entries.extend(self.cls_parse(file_, which_split, lang_abbre))
                else:
                    raise ValueError
        
        entries = sorted(entries, key=lambda x: x[0]) # groupby requires contiguous
        for language, triplets in itertools.groupby(entries, key=lambda x: x[0]):
            # get examples in this language
            triplets = list(triplets)
            trn_egs, val_egs, tst_egs = [], [], []
            for _language, split, eg in triplets:
                assert language == _language
                if split == "trn":
                    trn_egs.append(eg)
                elif split == "val":
                    val_egs.append(eg)
                elif split == "tst":
                    tst_egs.append(eg)
                else:
                    raise ValueError
            _dataset = RawDataset(
                name=f"{self.name}-{language}",
                language=language,
                metrics=self.metrics,
                label_list=self.label_list,
                label2idx=self.label2idx,
            )
            _dataset.trn_egs = trn_egs if len(trn_egs) else None
            _dataset.val_egs = tst_egs if len(tst_egs) else None
            _dataset.tst_egs = tst_egs if len(tst_egs) else None
            self.contents[language] = _dataset
        
    def cls_parse(self, input_file, which_split, lang_abbre):
        import pandas as pd
        sentence_pair_egs = []
        language = abbre2language[lang_abbre]
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                try:
                    assert len(line) == 3, f"{len(line)}, {input_file}, {idx}, {line}"
                    text_a, text_b, label = line[0], line[1], line[2]
                except:
                    assert len(line) == 2, f"{len(line)}, {input_file}, {idx}, {line}"
                    text_b, label = line[0], line[1]
                assert label in self.get_labels(), f"{label}, {input_file}, {type(label)}"
                sentence_pair_egs.append(
                    (
                        language,
                        which_split,
                        SentencePairExample(
                            uid=f"{language}-{idx}-{which_split}",
                            text_a=text_b,
                            label=label,
                        ),
                    )
                )
        print(len(sentence_pair_egs), input_file)
        return sentence_pair_egs
    
    def mt_parse(self, input_file, which_split, lang_abbre):
        import pandas as pd
        sentence_pair_egs = []
        language = abbre2language[lang_abbre]
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                
                if len(line) != 5:
                    continue
                assert len(line) == 5, f"{len(line)}, {input_file}, {idx}, {line}"
                
                if which_split == "trn":
                    text1_a, text1_b = line[0], line[1]
                    text2_a, text2_b, label = line[3], line[4], line[2]
                elif which_split == "tst":
                    text1_a, text1_b = line[3], line[4]
                    text2_a, text2_b, label = line[0], line[1], line[2]
                assert label in self.get_labels(), f"{label}, {input_file}, {type(label)}"
                
                sentence_pair_egs.append(
                    (
                        language,
                        which_split,
                        SentencePairExample(
                            uid=f"english-{idx}-{which_split}",
                            text_a=text1_b,
                            label=label,
                        ),
                    )
                )
                
                sentence_pair_egs.append(
                    (
                        language,
                        which_split,
                        SentencePairExample(
                            uid=f"{language}-{idx}-{which_split}",
                            text_a=text2_b,
                            label=label,
                        ),
                    )
                )
        print (len(sentence_pair_egs), input_file)
        return sentence_pair_egs
    
    def bt_parse(self, input_file, which_split, lang_abbre):
        import pandas as pd
        sentence_pair_egs = []
        language = abbre2language[lang_abbre]
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                try:
                    assert len(line) == 7, f"{len(line)}, {input_file}, {idx}, {line}"
                    text_a, text_b, label = line[5], line[6], line[2]
                except:
                    text_a, text_b, label = line[0], line[1], line[2]
                assert label in self.get_labels(), f"{label}, {input_file}, {type(label)}"
                sentence_pair_egs.append(
                    (
                        language,
                        which_split,
                        SentencePairExample(
                            uid=f"{language}-{idx}-{which_split}",
                            text_a=text_b,
                            label=label,
                        ),
                    )
                )
        print (len(sentence_pair_egs), input_file)
        return sentence_pair_egs
                