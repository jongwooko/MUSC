from ..common import (
    SentencePairExample,
    MultilingualRawDataset,
    RawDataset,
)
import itertools
import os
import json
import numpy as np
from collections import OrderedDict
from ..data_configs import abbre2language

class PAWSXDataset(MultilingualRawDataset):
    def __init__(self, conf):
        self.name = "pawsx"
        self.conf = conf
        self.lang_abbres = ["en", "de", "fr", "es", "ko", "zh", "ja"]
        self.metrics = ["accuracy"]
        self.label_list = ["0", "1"]
        self.label2idx = {"0": 0, "1": 1}
        self.num_labels = 2
        self.contents = OrderedDict()
        self.create_contents()

    def get_labels(self):
        return self.label_list

    def get_language_data(self, language):
        return self.contents[language]

    def create_contents(self):
        pawsx_ = "/data/FSXLT_dataset/data/download/pawsx/"
        entries = []
        for lang in self.lang_abbres:
            for which_split in ("train", "dev", "test"):
                file_ = os.path.join(pawsx_, f"{which_split}-{lang}.tsv")
                if not os.path.exists(file_):
                    print(f"[INFO]: skip {lang} {which_split}: not such file")
                    continue
                
                if which_split == "dev":
                    which_split = "val"
                    entries.extend(self.pawsx_parse(file_, which_split, lang))
                elif which_split == "test":
                    if lang == "en":
                        which_split = "tst"
                        entries.extend(self.pawsx_parse(file_, which_split, lang))
                    elif self.conf.trans_test:
                        which_split = "tst"
                        entries.extend(self.trans_parse(file_, which_split, lang))
                    else:
                        which_split = "tst"
                        entries.extend(self.pawsx_parse(file_, which_split, lang))
                elif which_split == "train":
                    if lang == "en" and self.conf.trans_train:
                        continue
                    elif self.conf.trans_train:
                        which_split = "trn"
                        entries.extend(self.trans_parse(file_, which_split, lang))
                    elif lang == "en":
                        which_split = "trn"
                        entries.extend(self.pawsx_parse(file_, which_split, lang))
                    else:
                        which_split = "trn"
                        entries.extend(self.pawsx_parse(file_, which_split, lang))
                else:
                    raise ValueError

        entries = sorted(entries, key=lambda x: x[0])  # groupby requires contiguous
        for language, triplets in itertools.groupby(entries, key=lambda x: x[0]):
            # get examples in this language
            triplets = list(triplets)
            trn_egs, val_egs, tst_egs = [], [], []
            for _, split, eg in triplets:
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
            _dataset.val_egs = val_egs if len(val_egs) else None
            _dataset.tst_egs = tst_egs if len(tst_egs) else None

            self.contents[language] = _dataset
    
    def pawsx_parse(self, input_file, which_split, lang):
        sentence_egs = []
        language = abbre2language[lang]
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                if len(line) == 3:
                    text_a, text_b, label = line[0], line[1], line[2]
                    assert label in self.get_labels(), f"{label}, {input_file}"
                elif len(line) == 5 and which_split == "trn":
                    text_a, text_b, label = line[2], line[3], line[4]
                elif len(line) == 5 and which_split == "tst":
                    text_a, text_b, label = line[0], line[1], line[4]
                # elif len(line) == 2:
                #     text_a, text_b, label = line[0], line[1], None
                else:
                    raise ValueError
                portion_identifier = -1
                sentence_egs.append(
                    (
                        language,
                        which_split,
                        SentencePairExample(
                            uid=f"{language}-{idx}-{which_split}",
                            text_a=text_a,
                            text_b=text_b,
                            label=label,
                            portion_identifier=portion_identifier,
                        ),
                    )
                )
        print(input_file, len(sentence_egs))
        return sentence_egs
    
    def trans_parse(self, input_file, which_split, lang):
        sentence_egs = []
        language = abbre2language[lang]

        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                if len(line) == 5 and which_split == "trn":
                    text1_a, text1_b = line[0], line[1]
                    text2_a, text2_b, label = line[2], line[3], line[4]
                    assert label in self.get_labels(), f"{label}, {input_file}"
                elif len(line) == 5 and which_split == "tst":
                    text1_a, text1_b = line[2], line[3]
                    text2_a, text2_b, label = line[0], line[1], line[4]
                    assert label in self.get_labels(), f"{label}, {input_file}"
                # elif len(line) == 2:
                #     text_a, text_b, label = line[0], line[1], None
                else:
                    print (len(line))
                    print (idx)
                    for i in range(len(line)):
                        print (line[i])
                    raise ValueError
                portion_identifier = -1
                sentence_egs.append(
                    (
                        language,
                        which_split,
                        SentencePairExample(
                            uid=f"english-{idx}-{which_split}",
                            text_a=text1_a,
                            text_b=text1_b,
                            label=label,
                            portion_identifier=portion_identifier,
                        ),
                    )
                )

                sentence_egs.append(
                    (
                        language,
                        which_split,
                        SentencePairExample(
                            uid=f"{language}-{idx}-{which_split}",
                            text_a=text2_a,
                            text_b=text2_b,
                            label=label,
                            portion_identifier=portion_identifier,
                        ),
                    )
                )
        print(input_file, len(sentence_egs))
        return sentence_egs

    def gen_imbalanced_data(self, sentence_egs, seq_num_per_cls):
        """
        Gen a list of imbalanced training data, and replace the origin with generated ones.
        """
        import random
        import copy
        new_sentence_egs = []
        _sentence_egs = copy.deepcopy(sentence_egs)
        _seq_num_per_cls = [0 for _ in range(self.num_labels)]
        
        random.shuffle(_sentence_egs)
        for (lang, split, data) in _sentence_egs:
            if _seq_num_per_cls[data.label] < seq_num_per_cls[data.label]:
                new_sentence_egs.append(
                    (
                        lang,
                        split,
                        SentencePairExample(
                            uid=data.uid,
                            text_a=data.text_a,
                            text_b=data.text_b,
                            label=data.label
                        )
                    )
                )
        print(len(new_sentence_egs))
        return new_sentence_egs
            
    def gen_mislabeled_data(self, sentence_egs, mislabel_type, mislabel_ratio):
        """
        Gen a list of mislabeled training data, and replace the origin with generated ones.
        """
        new_sentence_egs = []
        if mislabel_type == "uniform":
            for (lang, split, data) in sentence_egs:
                new_label = data.label
                if np.random.rand() < mislabel_ratio:
                    while new_label == data.label:
                        new_label = str(np.random.randint(self.num_labels))
                new_sentence_egs.append(
                    (
                        lang,
                        split,
                        SentencePairExample(
                            uid=data.uid,
                            text_a=data.text_a,
                            text_b=data.text_b,
                            label=new_label,
                        ),
                    )
                )
            return new_sentence_egs
        
        elif mislabeled_type == "model":
            pass
        
        else:
            raise NotImplementedError