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
        self.lang_abbres = ["de", "en", "es", "fr", "ja", "ko", "zh"]
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
        pawsx_ = "/input/jongwooko/xlt/data/download/pawsx/"
        entries = []
        for lang in self.lang_abbres:
            for which_split, wsplit in (
                ("train", "trn"),
                ("dev", "val"),
                ("test", "tst"),
            ):
                file_ = os.path.join(pawsx_, f"{which_split}-{lang}.tsv")
                if not os.path.exists(file_):
                    print(f"[INFO]: skip {lang} {wsplit}: not such file")
                    continue
                    
                if lang == "en" and wsplit == "trn":
                    sentence_egs = self.pawsx_parse(lang, file_, wsplit)
                    sentence_egs = self.gen_mislabeled_data(sentence_egs, self.conf.mislabel_type, self.conf.mislabel_ratio)    
                    entries.extend(sentence_egs)
                else:
                    entries.extend(self.pawsx_parse(lang, file_, wsplit))
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
    
    def pawsx_parse(self, lang, input_file, which_split):
        sentence_egs = []
        language = abbre2language[lang]
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                if len(line) == 3:
                    text_a, text_b, label = line[0], line[1], line[2]
                    assert label in self.get_labels(), f"{label}, {input_file}"
                # elif len(line) == 2:
                #     text_a, text_b, label = line[0], line[1], None
                else:
                    print (len(line))
                    print (lang)
                    print (line)
                    print (prev_line)
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
                prev_line = line
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