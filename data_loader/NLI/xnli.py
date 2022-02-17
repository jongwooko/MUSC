from multiprocessing.sharedctypes import Value
from ..common import (
    SentencePairExample,
    MultilingualRawDataset,
    RawDataset,
)
import itertools
import os
import json
from collections import OrderedDict
from ..data_configs import abbre2language


class XNLIDataset(MultilingualRawDataset):
    def __init__(self):
        self.name = "xnli"
        self.lang_abbres = [
            "ar",
            "bg",
            "de",
            "el",
            "en",
            "es",
            "fr",
            "hi",
            "ru",
            "sw",
            "th",
            "tr",
            "ur",
            "vi",
            "zh",
        ]
        self.metrics = ["accuracy"]
        self.label_list = ["contradiction", "entailment", "neutral"]
        self.label2idx = {"contradiction": 0, "entailment": 1, "neutral": 2}
        self.num_labels = 3
        self.contents = OrderedDict()
        self.create_contents()

    def get_labels(self):
        return self.label_list

    def get_language_data(self, language):
        return self.contents[language]

    def create_contents(self):
        # for mnli, we only use train (no dev)
        mnli_ = "./data/download/xnli/"
        entries = []
        for file_ in ["train-en.tsv"]:
            file_ = os.path.join(mnli_, file_)
            entries.extend(self.mnli_parse(file_, "trn"))

        xnli_ = "./data/download/xnli/"
        for lang_abbre in self.lang_abbres:
            for which_split in ("dev", "test"):
                file_ = os.path.join(xnli_, f"{which_split}-{lang_abbre}.tsv")
                if which_split == "dev":
                    which_split = "val"
                elif which_split == "test":
                    which_split = "tst"
                else:
                    raise ValueError
                entries.extend(self.xnli_parse(file_, which_split, lang_abbre))

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

    def mnli_parse(self, input_file, which_split):
        sentence_pair_egs = []
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                assert len(line) == 3
                text_a, text_b, label = line[0], line[1], line[2]
                assert label in self.get_labels(), f"{label}, {input_file}"
                sentence_pair_egs.append(
                    (
                        "english",
                        which_split,
                        SentencePairExample(
                            uid=f"english-{idx}-{which_split}",
                            text_a=text_a,
                            text_b=text_b,
                            label=label,
                        ),
                    )
                )
        assert len(sentence_pair_egs) == 392702, f"{len(sentence_pair_egs)}"
        return sentence_pair_egs

    def xnli_parse(self, input_file, which_split, lang_abbre):
        sentence_pair_egs = []
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                if len(line) == 3:
                    text_a, text_b, label = line[0], line[1], line[2]
                    assert label in self.get_labels(), f"{label}, {input_file}"
                elif len(line) == 2:
                    text_a, text_b, label = line[0], line[1], None
                else:
                    raise ValueError
                sentence_pair_egs.append(
                    (
                        abbre2language[lang_abbre],
                        which_split,
                        SentencePairExample(
                            uid=f"{abbre2language[lang_abbre]}-{idx}-{which_split}",
                            text_a=text_a,
                            text_b=text_b,
                            label=label,
                        ),
                    ),
                )
        print(len(sentence_pair_egs), input_file)
        return sentence_pair_egs