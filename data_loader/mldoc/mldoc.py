from ..common import (
    SentenceExample,
    MultilingualRawDataset,
    RawDataset,
)
import itertools
import os
from collections import OrderedDict
from ..data_configs import abbre2language, language2abbre


class MLDocDataset(MultilingualRawDataset):
    def __init__(self, conf):
        self.name = "mldoc"
        self.conf = conf
        self.lang_abbres = ["de", "en", "es", "fr", "it", "ru", "zh", "ja"]
        self.metrics = ["accuracy"]
        self.label_list = ["CCAT", "ECAT", "GCAT", "MCAT"]
        self.label2idx = {"CCAT": 0, "ECAT": 1, "GCAT": 2, "MCAT": 3}
        self.num_labels = 4
        self.num_trn_examples = 1000 # 1000, 2000, 5000, 10000
        self.contents = OrderedDict()
        self.create_contents()

    def get_labels(self):
        return self.label_list

    def get_language_data(self, language):
        return self.contents[language]

    def create_contents(self):
        mldoc_ = "./data/download/mldoc/"
#         mldoc_ = "/input/jongwooko/xlt/data/download/mldoc/"
        entries = []
        for abbr in self.lang_abbres:
            for which_split, wsplit in (
                ("train", "trn"),
                ("dev", "val"),
                ("test", "tst"),
            ):
                lang = abbre2language[abbr]
                if which_split == "train":
                    if self.conf.train_mt:
                        if lang == "english":
                            continue
                        which_split = f"english_{lang}.train.{self.num_trn_examples}"
                        file_ = os.path.join(mldoc_, which_split)
                        entries.extend(self.mt_parse(lang, file_, wsplit))
                    elif self.conf.train_bt:
                        if lang == "english":
                            continue
                        which_split = f"{lang}_english_{lang}.train.{self.num_trn_examples}"
                        file_ = os.path.join(mldoc_, which_split)
                        entries.extend(self.bt_parse(lang, file_, wsplit))
                    else:
                        which_split = f"{lang}.train.{self.num_trn_examples}"
                        file_ = os.path.join(mldoc_, which_split)
                        entries.extend(self.mldoc_parse(lang, file_, wsplit))
                if which_split == "dev":
                    which_split = f"{lang}.dev"
                    file_ = os.path.join(mldoc_, which_split)
                    entries.extend(self.mldoc_parse(lang, file_, wsplit))
                if which_split == "test":
                    if self.conf.test_mt and lang != "english":
                        which_split = f"{lang}_english.test"
                        file_ = os.path.join(mldoc_, which_split)
                        entries.extend(self.mt_parse(lang, file_, wsplit))
                    elif self.conf.test_bt:
                        which_split = f"{lang}_english_{lang}.test"
                        file_ = os.path.join(mldoc_, which_split)
                        entries.extend(self.bt_parse(lang, file_, wsplit))
                    else:
                        which_split = f"{lang}.test"
                        file_ = os.path.join(mldoc_, which_split)
                        entries.extend(self.mldoc_parse(lang, file_, wsplit))
                        
        entries = sorted(entries, key=lambda x: x[0])  # groupby requires contiguous
        for language, triplets in itertools.groupby(entries, key=lambda x: x[0]):
            # language, [(lang, split, eg)...]
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
                label2idx=self.label2idx,  # make all have the same mapping
            )
            _dataset.trn_egs = trn_egs if len(trn_egs) else None
            _dataset.val_egs = val_egs if len(val_egs) else None
            _dataset.tst_egs = tst_egs if len(tst_egs) else None

            self.contents[language] = _dataset

    def mldoc_parse(self, lang, input_file, which_split):
        sentence_egs = []
        lang = language2abbre[lang]
        language = abbre2language[lang]
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                
                portion_identifier = -1
                label = line[0].strip()
                text_a = line[1].strip()
                assert label in self.get_labels(), f"{label}, {input_file}"
                sentence_egs.append(
                    (
                        language,
                        which_split,
                        SentenceExample(
                            uid=f"{language}-{idx}-{which_split}",
                            text_a=text_a,
                            label=label,
                            portion_identifier=portion_identifier,
                        ),
                    )
                )
        return sentence_egs
    
    def mt_parse(self, lang, input_file, which_split):
        sentence_egs = []
        lang = language2abbre[lang]
        language = abbre2language[lang]
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                
                portion_identifier = -1
                label = line[0].strip()
                text1_a = line[1].strip()
                text2_a = line[2].strip()
                assert label in self.get_labels(), f"{label}, {input_file}"
                sentence_egs.append(
                    (
                        language,
                        which_split,
                        SentenceExample(
                            uid=f"english-{idx}-{which_split}",
                            text_a=text1_a,
                            label=label,
                            portion_identifier=portion_identifier,
                        ),
                    )
                )
                
                sentence_egs.append(
                    (
                        language,
                        which_split,
                        SentenceExample(
                            uid=f"{language}-{idx}-{which_split}",
                            text_a=text2_a,
                            label=label,
                            portion_identifier=portion_identifier,
                        ),
                    )
                )
        return sentence_egs
    
    def bt_parse(self, lang, input_file, which_split):
        sentence_egs = []
        lang = language2abbre[lang]
        language = abbre2language[lang]
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip().split("\t")               
#                 assert len(line)==4, f"{len(line)}, {idx}, {input_file}"
                
                portion_identifier = -1
                label = line[0].strip()
                try:
                    text_a = line[3].strip()
                except:
                    text_a = line[0].strip()
                assert label in self.get_labels(), f"{label}, {input_file}"
                sentence_egs.append(
                    (
                        language,
                        which_split,
                        SentenceExample(
                            uid=f"{language}-{idx}-{which_split}",
                            text_a=text_a,
                            label=label,
                            portion_identifier=portion_identifier,
                        ),
                    )
                )
        return sentence_egs