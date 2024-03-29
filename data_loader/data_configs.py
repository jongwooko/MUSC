from .data_iters import SeqClsDataIter, TaggingDataIter

task2dataiter = {
    "mrpc": SeqClsDataIter,
    "sst2": SeqClsDataIter,
    "mldoc": SeqClsDataIter,
    "marc": SeqClsDataIter,
    "argustan": SeqClsDataIter,
    "pawsx": SeqClsDataIter,
    "xnli": SeqClsDataIter,
    "posptb": TaggingDataIter,
    "conll2003": TaggingDataIter,
    "panx": TaggingDataIter,
    "udpos": TaggingDataIter,
    "cls": SeqClsDataIter,
}

task2datadir = {
    "mrpc": "data/glue_data/data/MRPC",
    "sst2": "data/glue_data/data/SST-2",
    "mnlimiss": "data/xnli/MNLI",
    "mnlimatched": "data/xnli/MNLI",
    "conll2003": "data/ner/conll2003",
    "argustan": "data/arguments/",
    "pawsx": "data/pawsx/",
    "xnli": "data/NLI/",
    "panx": "data/panx/",
    "udpos": "data/udpos/",
    "cls": "data/cls"
}

LANGUAGE2ID = {
    "english": 0,
    "german": 1,
    "french": 2,
    "chinese": 3,
    "spanish": 4,
    "italian": 5,
    "japanese": 6,
    "russian": 7,
    "dutch": 8,
    "korean": 9,
}

TASK2ID = {
    "sst2": 0,
    "germeval": 1,
    "rte": 2,
    "mrpc": 3,
    "xnli": 4,
    "mldoc": 5,
    "conll2003": 6,
    "pawsx": 7,
    "panx": 8,
    "udpos": 9,
    "cls": 10,
}

SPLIT2ID = {"trn": 0, "val": 1, "tst": 2}

abbre2language = {
    "en": "english",
    "af": "afrikaans",
    "ar": "arabic",
    "bg": "bulgarian",
    "bn": "bengali",
    "de": "german",
    "el": "greek",
    "es": "spanish",
    "et": "estonian",
    "eu": "basque",
    "fa": "persian",
    "fi": "finnish",
    "fr": "french",
    "he": "hebrew",
    "hi": "hindi",
    "hu": "hungarian",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "jv": "javanese",
    "ka": "georgian",
    "kk": "kazakh",
    "ko": "korean",
    "ml": "malayalam",
    "mr": "marathi",
    "ms": "malay",
    "my": "burmese",
    "nl": "dutch",
    "pt": "portuguese",
    "ru": "russian",
    "sw": "swahili",
    "ta": "tamil",
    "te": "telugu",
    "th": "thai",
    "tl": "tagalog",
    "tr": "turkish",
    "ur": "urdu",
    "vi": "vietnamese",
    "yo": "yoruba",
    "zh": "chinese",
}

language2abbre = {v: k for k, v in abbre2language.items()}