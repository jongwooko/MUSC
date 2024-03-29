# from .glue.datasets import MRPCDataset
from .mldoc import MLDocDataset
from .marc import MARCDataset
from .ner import PANXDataset # CONLL2003Dataset
# from .argus import ARGUStanceDataset
from .pawsx import PAWSXDataset
from .NLI import XNLIDataset
from .pos import UDPOSDataset
from .cls import CLSDataset

# from sampled_data_loader.mldoc.mldoc import SampledMLDocDataset

task2dataset = {
    "mldoc": MLDocDataset,
    "marc": MARCDataset,
#     "argustan": ARGUStanceDataset,
    "pawsx": PAWSXDataset,
    "xnli": XNLIDataset,
#     "conll2003": CONLL2003Dataset,
#     "mrpc": MRPCDataset,
    "panx": PANXDataset,
    "udpos": UDPOSDataset,
    "cls": CLSDataset,
}

task2labelsetsize = {
    "mrpc": 2,
    "mldoc": 4,
    "marc": 5,
    "cls": 2,
    "conll2003": -1,
    "panx": -1,
    "udpos": -1,
}