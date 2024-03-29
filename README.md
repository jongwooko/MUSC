# [Official] Synergy with Translation Artifacts for Training and Inference in Multilingual Tasks
This repository contains code for the paper "Synergy with Translation Artifacts for Training and Inference in Multilingual Tasks" presented in EMNLP 2022.

## Reproducibility Checklist
- We used "bert-base-multilingual-cased". Vocab size is about 120,000 and the number of parameters is about 180M.
- We used GeForce RTX 3090. For training MUSC on XNLI (the largest time-consuming task), about 2 days are required.

## How to start
All steps start from the root directory.

1. Set conda env
```
cd data
bash install_tools.sh
```

2. Download datasets
- For MLDocs dataset, refer to https://github.com/facebookresearch/MLDoc
- For MARC dataset, refer to https://docs.opendata.aws/amazon-reviews-ml/readme.html
- For XTREME datasets (XNLI, PAWSX)

```
source activate fsxlt
conda install -c conda-forge transformers
pip install networkx==1.11

cd data
bash scripts/download_data.sh
```

3. MUSC (refer to exps folder)
```
source activate fsxlt
pip install -r requirements.txt
```

## Contact
- Jaehoon Oh: jhoon.oh@kaist.ac.kr
- Jongwoo Ko: jongwoo.ko@kaist.ac.kr

## References
- How Multilingual is Multilingual BERT?
- Cross-lingual Language Model Pretraining
- XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization (https://github.com/google-research/xtreme)
- A Closer Look at Few-Shot Crosslingual Transfer: The Choice of Shots Matters (https://github.com/fsxlt/code)
- https://github.com/fsxlt/buckets
