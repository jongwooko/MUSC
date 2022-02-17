# FSXLT

## References
- How Multilingual is Multilingual BERT?
- Cross-lingual Language Model Pretraining
- XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization (https://github.com/google-research/xtreme)
- A Closer Look at Few-Shot Crosslingual Transfer: The Choice of Shots Matters (https://github.com/fsxlt/code)
- https://github.com/fsxlt/buckets

## TODO
- [ ] Solving Errors for test dataset do not have tagging.
- [ ] Adapt Training to target languages.
- [ ] Code works on all datasets. (PAWSX, XNLI)

## How to start (TODO: Jaehoon)
All steps start from the root directory (i.e., FSXLT folder).

1. Set conda env
```
cd data
bash install_tools.sh
```

2. Download datasets
- For MLDocs dataset, refer to https://github.com/facebookresearch/MLDoc
- For MARC dataset, refer to https://docs.opendata.aws/amazon-reviews-ml/readme.html
- For XTREME datasets (XNLI, PAWSX, POS, NER), download NER dataset manually following ./data/README.md in advance.

```
source activate fsxlt
conda install -c conda-forge transformers
pip install networkx==1.11

cd data
bash scripts/download_data.sh
```

3. Source Fine-tuning (Zero-Shot Cross-Lingual Transfer, ZS-XLT)
```
pip install -r requirements.txt
```

- [ ] for MLDoc dataset, `bash exps/mldoc_source_ft.sh`
- [ ] for MARC dataset,  `bash exps/marc_source_ft.sh`
- for XNLI dataset,  `bash exps/xnli_source_ft.sh`
- for PAWSX dataset, `bash exps/pawsx_source_ft.sh`
- for POS dataset,   `bash exps/pos_source_ft.sh` (Check data_loader/pos/udpos.py line 128)
- for NER dataset,   `bash exps/ner_source_ft.sh` (Check data_loader/ner/panx.py line 139)

4. Target Fine-tuning (Few-Shot Cross-Lingual Transfer, FS-XLT)

5. Test

## Related to the target datasets
- MLDoc  (Not provided. Refer to ./buckets/readme.md)
- MARC   (Refer to ./buckets/marc)
- XNLI   (Refer to ./buckets/xnli)
- PAWSX  (Refer to ./buckets/pawsx)
- POS    (Refer to ./buckets/udpos)
- NER    (Refer to ./buckets/panx)