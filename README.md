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
1. Set conda env
```
cd data
bash install_tools.sh
```

2. Download datasets
- Download NER dataset manually following ./data/README.md

```
source activate fsxlt
conda install -c conda-forge transformers
bash scripts/download_data.sh
```

- [ ] udpos download (Error: 'DependencyTree' object has no attribute 'node')
- [ ] MLDocs, MARC download

## Related to the target datasets
- MLDoc  (Not provided. Refer to ./buckets/readme.md)
- MARC   (Refer to ./buckets/marc)
- XNLI   (Refer to ./buckets/xnli)
- PAWSX  (Refer to ./buckets/pawsx)
- POS    (Refer to ./buckets/udpos)
- NER    (Refer to ./buckets/panx)