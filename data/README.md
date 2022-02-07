# XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization

# Download the data

In order to run experiments on XTREME, the first step is to download the dependencies. We assume you have installed [`anaconda`](https://www.anaconda.com/) and use Python 3.7+. The additional requirements including `transformers`, `seqeval` (for sequence labelling evaluation), `tensorboardx`, `jieba`, `kytea`, and `pythainlp` (for text segmentation in Chinese, Japanese, and Thai), and `sacremoses` can be installed by running the following script:
```
bash install_tools.sh
```

The next step is to download the data. To this end, first create a `download` folder with ```mkdir -p download``` in the root of this project. You then need to manually download `panx_dataset` (for NER) from [here](https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN) (note that it will download as `AmazonPhotos.zip`) to the `download` directory. Finally, run the following command to download the remaining datasets:
```
bash scripts/download_data.sh
```

Note that in order to prevent accidental evaluation on the test sets while running experiments,
we remove labels of the test data during pre-processing and change the order of the test sentences
for cross-lingual sentence retrieval.