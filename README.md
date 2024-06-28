## Unified Training for Cross-Lingual Abstractive Summarization by Aligning Parallel Machine Translation Pairs

This repository contains the code for our paper [Unified Training for Cross-Lingual Abstractive Summarization by Aligning Parallel Machine Translation Pairs]. (The paper will be released after publication.)

## Environment
The used python version is 3.10.
```markdown
transformers==4.26.1
torch==1.13.1
```

## Datasets

We do experiments on two benchmark CLS datasets Zh2EnSum and En2ZhSum, which can be downloaded in [this repo](https://github.com/ZNLP/NCLS-Corpora).

| Dataset |   Train   | Validation | Test |
|:--------|:---------:|:----------:|:-----|
|  Zh2EnSum | 1,693,713 |   3,000    | 3,000  |
| En2ZhSum |  364,687  |    3,000     | 3,000  |

## Parameter
We give a config example: config.yaml.

## Quick start

```markdown
python main.py --config config.yaml
```