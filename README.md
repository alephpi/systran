# Reduction of word repetition
The work is done during a 5-month internship at [Systran](https://www.systransoft.com/) under the supervision of [Josep Crego](https://scholar.google.com/citations?user=lw_aQqQAAAAJ)

The training corpus is omitted, you can download the corpus on your own.

First train a tokenizer over the training corpus, e.g. we've done this by a BPE tokenizer from OpenNMT-Tokenizer(pyonmttok)

1. baseline model: run `run.sh` and then `infer.sh`, we get a trained baseline model and baseline translation.
2. inference with penalization, use the same checkpoint of baseline model and run `infer_rep_penalty_only.sh`, we get a penalized translation with reduced word repetition
3. training with penalization, run `run_rep_penalty.sh` and then `infer_rep_penalty.sh`, we get a model trained with penalization and inference normally(without penalization like 2), we also get a penalized translation with reduced word repetition.
4. evaluation: compute BLEU score with [sacrebleu](https://github.com/mjpost/sacrebleu) and word repetition times with `wordrep` in this repo

The training and inference is preferred to run on a GPU. 

## ENV configuration

We recommend you use `conda/mamba` to manage the environment, e.g. run
```bash
micromamba env create -f mini-trans.yml
micromamba activate mini-trans
pip install -r requirements.txt
``` 

Implementation on top of the template written by [Guillaume Klein](https://github.com/guillaumekln)
Below are the original readme file.
# Transformer training with PyTorch

This repository contains an example of Transformer training with PyTorch. While the code is quite minimal, the training is faster than [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf) and models can reach a similar accuracy.

The code implements:

* pre-norm Transformer
* gradient accumulation
* mixed precision training
* multi-GPU training
* checkpoint averaging
* beam search decoding

The default parameters are mostly copied from the [Scaling NMT](https://aclanthology.org/W18-6301/) paper.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train

```bash
python3 train.py --src train.en.tok --tgt train.de.tok --src_vocab vocab.en --tgt_vocab vocab.de --save_dir checkpoints/
```

For multi-GPU training, use `--num_gpus N`.

Vocabularies that work with OpenNMT-tf also work here. If you are building your own vocabulary, make sure that it meets the following requirements:

* must have one token per line (no token frequencies or other annotations)
* must start with a padding token (use `<blank>` or `<pad>`)
* must contain the tokens `<s>` and `</s>`
* may contain the token `<unk>` (if not present, the token is automatically added in the training)

### Average checkpoints

```bash
python3 average.py checkpoints/ --output averaged_checkpoint.pt
```

### Run inference

```bash
python3 beam_search.py --ckpt averaged_checkpoint.pt --src_vocab vocab.en --tgt_vocab vocab.de < test.en.tok
```
