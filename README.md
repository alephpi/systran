# Transformer training with PyTorch

This repository contains an example of Transformer training with PyTorch. While the code is quite minimal, the training is faster than [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf) and models can reach a similar accuracy.

The code implements:

* pre-norm Transformer
* gradient accumulation
* mixed precision training
* multi-GPU training
* checkpoint averaging

The default parameters are mostly copied from the [Scaling NMT](https://aclanthology.org/W18-6301/) paper.

Note that the repository does not contain inference code in PyTorch, but the models can be converted to [CTranslate2](https://github.com/OpenNMT/CTranslate2).

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

### Convert to CTranslate2

```bash
python3 convert.py averaged_checkpoint.pt --src_vocab vocab.en --tgt_vocab vocab.de --output_dir ct2_model
```

### Compute the validation loss

```bash
python3 eval.py ct2_model valid.en.tok valid.de.tok
```

### Translate

```bash
python3 translate.py ct2_model < input.txt.tok > output.txt.tok
```

To run translation on the GPU (or multiple GPUs), use `--num_gpus N`.
