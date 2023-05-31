import logging
import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import spacy
import os
from typing import Dict

from data import load_vocabulary
_, tgt_vocab_rev = load_vocabulary('../corpus/vocab_en_fr.txt')


def init_logger(level=None):
    if level is None:
        level = "INFO"

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d000 UTC [%(module)s@%(processName)s] %(levelname)s %(message)s",
        datefmt="%Y-%b-%d %H:%M:%S",
        level=os.getenv("LOG_LEVEL", level),
    )


def get_logger():
    return logging.getLogger("pytorch_transformer")


def probe(probs: torch.Tensor):
    v, ids = torch.topk(probs, k=10)
    print([tgt_vocab_rev[i] for i in ids])
    plt.bar(range(v.shape[0]), v)
    plt.show()


def vis(tgt_ids: torch.Tensor, score: torch.Tensor):
    for i, sent in enumerate(tgt_ids.view(-1, tgt_ids.shape[-1]).tolist()):
        print(f'{score.view(-1)[i].item():.4f}',
              ' '.join([tgt_vocab_rev[i] for i in sent]))


def diff(t1: torch.Tensor, t2: torch.Tensor):
    return set(tuple(idx.tolist()) for idx in t1.nonzero()).difference(set(tuple(idx.tolist()) for idx in t2.nonzero()))

def calc_mask(target_vocab: Dict[str, int], path='./penalty_mask.pt'):
    if os.path.exists(path):
        mask = torch.load(path)
    else:
        IS_CLOSE = ['ADP', 'AUX', 'CCONJ', 'DET',
                    'NUM', 'PART', 'PRON', 'SCONJ']
        IS_OTHER = ['PUNCT', 'SYM', 'X']
        IGNORE_TAG = IS_CLOSE + IS_OTHER
        print(IGNORE_TAG)
        # the first four special tokens are ignorable
        ignore_ids = [0, 1, 2, 3]
        # remember to generalize for other target langs.
        nlp = spacy.load('fr_core_news_lg')
        for token in tqdm(list(target_vocab.keys())[4:]):
            pos = nlp(token.strip('￭'))[0].pos_
            if pos in IGNORE_TAG:
                ignore_ids.append(target_vocab[token])
        mask = torch.ones(len(target_vocab))
        mask[ignore_ids] = 0
        torch.save(mask, path)
    return mask
