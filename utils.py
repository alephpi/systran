import re
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

def calc_mask_marker(target_vocab: Dict[str, int], penalty_mask_path='./penalty_mask.pt', leading_marker_path='./leading_marker.pt'):
    if os.path.exists(leading_marker_path):
        leading_marker = torch.load(leading_marker_path)
    else:
        leading_ids = []
        for token in tqdm(list(target_vocab.keys())):
            if ('￭' in token) & ('￭' not in token[:-1]): 
                leading_ids.append(target_vocab[token])
        leading_marker = torch.zeros(len(target_vocab))
        leading_marker[leading_ids] = 1
        torch.save(leading_marker, leading_marker_path)
    if os.path.exists(penalty_mask_path):
        penalty_mask = torch.load(penalty_mask_path)
    else:
        IS_CLOSE = ['ADP', 'AUX', 'CCONJ', 'DET',
                    'NUM', 'PART', 'PRON', 'SCONJ']
        IS_OTHER = ['PUNCT', 'SYM', 'X']
        IGNORE_TAG = IS_CLOSE + IS_OTHER
        # the first four special tokens are ignorable
        ignore_ids = [0, 1, 2, 3]
        # remember to generalize for other target langs.
        nlp = spacy.load('fr_core_news_lg')
        for token in tqdm(list(target_vocab.keys())[4:]):
            if '￭' in token:
                # subtokens with joiner may be punctuation or a word piece, we ignore the punctuation
                # if is_letter_free(token.strip('￭')):
                #     ignore_ids.append(target_vocab[token])
                # ignore all joiner form including leading form
                ignore_ids.append(target_vocab[token])
                # if it is a leading form, i.e. joiner appears at right
            else:
                pos = nlp(token)[0].pos_
                if pos in IGNORE_TAG:
                    ignore_ids.append(target_vocab[token])
        penalty_mask = torch.ones(len(target_vocab))
        penalty_mask[ignore_ids] = 0
        torch.save(penalty_mask, penalty_mask_path)
    return penalty_mask, leading_marker

def is_letter_free(text):
    pattern = r'^[^\w\s]*$'
    return re.match(pattern, text) is not None
