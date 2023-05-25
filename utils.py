import logging
import os

import matplotlib.pyplot as plt
import torch
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
        print(f'{score.view(-1)[i].item():.4f}', ' '.join([tgt_vocab_rev[i] for i in sent]))
