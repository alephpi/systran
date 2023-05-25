import argparse
import sys
from typing import Dict

import torch

from data import encode_line, load_vocabulary
from dataset import TextFileDataset, MapDataset, BatchDataset, to_tensor
from model import Transformer

batch_size = 16
beam_size = 5
length_penalty = 1
max_length = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_vocab", required=True, help="Path to the source vocabulary"
    )
    parser.add_argument(
        "--tgt_vocab", required=True, help="Path to the target vocabulary"
    )
    parser.add_argument("--ckpt", required=True, help="Path to the checkpoint")
    parser.add_argument("--device", default="cpu", help="Device to use")
    args = parser.parse_args()

    source_vocabulary, _ = load_vocabulary(args.src_vocab)
    target_vocabulary, target_vocabulary_rev = load_vocabulary(args.tgt_vocab)

    bos = target_vocabulary["<s>"]
    eos = target_vocabulary["</s>"]

    model = Transformer(
        len(source_vocabulary),
        len(target_vocabulary),
        share_embeddings=True,
    )

    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()

    dataset = create_dataset(sys.stdin, source_vocabulary, args.device)

    with torch.no_grad():
        for batch in dataset:
            result = beam_search(model, batch, bos, eos)

            # result = (batch_size, ), hypo = (score, sent)
            for hypotheses in result:
                tokens = hypotheses[0][1]
                if tokens and tokens[-1] == eos:
                    tokens.pop(-1)
                tokens = [target_vocabulary_rev[token_id]
                          for token_id in tokens]
                print(" ".join(tokens), flush=True)


def create_dataset(path, source_vocabulary, device):
    dataset = TextFileDataset(path)
    dataset = MapDataset(
        dataset, lambda line: encode_line(
            line, source_vocabulary, add_eos=True)
    )
    dataset = BatchDataset(dataset, batch_size)
    dataset = MapDataset(
        dataset, lambda batch: to_tensor(batch, device=device))
    return dataset


def beam_search(model: Transformer, src_ids: torch.Tensor, bos, eos, rep_penalty=False):
    print("my beam search")
    batch_size = src_ids.shape[0]
    # encoder_output.shape = (batch, sent, embedding)
    # why src_mask has 4 dimension? (batch, 1, 1, sent)
    encoder_output, src_mask = model.encode(src_ids)
    # encoder_output = (batch*beam, sent, embedding)
    encoder_output = repeat_beam(encoder_output, beam_size)
    # encoder_output = (batch*beam, 1, 1, embedding)
    src_mask = repeat_beam(src_mask, beam_size)

    tgt_ids = torch.full(
        (batch_size, beam_size, 1), bos, dtype=src_ids.dtype, device=src_ids.device
    )  # initialize tgt sequence, begin with <bos>

    cum_log_probs = torch.full(
        (batch_size, beam_size), float("-inf"), device=src_ids.device
    )  # hold log prob for each candidates in each batch
    cum_log_probs[:, 0] = 0

    finished = [False for _ in range(batch_size)]  # flag if search is finished
    finished_hypotheses = [[]
                           for _ in range(batch_size)]  # inner list stores beam
    kv_cache = {}

    for step in range(max_length):  # inference step, beam forward
        # the last output for next input, view it as a batch of 10
        tgt_inputs = tgt_ids[:, :, -1].view(-1, 1)
        decoder_output = model.decode(
            tgt_inputs,
            encoder_output,
            src_mask=src_mask,
            kv_cache=kv_cache,
        )

        decoder_output = decoder_output[:, -1]  # choose the last token
        logits = model.output_layer(decoder_output)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # add penalization to the appeared tokens to reduce future repetitions
        if rep_penalty:
            log_probs = add_penalty(
                tgt_ids, log_probs, method='naive', naive_penalty=2)
        log_probs += cum_log_probs.reshape(-1, 1)

        vocab_size = log_probs.shape[-1]
        # we need to keep beam_size active sentences, hence we need a candidate list of 2*beam_size
        # we want to find top-2*beam_size prob(and id) from beam_size*vocab_size candidates for each beam
        cum_log_probs, top_ids = torch.topk(
            # so we need to arrange together, that's why we merge the two axis.
            log_probs.view(-1, beam_size * vocab_size),
            k=beam_size * 2,
            dim=-1,
        )
        # once we find them, we need to convert back to find the beam it comes from.
        from_beam = top_ids // vocab_size
        top_ids = top_ids % vocab_size

        tgt_ids = index_beam(tgt_ids, from_beam)
        tgt_ids = torch.cat([tgt_ids, top_ids.unsqueeze(-1)], dim=-1)

        for i in range(batch_size):
            if finished[i]:
                continue

            # loop to decide if k-th beam need reload a new candidate (sentence is finished)
            for k in range(beam_size):
                # top_ids stores the candidate tokens chosen in this step
                last_id = top_ids[i, k]

                if last_id != eos and step + 1 < max_length:
                    continue
                # if so, pop up the finished sentence, add it to hypotheses list
                hypothesis = tgt_ids[i, k, 1:].tolist()
                score = cum_log_probs[i, k] / \
                    (len(hypothesis) ** length_penalty)

                finished_hypotheses[i].append((score, hypothesis))

                # Replace the finished hypothesis by an active candidate.
                for j in range(beam_size, beam_size * 2):
                    if top_ids[i, j] != eos:
                        tgt_ids[i, k, -1] = top_ids[i, j]
                        cum_log_probs[i, k] = cum_log_probs[i, j]
                        from_beam[i, k] = from_beam[i, j]
                        top_ids[i, j] = eos
                        break

            if len(finished_hypotheses[i]) >= beam_size:
                finished[i] = True
                finished_hypotheses[i] = sorted(
                    finished_hypotheses[i],
                    key=lambda item: item[0],
                    reverse=True,
                )

        if all(finished):
            break
        # truncate 2*beam_size candidates to beam_size for next step
        tgt_ids = tgt_ids[:, :beam_size].contiguous()
        cum_log_probs = cum_log_probs[:, :beam_size].contiguous()
        from_beam = from_beam[:, :beam_size].contiguous()

        # what's the purpose?
        update_kv_cache(kv_cache, from_beam)
        # how from_beam actually works?

    return finished_hypotheses


def add_penalty(tgt_ids, log_probs, method, naive_penalty=None):
    # where to apply
    penalty_matrix = torch.zeros_like(log_probs)
    appeared_token_ids = tgt_ids.view(beam_size*batch_size, -1)
    # loop over position in sequence
    for i in appeared_token_ids.size(-1):
        penalty_matrix.scatter_add(1, appeared_token_ids[:, i])
    # how much to apply
    if method == 'naive':
        if naive_penalty is None:
            raise ValueError("please specify naive_penalty for method='naive'")
        log_probs -= penalty_matrix * naive_penalty
    else:
        pass
    return log_probs


def repeat_beam(x: torch.Tensor, beam_size):
    return x.repeat_interleave(beam_size, dim=0)


def index_beam(x: torch.Tensor, beam_ids):
    batch_size, beam_size = x.shape[:2]
    # we want to obtain tgt_ids at this step, which (target sequence) looks like (batch, beam, seq)
    # that is to choose the right beams
    # either we directly choose them by a 2-tuple index, or we do the following, we merge again
    batch_offset = torch.arange(batch_size, device=beam_ids.device) * beam_size
    flat_beam_ids = (beam_ids + batch_offset.view(-1, 1)).view(-1)

    flat_x = x.view(x.shape[0] * x.shape[1], *x.shape[2:])
    flat_x = flat_x.index_select(0, flat_beam_ids)

    x = flat_x.view(batch_size, flat_x.shape[0] // batch_size, *x.shape[2:])
    return x


def update_kv_cache(kv_cache: Dict[str, torch.Tensor], beam_ids):
    batch_size, beam_size = beam_ids.shape
    batch_offset = torch.arange(batch_size, device=beam_ids.device) * beam_size
    flat_beam_ids = (beam_ids + batch_offset.view(-1, 1)).view(-1)

    for name, value in kv_cache.items():
        kv_cache[name] = value.index_select(0, flat_beam_ids)


if __name__ == "__main__":
    main()
