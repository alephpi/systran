import argparse
import sys

import torch

from data import encode_line, load_vocabulary
from dataset import TextFileDataset, MapDataset, BatchDataset, to_tensor
from model import Transformer
from utils import init_logger


def main():
    init_logger()

    parser = argparse.ArgumentParser()

    data_options = parser.add_argument_group("Data options")
    data_options.add_argument(
        "--src_vocab", required=True, help="Path to the source vocabulary"
    )
    data_options.add_argument(
        "--tgt_vocab", required=True, help="Path to the target vocabulary"
    )
    data_options.add_argument("--batch_size", type=int, default=16, help="Batch size")

    model_options = parser.add_argument_group("Model options")
    model_options.add_argument("--ckpt", required=True, help="Path to the checkpoint")
    model_options.add_argument("--device", default="cpu", help="Device to use")

    decoding_options = parser.add_argument_group("Decoding options")
    decoding_options.add_argument("--beam_size", type=int, default=5, help="Beam size")
    decoding_options.add_argument(
        "--length_penalty", type=float, default=1, help="Length penalty"
    )
    decoding_options.add_argument(
        "--max_length", type=int, default=256, help="Maximum decoding length"
    )

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

    dataset = create_dataset(sys.stdin, source_vocabulary, args.batch_size, args.device)

    with torch.no_grad():
        for batch in dataset:
            result = beam_search(
                model,
                batch,
                bos,
                eos,
                beam_size=args.beam_size,
                length_penalty=args.length_penalty,
                max_length=args.max_length,
            )

            for hypotheses in result:
                tokens = hypotheses[0][1]
                if tokens and tokens[-1] == eos:
                    tokens.pop(-1)
                tokens = [target_vocabulary_rev[token_id] for token_id in tokens]
                print(" ".join(tokens), flush=True)


def create_dataset(path, source_vocabulary, batch_size, device):
    dataset = TextFileDataset(path)
    dataset = MapDataset(
        dataset, lambda line: encode_line(line, source_vocabulary, add_eos=True)
    )
    dataset = BatchDataset(dataset, batch_size)
    dataset = MapDataset(dataset, lambda batch: to_tensor(batch, device=device))
    return dataset


def beam_search(
    model,
    src_ids,
    bos,
    eos,
    beam_size=5,
    length_penalty=1,
    max_length=256,
):
    batch_size = src_ids.shape[0]

    encoder_output, src_mask = model.encode(src_ids)
    encoder_output = repeat_beam(encoder_output, beam_size)
    src_mask = repeat_beam(src_mask, beam_size)

    tgt_ids = torch.full(
        (batch_size, beam_size, 1), bos, dtype=src_ids.dtype, device=src_ids.device
    )

    cum_log_probs = torch.full(
        (batch_size, beam_size), float("-inf"), device=src_ids.device
    )
    cum_log_probs[:, 0] = 0

    finished = [False for _ in range(batch_size)]
    finished_hypotheses = [[] for _ in range(batch_size)]

    kv_cache = {}

    for step in range(max_length):
        tgt_inputs = tgt_ids[:, :, -1].view(-1, 1)

        decoder_output = model.decode(
            tgt_inputs,
            encoder_output,
            src_mask=src_mask,
            kv_cache=kv_cache,
        )

        decoder_output = decoder_output[:, -1]
        logits = model.output_layer(decoder_output)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs += cum_log_probs.reshape(-1, 1)

        vocab_size = log_probs.shape[-1]

        cum_log_probs, top_ids = torch.topk(
            log_probs.view(-1, beam_size * vocab_size),
            k=beam_size * 2,
            dim=-1,
        )

        from_beam = top_ids // vocab_size
        top_ids = top_ids % vocab_size

        tgt_ids = index_beam(tgt_ids, from_beam)
        tgt_ids = torch.cat([tgt_ids, top_ids.unsqueeze(-1)], dim=-1)

        for i in range(batch_size):
            if finished[i]:
                continue

            for k in range(beam_size):
                last_id = top_ids[i, k]

                if last_id != eos and step + 1 < max_length:
                    continue

                hypothesis = tgt_ids[i, k, 1:].tolist()
                score = cum_log_probs[i, k] / (len(hypothesis) ** length_penalty)

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

        tgt_ids = tgt_ids[:, :beam_size].contiguous()
        cum_log_probs = cum_log_probs[:, :beam_size].contiguous()
        from_beam = from_beam[:, :beam_size].contiguous()

        update_kv_cache(kv_cache, from_beam)

    return finished_hypotheses


def repeat_beam(x, beam_size):
    return x.repeat_interleave(beam_size, dim=0)


def index_beam(x, beam_ids):
    batch_size, beam_size = x.shape[:2]

    batch_offset = torch.arange(batch_size, device=beam_ids.device) * beam_size
    flat_beam_ids = (beam_ids + batch_offset.view(-1, 1)).view(-1)

    flat_x = x.view(x.shape[0] * x.shape[1], *x.shape[2:])
    flat_x = flat_x.index_select(0, flat_beam_ids)

    x = flat_x.view(batch_size, flat_x.shape[0] // batch_size, *x.shape[2:])
    return x


def update_kv_cache(kv_cache, beam_ids):
    batch_size, beam_size = beam_ids.shape
    batch_offset = torch.arange(batch_size, device=beam_ids.device) * beam_size
    flat_beam_ids = (beam_ids + batch_offset.view(-1, 1)).view(-1)

    for name, value in kv_cache.items():
        kv_cache[name] = value.index_select(0, flat_beam_ids)


if __name__ == "__main__":
    main()
