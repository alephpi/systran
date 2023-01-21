import argparse

import ctranslate2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to the CTranslate2 model")
    parser.add_argument("src", help="Path to the source validation file")
    parser.add_argument("tgt", help="Path to the source validation file")
    parser.add_argument(
        "--num_gpus", type=int, default=0, help="Number of GPUs to use"
    )
    parser.add_argument(
        "--num_cpus", type=int, default=8, help="Number of CPUs to use"
    )
    args = parser.parse_args()

    use_gpu = args.num_gpus > 0

    if use_gpu:
        translator_args = dict(
            device="cuda",
            device_index=list(range(args.num_gpus)),
            compute_type="float16",
        )
    else:
        translator_args = dict(
            device="cpu",
            intra_threads=1,
            inter_threads=args.num_cpus,
        )

    translator = ctranslate2.Translator(args.model, **translator_args)

    with open(args.src) as src_file, open(args.tgt) as tgt_file:
        loss = evaluate(translator, src_file, tgt_file)

    print(loss)


def evaluate(translator, src_file, tgt_file):
    tokenize = lambda line: line.strip().split()

    src = map(tokenize, src_file)
    tgt = map(tokenize, tgt_file)

    results = translator.score_iterable(src, tgt)

    total_loss = 0
    total_tokens = 0

    for result in results:
        total_loss += -sum(result.log_probs)
        total_tokens += len(result.tokens)

    return total_loss / total_tokens


if __name__ == "__main__":
    main()
