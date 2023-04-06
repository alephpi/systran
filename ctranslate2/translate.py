import argparse
import sys

import ctranslate2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to the CTranslate2 model")
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
    translate(
        translator,
        sys.stdin,
        sys.stdout,
        max_batch_size=64 if use_gpu else 32,
        beam_size=5,
        length_penalty=1,
    )


def translate(translator, in_file, out_file, **kwargs):
    inputs = map(lambda line: line.strip().split(), in_file)

    results = translator.translate_iterable(inputs, **kwargs)

    for result in results:
        tokens = result.hypotheses[0]
        line = " ".join(tokens)

        out_file.write(line)
        out_file.write("\n")


if __name__ == "__main__":
    main()
