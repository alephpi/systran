import argparse
import sys

import ctranslate2


use_gpu = False

if use_gpu:
    translator_args = dict(
        device="cuda",
        device_index=0,
        compute_type="float16",
    )
else:
    translator_args = dict(
        device="cpu",
        intra_threads=1,
        inter_threads=12,
    )

translate_args = dict(
    max_batch_size=64,
    beam_size=4,
    length_penalty=0.6,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to the CTranslate2 model")
    args = parser.parse_args()

    translator = ctranslate2.Translator(args.model, **translator_args)

    inputs = map(lambda line: line.strip().split(), sys.stdin)

    results = translator.translate_iterable(inputs, **translate_args)

    for result in results:
        tokens = result.hypotheses[0]
        line = " ".join(tokens)

        sys.stdout.write(line)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
