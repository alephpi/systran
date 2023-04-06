import argparse

import torch
import ctranslate2

import numpy as np

from data import load_vocabulary
from train import num_layers, num_heads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_vocab", required=True, help="Path to the source vocabulary"
    )
    parser.add_argument(
        "--tgt_vocab", required=True, help="Path to the target vocabulary"
    )
    parser.add_argument("checkpoint", help="Path to the checkpoint to convert")
    ctranslate2.converters.Converter.declare_arguments(parser)
    args = parser.parse_args()

    converter = CT2Converter(
        args.checkpoint,
        args.src_vocab,
        args.tgt_vocab,
    )

    converter.convert_from_args(args)


class CT2Converter(ctranslate2.converters.Converter):
    def __init__(self, checkpoint_path, source_vocabulary_path, target_vocabulary_path):
        self.checkpoint_path = checkpoint_path
        self.source_vocabulary_path = source_vocabulary_path
        self.target_vocabulary_path = target_vocabulary_path

    def _load(self):
        _, source_vocabulary = load_vocabulary(self.source_vocabulary_path)
        _, target_vocabulary = load_vocabulary(self.target_vocabulary_path)

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        weights = checkpoint["model"]

        spec = ctranslate2.specs.TransformerSpec.from_config(
            num_layers=num_layers,
            num_heads=num_heads,
            pre_norm=True,
        )

        spec.register_source_vocabulary(source_vocabulary)
        spec.register_target_vocabulary(target_vocabulary)
        spec.config.add_source_eos = True

        set_embeddings(spec.encoder.embeddings[0], weights, "src_embeddings")
        set_embeddings(spec.decoder.embeddings, weights, "tgt_embeddings")

        set_position_encodings(spec.encoder.position_encodings, weights)
        set_position_encodings(spec.decoder.position_encodings, weights)

        set_layer_norm(spec.encoder.layer_norm, weights, "encoder.norm")
        set_layer_norm(spec.decoder.layer_norm, weights, "decoder.norm")

        set_linear(spec.decoder.projection, weights, "output_layer")

        for i, layer_spec in enumerate(spec.encoder.layer):
            layer_scope = "encoder.layers.%d" % i

            set_layer_norm(
                layer_spec.self_attention.layer_norm, weights, "%s.norm1" % layer_scope
            )
            set_attention(
                layer_spec.self_attention,
                weights,
                "%s.self_attention" % layer_scope,
                True,
            )

            set_layer_norm(layer_spec.ffn.layer_norm, weights, "%s.norm2" % layer_scope)
            set_linear(layer_spec.ffn.linear_0, weights, "%s.ffn.inner" % layer_scope)
            set_linear(layer_spec.ffn.linear_1, weights, "%s.ffn.outer" % layer_scope)

        for i, layer_spec in enumerate(spec.decoder.layer):
            layer_scope = "decoder.layers.%d" % i

            set_layer_norm(
                layer_spec.self_attention.layer_norm, weights, "%s.norm1" % layer_scope
            )
            set_attention(
                layer_spec.self_attention,
                weights,
                "%s.self_attention" % layer_scope,
                True,
            )

            set_layer_norm(
                layer_spec.attention.layer_norm, weights, "%s.norm2" % layer_scope
            )
            set_attention(
                layer_spec.attention, weights, "%s.attention" % layer_scope, False
            )

            set_layer_norm(layer_spec.ffn.layer_norm, weights, "%s.norm3" % layer_scope)
            set_linear(layer_spec.ffn.linear_0, weights, "%s.ffn.inner" % layer_scope)
            set_linear(layer_spec.ffn.linear_1, weights, "%s.ffn.outer" % layer_scope)

        return spec


def set_attention(spec, weights, scope, self_attention):
    if self_attention:
        set_linear(spec.linear[0], weights, "%s.in_proj" % scope)
    else:
        set_linear(spec.linear[0], weights, "%s.query_proj" % scope)
        set_linear(spec.linear[1], weights, "%s.value_proj" % scope)

    set_linear(spec.linear[-1], weights, "%s.out_proj" % scope)


def set_position_encodings(spec, weights):
    spec.encodings = weights["position_encodings"].numpy()


def set_embeddings(spec, weights, scope):
    spec.weight = weights["%s.weight" % scope].numpy()


def set_layer_norm(spec, weights, scope):
    spec.gamma = weights["%s.weight" % scope].numpy()
    spec.beta = weights["%s.bias" % scope].numpy()


def set_linear(spec, weights, scope):
    spec.weight = weights["%s.weight" % scope].numpy()

    try:
        spec.bias = weights["%s.bias" % scope].numpy()
    except KeyError:
        pass


if __name__ == "__main__":
    main()
