def load_vocabulary(path):
    """Loads the vocabulary from a path."""

    with open(path) as vocabulary:
        ids_to_tokens = [line.rstrip("\r\n") for line in vocabulary]

    if "<unk>" not in ids_to_tokens:
        ids_to_tokens.append("<unk>")

    tokens_to_ids = {token: i for i, token in enumerate(ids_to_tokens)}

    return tokens_to_ids, ids_to_tokens


def encode_line(line, vocabulary, add_bos=False, add_eos=False, tokenize_fn=None):
    """Converts a text line into a list of token IDs."""

    bos_id = vocabulary["<s>"]
    eos_id = vocabulary["</s>"]
    unk_id = vocabulary["<unk>"]

    line = line.rstrip("\r\n")

    if tokenize_fn is None:
        tokens = line.split()
    else:
        tokens = tokenize_fn(line)

    if not tokens:
        return []

    ids = [vocabulary.get(token, unk_id) for token in tokens]

    if add_bos:
        ids.insert(0, bos_id)
    if add_eos:
        ids.append(eos_id)

    return ids
