import torch


class Transformer(torch.nn.Module):
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        num_layers=6,
        num_heads=16,
        dim_model=1024,
        dim_ffn=4096,
        dropout=0.1,
        max_length=1024,
        share_embeddings=False,
        padding_idx=0,
    ):
        super().__init__()

        self.config = dict(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_model=dim_model,
            dim_ffn=dim_ffn,
            dropout=dropout,
            max_length=max_length,
            share_embeddings=share_embeddings,
            padding_idx=padding_idx,
        )

        self.emb_scale = dim_model**0.5
        self.padding_idx = padding_idx

        self.encoder = TransformerEncoder(
            num_layers, dim_model, dim_ffn, num_heads, dropout
        )
        self.decoder = TransformerDecoder(
            num_layers, dim_model, dim_ffn, num_heads, dropout
        )

        self.src_embeddings = torch.nn.Embedding(
            src_vocab_size, dim_model, padding_idx=padding_idx
        )
        self.tgt_embeddings = (
            self.src_embeddings
            if share_embeddings
            else torch.nn.Embedding(tgt_vocab_size, dim_model, padding_idx=padding_idx)
        )

        self.output_layer = torch.nn.Linear(dim_model, tgt_vocab_size, bias=False)
        self.output_layer.weight = self.tgt_embeddings.weight

        self.dropout = torch.nn.Dropout(dropout)

        position_encodings = get_positional_embeddings(max_length, dim_model)
        self.register_buffer("position_encodings", position_encodings)

        triangular_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            max_length
        )
        self.register_buffer("triangular_mask", triangular_mask, persistent=False)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.uniform_(module.weight, -0.07, 0.07)

    def forward(self, src_ids, tgt_ids):
        encoder_output, src_mask = self.encode(src_ids)
        decoder_output = self.decode(tgt_ids, encoder_output, src_mask=src_mask)
        logits = self.output_layer(decoder_output)
        return logits

    def encode(self, src_ids):
        src_max_len = src_ids.shape[1]

        src_inputs = self.src_embeddings(src_ids) * self.emb_scale
        src_inputs += self.position_encodings[:src_max_len].unsqueeze(0)
        src_inputs = self.dropout(src_inputs)

        src_padding_mask = src_ids.eq(self.padding_idx)
        src_mask = src_inputs.new_zeros(src_padding_mask.shape)
        src_mask = src_mask.masked_fill(src_padding_mask, float("-inf"))
        src_mask = src_mask.view(-1, 1, 1, src_max_len)

        memory = self.encoder(src_inputs, mask=src_mask)
        return memory, src_mask

    def decode(self, tgt_ids, encoder_output, src_mask=None, kv_cache=None):
        batch_size, tgt_max_len = tgt_ids.shape
        offset = next(iter(kv_cache.values())).shape[2] if kv_cache else 0

        tgt_inputs = self.tgt_embeddings(tgt_ids) * self.emb_scale
        tgt_inputs += self.position_encodings[offset:offset + tgt_max_len].unsqueeze(0)
        tgt_inputs = self.dropout(tgt_inputs)

        tgt_padding_mask = tgt_ids.eq(self.padding_idx).unsqueeze(1)
        tgt_mask = self.triangular_mask[:tgt_max_len, :tgt_max_len].unsqueeze(0)
        tgt_mask = tgt_mask.expand(batch_size, -1, -1)
        tgt_mask = tgt_mask.masked_fill(tgt_padding_mask, float("-inf"))
        tgt_mask = tgt_mask.view(-1, 1, tgt_max_len, tgt_max_len)

        outputs = self.decoder(
            tgt_inputs,
            encoder_output,
            mask=tgt_mask,
            memory_mask=src_mask,
            kv_cache=kv_cache,
        )

        return outputs


class TransformerEncoder(torch.nn.Module):
    def __init__(self, num_layers, embed_dim, ffn_dim, attention_heads, dropout):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, ffn_dim, attention_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)
        return x


class TransformerDecoder(torch.nn.Module):
    def __init__(self, num_layers, embed_dim, ffn_dim, attention_heads, dropout):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                TransformerDecoderLayer(embed_dim, ffn_dim, attention_heads, dropout, i)
                for i in range(num_layers)
            ]
        )

        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x, memory, mask=None, memory_mask=None, kv_cache=None):
        for layer in self.layers:
            x = layer(x, memory, mask=mask, memory_mask=memory_mask, kv_cache=kv_cache)

        x = self.norm(x)
        return x


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, embed_dim, ffn_dim, attention_heads, dropout):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.self_attention = MultiHeadAttention(
            embed_dim, attention_heads, dropout, self_attention=True
        )

        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        y = self.self_attention(self.norm1(x), mask=mask)
        x = self.dropout(y) + x

        y = self.ffn(self.norm2(x))
        x = self.dropout(y) + x

        return x


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self, embed_dim, ffn_dim, attention_heads, dropout, layer_index):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.self_attention = MultiHeadAttention(
            embed_dim, attention_heads, dropout, self_attention=True, layer_index=layer_index
        )

        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(
            embed_dim, attention_heads, dropout, self_attention=False, layer_index=layer_index
        )

        self.norm3 = torch.nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, memory, mask=None, memory_mask=None, kv_cache=None):
        y = self.self_attention(self.norm1(x), mask=mask, kv_cache=kv_cache)
        x = self.dropout(y) + x

        y = self.attention(self.norm2(x), memory, mask=memory_mask, kv_cache=kv_cache)
        x = self.dropout(y) + x

        y = self.ffn(self.norm3(x))
        x = self.dropout(y) + x

        return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, attention_heads, dropout, self_attention, layer_index=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.dropout = dropout

        if self_attention:
            self.in_proj = torch.nn.Linear(embed_dim, embed_dim * 3)
        else:
            self.query_proj = torch.nn.Linear(embed_dim, embed_dim)
            self.value_proj = torch.nn.Linear(embed_dim, embed_dim * 2)

        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

        self.cache_prefix = "self_attention" if self_attention else "attention"
        self.cache_prefix = "%s_%d" % (self.cache_prefix, layer_index)

    def forward(self, query, value=None, mask=None, kv_cache=None):
        if kv_cache is not None:
            cached_key = kv_cache.get("%s_key" % self.cache_prefix)
            cached_value = kv_cache.get("%s_value" % self.cache_prefix)
        else:
            cached_key, cached_value = None, None

        if value is None:
            proj = self.in_proj(query)
            proj = split_heads(proj, self.attention_heads * 3)
            query, key, value = proj.split(self.attention_heads, dim=1)

            if cached_key is not None:
                key = torch.cat([cached_key, key], dim=2)
                value = torch.cat([cached_value, value], dim=2)

        else:
            query = self.query_proj(query)
            query = split_heads(query, self.attention_heads)

            if cached_key is not None:
                key = cached_key
                value = cached_value
            else:
                proj = self.value_proj(value)
                proj = split_heads(proj, self.attention_heads * 2)
                key, value = proj.split(self.attention_heads, dim=1)

        if kv_cache is not None:
            kv_cache["%s_key" % self.cache_prefix] = key
            kv_cache["%s_value" % self.cache_prefix] = value

        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0,
        )

        output = combine_heads(output)
        output = self.out_proj(output)

        return output


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, outer_dim, inner_dim, dropout):
        super().__init__()
        self.inner = torch.nn.Linear(outer_dim, inner_dim)
        self.outer = torch.nn.Linear(inner_dim, outer_dim)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.inner(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.outer(x)
        return x


def split_heads(x, heads):
    x = x.reshape(x.shape[0], x.shape[1], heads, -1)
    x = x.transpose(1, 2)
    return x


def combine_heads(x):
    x = x.transpose(1, 2)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    return x


def get_positional_embeddings(length, depth, device=None):
    channels = torch.arange(depth // 2).unsqueeze(0)
    positions = torch.arange(0, length).unsqueeze(1)
    scaled_positions = positions / torch.pow(10000, (2 * channels) / depth)
    sin = torch.sin(scaled_positions)
    cos = torch.cos(scaled_positions)
    encodings = torch.hstack([sin, cos])
    return encodings.to(device)
