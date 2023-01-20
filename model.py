import torch


class Transformer(torch.nn.Module):
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
        batch_size = src_ids.shape[0]
        src_max_len = src_ids.shape[1]
        tgt_max_len = tgt_ids.shape[1]

        src_inputs = self.src_embeddings(src_ids) * self.emb_scale
        tgt_inputs = self.tgt_embeddings(tgt_ids) * self.emb_scale

        src_inputs += self.position_encodings[:src_max_len].unsqueeze(0)
        tgt_inputs += self.position_encodings[:tgt_max_len].unsqueeze(0)

        src_inputs = self.dropout(src_inputs)
        tgt_inputs = self.dropout(tgt_inputs)

        src_padding_mask = src_ids.eq(self.padding_idx)
        src_mask = src_inputs.new_zeros(src_padding_mask.shape)
        src_mask = src_mask.masked_fill(src_padding_mask, float("-inf"))
        src_mask = src_mask.view(-1, 1, 1, src_max_len)

        tgt_padding_mask = tgt_ids.eq(self.padding_idx).unsqueeze(1)
        tgt_mask = self.triangular_mask[:tgt_max_len, :tgt_max_len].unsqueeze(0)
        tgt_mask = tgt_mask.expand(batch_size, -1, -1)
        tgt_mask = tgt_mask.masked_fill(tgt_padding_mask, float("-inf"))
        tgt_mask = tgt_mask.view(-1, 1, tgt_max_len, tgt_max_len)

        memory = self.encoder(src_inputs, mask=src_mask)
        outputs = self.decoder(tgt_inputs, memory, mask=tgt_mask, memory_mask=src_mask)

        logits = self.output_layer(outputs)
        return logits


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
                TransformerDecoderLayer(embed_dim, ffn_dim, attention_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x, memory, mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, memory, mask=mask, memory_mask=memory_mask)

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
    def __init__(self, embed_dim, ffn_dim, attention_heads, dropout):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.self_attention = MultiHeadAttention(
            embed_dim, attention_heads, dropout, self_attention=True
        )

        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(
            embed_dim, attention_heads, dropout, self_attention=False
        )

        self.norm3 = torch.nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, memory, mask=None, memory_mask=None):
        y = self.self_attention(self.norm1(x), mask=mask)
        x = self.dropout(y) + x

        y = self.attention(self.norm2(x), memory, mask=memory_mask)
        x = self.dropout(y) + x

        y = self.ffn(self.norm3(x))
        x = self.dropout(y) + x

        return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, attention_heads, dropout, self_attention):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.dim_per_head = embed_dim // attention_heads
        self.query_scale = self.dim_per_head**-0.5

        if self_attention:
            self.in_proj = torch.nn.Linear(embed_dim, embed_dim * 3)
        else:
            self.query_proj = torch.nn.Linear(embed_dim, embed_dim)
            self.value_proj = torch.nn.Linear(embed_dim, embed_dim * 2)

        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query, value=None, mask=None):
        if value is None:
            proj = self.in_proj(query)
            proj = split_heads(proj, self.attention_heads * 3)
            query, key, value = proj.split(self.attention_heads, dim=1)

        else:
            query = self.query_proj(query)
            query = split_heads(query, self.attention_heads)

            proj = self.value_proj(value)
            proj = split_heads(proj, self.attention_heads * 2)
            key, value = proj.split(self.attention_heads, dim=1)

        dot = torch.matmul(query * self.query_scale, key.transpose(2, 3))

        if mask is not None:
            dot += mask

        attention_weight = self.softmax(dot)
        attention_weight = self.dropout(attention_weight)

        output = torch.matmul(attention_weight, value)
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
