import math
from typing import Optional

import torch
import torch.nn as nn


def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def generate_square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones((size, size), device=device), diagonal=1).bool()
    return ~mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        depth = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(depth)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return torch.matmul(weights, value), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size = query.size(0)

        def transform(x, linear):
            x = linear(x)
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = transform(query, self.q_linear)
        k = transform(key, self.k_linear)
        v = transform(value, self.v_linear)

        attention_output, _ = self.attention(q, k, v, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out_linear(attention_output)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(1)].transpose(0, 1)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(attn_output))
        ff_output = self.feed_forward(src)
        return self.norm2(src + self.dropout(ff_output))


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn_output))
        attn_output = self.enc_dec_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(attn_output))
        ff_output = self.feed_forward(tgt)
        return self.norm3(tgt + self.dropout(ff_output))


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.dropout = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(num_encoder_layers, embed_dim, num_heads, hidden_dim, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, embed_dim, num_heads, hidden_dim, dropout)
        self.generator = nn.Linear(embed_dim, vocab_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        src_emb = self.embedding(src) * math.sqrt(self.embed_dim)
        src_emb = self.dropout(self.positional_encoding(src_emb))
        memory = self.encoder(src_emb, src_mask)

        tgt_emb = self.embedding(tgt) * math.sqrt(self.embed_dim)
        tgt_emb = self.dropout(self.positional_encoding(tgt_emb))
        output = self.decoder(tgt_emb, memory, tgt_mask, src_mask)
        return self.generator(output)

    def greedy_decode(self, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, sos_idx: int, eos_idx: int) -> torch.Tensor:
        batch_size = src.size(0)
        memory = self.encoder(self.positional_encoding(self.embedding(src) * math.sqrt(self.embed_dim)), src_mask)
        ys = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=src.device)

        for i in range(max_len - 1):
            tgt_mask = generate_square_subsequent_mask(ys.size(1), src.device).unsqueeze(0).unsqueeze(0)
            decoder_output = self.decoder(self.positional_encoding(self.embedding(ys) * math.sqrt(self.embed_dim)), memory, tgt_mask, src_mask)
            logits = self.generator(decoder_output[:, -1:])
            next_token = logits.argmax(dim=-1)
            ys = torch.cat([ys, next_token], dim=1)
            if torch.all(next_token.squeeze(-1) == eos_idx):
                break
        return ys
