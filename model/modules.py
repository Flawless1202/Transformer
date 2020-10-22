import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):

    def __init__(self, scale, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.scale = scale
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q * self.scale, k.transpose(2, 3))

        attn = attn.masked_fill(mask == 0, -torch.finfo(attn.dtype).min) if mask is not None else attn

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

        self.embed_qs = nn.Linear(d_model, d_k * num_heads)
        self.embed_ks = nn.Linear(d_model, d_k * num_heads)
        self.embed_vs = nn.Linear(d_model, d_v * num_heads)

        self.attn = ScaledDotProductAttention(scale=d_k ** (- 0.5))
        self.fc = nn.Linear(num_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, num_heads = self.d_k, self.d_v, self.num_heads
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.embed_qs(q).view(batch_size, len_q, num_heads, d_k).transpose(1, 2)
        k = self.embed_qs(k).view(batch_size, len_k, num_heads, d_k).transpose(1, 2)
        v = self.embed_qs(v).view(batch_size, len_v, num_heads, d_v).transpose(1, 2)

        mask = mask.unsqueeze(1) if mask is not None else mask

        q, attn = self.attn(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        q = self.norm(residual + q)

        return q, attn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.fc2(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x


def padding_mask(seq_k, seq_q):
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, seq_q.size(1), -1)
    return pad_mask


def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (- (torch.tensor(10000.0).log() / d_model)))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].clone().detach()
