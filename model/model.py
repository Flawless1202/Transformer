import torch
import torch.nn as nn
import numpy as np
from .modules import PositionalEncoding, MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_hid, num_heads, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(num_heads, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_hid, dropout=dropout)

    def forward(self, enc_input, self_attn_mask=None):
        enc_output, enc_self_attn = self.self_attn(enc_input, enc_input, enc_input, mask=self_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_self_attn


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_hid, num_heads, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(num_heads, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(num_heads, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_hid, dropout=dropout)

    def forward(self, dec_input, enc_output, self_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_self_attn = self.self_attn(dec_input, dec_input, dec_input, mask=self_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):

    def __init__(self, n_src_vocab, d_word_vec, n_layers, n_heads, d_k, d_v, d_model, d_hid, pad_idx,
                 dropout=0.1, max_len=200.):
        super(Encoder, self).__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_word_vec, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_hid, n_heads, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_self_attn_list = []

        enc_output = self.dropout(self.pos_enc(self.src_word_emb(src_seq)))
        enc_output = self.norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_self_attn = enc_layer(enc_output, self_attn_mask=src_mask)
            enc_self_attn_list += [enc_self_attn] if return_attns else []

        return enc_output, enc_self_attn_list if return_attns else enc_output,


class Decoder(nn.Module):

    def __init__(self, n_trg_vocab, d_word_vec, n_layers, n_heads, d_k, d_v, d_model, d_hid, pad_idx,
                 max_len=200, dropout=0.1):
        super(Decoder, self).__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_word_vec, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_hid, n_heads, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        dec_self_attn_list, dec_enc_attn_list = [], []

        dec_output = self.dropout(self.pos_enc(self.trg_word_emb(trg_seq)))
        dec_output = self.norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_self_attn, dec_enc_attn = dec_layer(dec_output, enc_output,
                                                                self_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_self_attn_list += [dec_self_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        return dec_output, dec_self_attn_list, dec_enc_attn_list if return_attns else dec_output,


class Transformer(nn.Module):

    def __init__(self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
                 d_word_vec=512, d_model=512, d_hid=2048,
                 n_layers=6, n_heads=8, d_k=64, d_v=64, dropout=0.1, max_len=200,
                 trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True):
        super(Transformer, self).__init__()

        assert d_model == d_word_vec, \
            "To facilitate the resudual connections, the dimensions of all module outputs shall be the same."

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.x_logit_scale = (d_model ** (- 0.5))

        self.encoder = Encoder(n_src_vocab, d_word_vec, n_layers, n_heads, d_k, d_v, d_model, d_hid,
                               pad_idx=src_pad_idx, max_len=max_len, dropout=dropout)
        self.decoder = Decoder(n_trg_vocab, d_word_vec, n_layers, n_heads, d_k, d_v, d_model, d_hid,
                               pad_idx=trg_pad_idx, max_len=max_len, dropout=dropout)
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        self.init_weights(trg_emb_prj_weight_sharing, emb_src_trg_weight_sharing)

    def init_weights(self, trg_emb_prj_weight_sharing, emb_src_trg_weight_sharing):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if trg_emb_prj_weight_sharing:
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight  # ???

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq, trg_seq):
        src_mask = self._get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = self._get_pad_mask(trg_seq, self.trg_pad_idx) & self._get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))

    @staticmethod
    def _get_pad_mask(seq, pad_idx):
        return (seq != pad_idx).unsqueeze(-2)

    @staticmethod
    def _get_subsequent_mask(seq):
        len_seq = seq.size(-1)
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_seq, len_seq), device=seq.device), diagonal=1)).bool()
        return subsequent_mask
