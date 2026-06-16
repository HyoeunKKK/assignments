"""FastSpeech2 model (duration-only variant).

Encoder/Decoder: stacks of FFT blocks (multi-head self-attention + 1D-conv FFN).
Variance adaptor: duration predictor only.
Length regulator: repeat encoder hidden states according to durations.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (max_len, d_model)


class ConvFFN(nn.Module):
    """1D-Conv feed-forward network used in FastSpeech FFT block."""

    def __init__(self, d_model: int, d_ff: int, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size, padding=pad)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        y = x.transpose(1, 2)
        y = F.relu(self.conv1(y))
        y = self.dropout(y)
        y = self.conv2(y)
        y = y.transpose(1, 2)
        return y


class FFTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = ConvFFN(d_model, d_ff, kernel_size, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # Pre-LN
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(a)
        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class FFTStack(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int,
                 kernel_size: int = 9, dropout: float = 0.1, max_len: int = 4000):
        super().__init__()
        self.layers = nn.ModuleList(
            [FFTBlock(d_model, n_heads, d_ff, kernel_size, dropout) for _ in range(n_layers)]
        )
        self.register_buffer("pe", get_sinusoidal_pe(max_len, d_model), persistent=False)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x, key_padding_mask=None):
        # x: (B, T, D)
        T = x.size(1)
        x = x + self.pe[:T].unsqueeze(0)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


class DurationPredictor(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 256, kernel_size: int = 3, dropout: float = 0.5):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(d_model, d_hidden, kernel_size, padding=pad)
        self.norm1 = nn.LayerNorm(d_hidden)
        self.conv2 = nn.Conv1d(d_hidden, d_hidden, kernel_size, padding=pad)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.linear = nn.Linear(d_hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (B, T, D); mask: (B, T) True at padding
        y = x.transpose(1, 2)
        y = F.relu(self.conv1(y)).transpose(1, 2)
        y = self.dropout(self.norm1(y)).transpose(1, 2)
        y = F.relu(self.conv2(y)).transpose(1, 2)
        y = self.dropout(self.norm2(y))
        y = self.linear(y).squeeze(-1)  # (B, T) log-duration prediction
        if mask is not None:
            y = y.masked_fill(mask, 0.0)
        return y


def length_regulate(x: torch.Tensor, durations: torch.Tensor, max_len: int | None = None):
    """Expand phoneme-level hidden states to frame-level using integer durations.

    x: (B, N, D)
    durations: (B, N) integer counts (must be >= 0)
    Returns expanded tensor (B, T, D) and frame lengths (B,).
    """
    B, N, D = x.shape
    lengths = durations.sum(dim=1)  # (B,)
    if max_len is None:
        T = int(lengths.max().item()) if B > 0 else 0
    else:
        T = max_len
    T = max(T, 1)
    out = x.new_zeros(B, T, D)
    for i in range(B):
        ex = torch.repeat_interleave(x[i], durations[i], dim=0)
        L = ex.shape[0]
        if L >= T:
            out[i] = ex[:T]
        else:
            out[i, :L] = ex
    return out, lengths


def make_pad_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """True at padded positions. shape: (B, max_len)"""
    if max_len is None:
        max_len = int(lengths.max().item())
    ar = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return ar >= lengths.unsqueeze(1)


class Postnet(nn.Module):
    """Tacotron2-style 5-layer 1D Conv postnet for mel residual refinement."""

    def __init__(self, n_mels: int = 80, d_hidden: int = 256, n_layers: int = 5,
                 kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        layers = []
        in_c = n_mels
        for i in range(n_layers):
            out_c = n_mels if i == n_layers - 1 else d_hidden
            layers.append(nn.Conv1d(in_c, out_c, kernel_size, padding=pad))
            if i < n_layers - 1:
                layers.append(nn.BatchNorm1d(out_c))
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.BatchNorm1d(out_c))
                layers.append(nn.Dropout(dropout))
            in_c = out_c
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, n_mels) -> residual (B, T, n_mels)
        y = x.transpose(1, 2)
        y = self.net(y)
        return y.transpose(1, 2)


class FastSpeech2(nn.Module):
    def __init__(self, vocab_size: int, n_mels: int = 80,
                 d_model: int = 256, n_heads: int = 2, d_ff: int = 1024,
                 enc_layers: int = 4, dec_layers: int = 4,
                 enc_kernel: int = 9, dec_kernel: int = 9,
                 dur_hidden: int = 256, dur_kernel: int = 3,
                 dropout: float = 0.1, dur_dropout: float = 0.5,
                 max_enc_len: int = 4000, max_dec_len: int = 4000,
                 pad_id: int = 0, use_postnet: bool = True):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.encoder = FFTStack(enc_layers, d_model, n_heads, d_ff, enc_kernel, dropout, max_enc_len)
        self.duration_predictor = DurationPredictor(d_model, dur_hidden, dur_kernel, dur_dropout)
        self.decoder = FFTStack(dec_layers, d_model, n_heads, d_ff, dec_kernel, dropout, max_dec_len)
        self.mel_linear = nn.Linear(d_model, n_mels)
        self.postnet = Postnet(n_mels=n_mels, d_hidden=256) if use_postnet else None
        self.n_mels = n_mels

    def forward(self, phonemes, phoneme_lengths, durations=None, mel_lengths=None):
        """Training forward when durations & mel_lengths provided (teacher forced).
        Inference forward when durations is None (use predicted durations).

        Returns dict with mel_before, mel_after, log_duration_pred, durations_used.
        """
        B = phonemes.size(0)
        src_mask = make_pad_mask(phoneme_lengths, max_len=phonemes.size(1))
        x = self.embed(phonemes)
        h_enc = self.encoder(x, key_padding_mask=src_mask)

        log_dur_pred = self.duration_predictor(h_enc.detach() if self.training else h_enc, mask=src_mask)

        if durations is not None:
            dur_for_expand = durations
            max_len = int(mel_lengths.max().item()) if mel_lengths is not None else None
        else:
            # at inference: convert log-duration to duration
            dur_for_expand = torch.clamp(torch.round(torch.exp(log_dur_pred) - 1.0), min=0).long()
            # zero out durations on padded phonemes
            dur_for_expand = dur_for_expand.masked_fill(src_mask, 0)
            max_len = None

        h_exp, lengths_out = length_regulate(h_enc, dur_for_expand, max_len=max_len)
        tgt_mask = make_pad_mask(lengths_out, max_len=h_exp.size(1))
        # If sequence is empty (no frames), avoid all-masked attention
        if (~tgt_mask).any():
            h_dec = self.decoder(h_exp, key_padding_mask=tgt_mask)
        else:
            h_dec = h_exp
        mel_before = self.mel_linear(h_dec)
        if self.postnet is not None:
            mel_after = mel_before + self.postnet(mel_before)
        else:
            mel_after = mel_before
        # Zero padded mel positions
        if tgt_mask is not None:
            keep = (~tgt_mask).unsqueeze(-1).float()
            mel_before = mel_before * keep
            mel_after = mel_after * keep
        return {
            "mel_before": mel_before,         # (B, T, n_mels)
            "mel_after": mel_after,           # (B, T, n_mels)
            "log_duration_pred": log_dur_pred,  # (B, N)
            "durations_used": dur_for_expand,  # (B, N)
            "mel_lengths_out": lengths_out,    # (B,)
            "src_mask": src_mask,              # (B, N) True at pad
            "tgt_mask": tgt_mask,              # (B, T) True at pad
        }
