import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import VOCAB_SIZE


class CTCModel(nn.Module):
    """
    CTC head on top of pre-extracted W2V2-BERT hidden states.

    Architecture:
      1. Learnable scalar weights per layer (Softmax normalised) → weighted sum
      2. LayerNorm + Dropout
      3. 2-layer BiLSTM for temporal modelling
      4. Linear projection to vocab
    """

    def __init__(
        self,
        n_layers: int = 8,
        hidden_dim: int = 1024,
        lstm_hidden: int = 512,
        lstm_layers: int = 2,
        vocab_size: int = VOCAB_SIZE,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # BiLSTM: input=1024, hidden=512 (×2 directions = 1024 out)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.ctc_head = nn.Linear(lstm_hidden * 2, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, n_layers, T, hidden_dim)
        weights = F.softmax(self.layer_weights, dim=0)             # (n_layers,)
        x = (hidden_states * weights[None, :, None, None]).sum(1)  # (B, T, 1024)
        x = self.norm(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)                                        # (B, T, 1024)
        x = self.dropout(x)
        return self.ctc_head(x)                                    # (B, T, vocab_size)
