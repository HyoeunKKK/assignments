import torch
import torch.nn as nn
from typing import List, Sequence, Tuple


class DSConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: Tuple[int, int] = (1, 1)):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_ch)

        self.pw = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=1,
            bias=False,
        )
        self.pw_bn = nn.BatchNorm2d(out_ch)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class DSCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 12,
        channels: Sequence[int] = (64, 64, 64, 128, 128, 128),
        block_strides: Sequence[Sequence[int]] = (
            (1, 1),
            (1, 1),
            (2, 2),
            (1, 1),
            (1, 1),
        ),
        dropout: float = 0.2,
    ):
        super().__init__()

        channels = list(channels)
        assert len(channels) >= 2, "channels must contain at least stem + 1 block output"
        assert len(block_strides) == len(channels) - 1, (
            "block_strides length must be len(channels) - 1"
        )

        stem_out = channels[0]

        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_out, kernel_size=3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(stem_out),
            nn.ReLU(inplace=True),
        )

        blocks = []
        in_ch = stem_out
        for out_ch, stride in zip(channels[1:], block_strides):
            blocks.append(
                DSConvBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    stride=tuple(stride),
                )
            )
            in_ch = out_ch

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)