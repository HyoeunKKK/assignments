from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import encode

LIBRISPEECH_ROOT = Path('/mnt/elice/datahub/speech-dataset/librispeech/LibriSpeech')
FEATURES_ROOT = Path('/mnt/elice/datahub/speech-dataset/cached_features')


def _load_one(args):
    npz_path, utt_id = args
    try:
        d = np.load(npz_path)
        return utt_id, d['hidden_states'], int(d['seq_len'])  # float16 kept to save RAM
    except Exception:
        return utt_id, None, None


class LibriSpeechDataset(Dataset):
    def __init__(self, split: str, cache_to_ram: bool = False, num_load_workers: int = 8):
        self.split = split
        self.features_dir = FEATURES_ROOT / split

        # Build utterance_id → transcript mapping
        transcripts: dict[str, str] = {}
        for trans_file in (LIBRISPEECH_ROOT / split).rglob('*.trans.txt'):
            with open(trans_file) as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        transcripts[parts[0]] = parts[1]

        # Build utterance list
        self.items: list[str] = []
        for npz_path in sorted(self.features_dir.glob('*.npz')):
            utt_id = npz_path.stem
            if utt_id in transcripts and len(encode(transcripts[utt_id])) > 0:
                self.items.append(utt_id)
        self.transcripts = transcripts

        # Optional: load all features into RAM (pays off over many epochs)
        self.cache: dict[str, tuple] = {}
        if cache_to_ram:
            print(f'[{split}] Loading {len(self.items)} files into RAM...')
            args_list = [
                (self.features_dir / f'{uid}.npz', uid)
                for uid in self.items
            ]
            with ThreadPoolExecutor(max_workers=num_load_workers) as ex:
                for utt_id, hidden, seq_len in tqdm(
                    ex.map(_load_one, args_list), total=len(args_list), desc=split
                ):
                    if hidden is not None:
                        self.cache[utt_id] = (hidden, seq_len)
            # Remove utterances that failed to load
            self.items = [uid for uid in self.items if uid in self.cache]
            print(f'[{split}] Cached {len(self.items)} utterances.')

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        utt_id = self.items[idx]
        if utt_id in self.cache:
            hidden_np, seq_len = self.cache[utt_id]
        else:
            d = np.load(self.features_dir / f'{utt_id}.npz')
            hidden_np = d['hidden_states']
            seq_len = int(d['seq_len'])

        hidden = torch.from_numpy(hidden_np.astype(np.float32))  # (8, T, 1024)
        encoded = torch.tensor(encode(self.transcripts[utt_id]), dtype=torch.long)
        return hidden, seq_len, encoded, self.transcripts[utt_id]


def collate_fn(batch):
    hiddens, seq_lens, encodeds, texts = zip(*batch)

    B = len(hiddens)
    n_layers = hiddens[0].shape[0]
    hidden_dim = hiddens[0].shape[2]
    max_T = max(h.shape[1] for h in hiddens)

    padded = torch.zeros(B, n_layers, max_T, hidden_dim)
    for i, h in enumerate(hiddens):
        T = h.shape[1]
        padded[i, :, :T, :] = h

    input_lengths = torch.tensor(seq_lens, dtype=torch.long)
    targets = torch.cat(encodeds)
    target_lengths = torch.tensor([len(e) for e in encodeds], dtype=torch.long)

    return padded, input_lengths, targets, target_lengths, texts


def get_dataloader(
    split: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    cache_to_ram: bool = False,
) -> tuple[DataLoader, LibriSpeechDataset]:
    dataset = LibriSpeechDataset(split, cache_to_ram=cache_to_ram)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers if not cache_to_ram else 0,  # RAM cache → no need for workers
        pin_memory=True,
        persistent_workers=(num_workers > 0 and not cache_to_ram),
    )
    return loader, dataset
