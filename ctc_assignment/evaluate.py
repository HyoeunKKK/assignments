"""Evaluation script: reports WER on test-clean / test-other and shows prediction samples."""

import argparse

import torch
from tqdm import tqdm

from dataset import get_dataloader
from model import CTCModel
from utils import decode, compute_wer


@torch.no_grad()
def run_eval(model: CTCModel, split: str, device: torch.device, batch_size: int = 64, num_workers: int = 4):
    loader, _ = get_dataloader(split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()

    all_hyps, all_refs = [], []
    samples: list[tuple[str, str]] = []

    for hidden_states, input_lengths, _, _, texts in tqdm(loader, desc=split):
        hidden_states = hidden_states.to(device)
        log_probs = model(hidden_states).log_softmax(dim=-1)  # (B, T, V)
        preds = log_probs.argmax(dim=-1)                      # (B, T)

        for i, length in enumerate(input_lengths):
            hyp = decode(preds[i, :length].cpu().numpy())
            ref = texts[i].lower()
            all_hyps.append(hyp)
            all_refs.append(ref)
            if len(samples) < 5:
                samples.append((ref, hyp))

    wer = compute_wer(all_hyps, all_refs)

    print(f'\n{"="*60}')
    print(f'Split: {split}')
    print(f'WER: {wer * 100:.2f}%  ({len(all_hyps)} utterances)')
    print(f'{"="*60}')
    print('\nSample predictions:')
    for i, (ref, hyp) in enumerate(samples[:2], 1):
        print(f'\n[Sample {i}]')
        print(f'  REF: {ref}')
        print(f'  HYP: {hyp}')

    return wer, all_hyps, all_refs


def main():
    parser = argparse.ArgumentParser(description='CTC ASR evaluation')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = CTCModel()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)

    # Show learned layer weights
    weights = torch.softmax(model.layer_weights, dim=0).detach().cpu()
    print('\nLearned layer weights:')
    for i, w in enumerate(weights):
        print(f'  layer {i}: {w.item():.4f}')

    run_eval(model, 'test-clean', device, args.batch_size, args.workers)
    run_eval(model, 'test-other', device, args.batch_size, args.workers)


if __name__ == '__main__':
    main()
