import argparse
import math
import os

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from dataset import get_dataloader
from model import CTCModel
from utils import decode, compute_wer


def greedy_decode_batch(log_probs: torch.Tensor, input_lengths: torch.Tensor) -> list[str]:
    preds = log_probs.argmax(dim=-1)
    return [decode(preds[i, :length].cpu().numpy()) for i, length in enumerate(input_lengths)]


@torch.no_grad()
def evaluate(model: CTCModel, loader, device: torch.device) -> float:
    model.eval()
    all_hyps, all_refs = [], []
    for hidden_states, input_lengths, _, _, texts in loader:
        hidden_states = hidden_states.to(device)
        log_probs = model(hidden_states).log_softmax(dim=-1)
        all_hyps.extend(greedy_decode_batch(log_probs, input_lengths))
        all_refs.extend(t.lower() for t in texts)
    model.train()
    return compute_wer(all_hyps, all_refs)


def get_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- WandB ----
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config=vars(args),
    )

    # ---- Data ----
    print('Loading datasets...')
    train_loader, _ = get_dataloader('train-clean-100', args.batch_size, shuffle=True,
                                     num_workers=args.workers, cache_to_ram=args.cache_ram)
    val_clean_loader, _ = get_dataloader('dev-clean', args.batch_size, shuffle=False,
                                         num_workers=args.workers, cache_to_ram=args.cache_ram)
    val_other_loader, _ = get_dataloader('dev-other', args.batch_size, shuffle=False,
                                         num_workers=args.workers, cache_to_ram=args.cache_ram)
    print(f'Train: {len(train_loader)} batches | val-clean: {len(val_clean_loader)} | val-other: {len(val_other_loader)}')

    # ---- Model ----
    model = CTCModel(dropout=args.dropout).to(device)
    wandb.watch(model, log='gradients', log_freq=100)
    print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    use_amp = args.fp16 and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    global_step = 0
    best_wer = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', dynamic_ncols=True)
        for hidden_states, input_lengths, targets, target_lengths, _ in pbar:
            lr = get_lr(global_step, warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            hidden_states = hidden_states.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(hidden_states)
                log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T, B, V)
                loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            epoch_loss += loss_val

            if global_step % args.log_interval == 0:
                wandb.log({
                    'train/loss': loss_val,
                    'train/lr': lr,
                    'train/grad_norm': grad_norm.item(),
                }, step=global_step)

            pbar.set_postfix(loss=f'{loss_val:.3f}', lr=f'{lr:.2e}')
            global_step += 1

        avg_loss = epoch_loss / len(train_loader)

        # ---- Validation ----
        wer_clean = evaluate(model, val_clean_loader, device)
        wer_other = evaluate(model, val_other_loader, device)

        # Log layer weights
        weights = torch.softmax(model.layer_weights, dim=0).detach().cpu()
        layer_weight_log = {f'model/layer_weight_{i}': w.item() for i, w in enumerate(weights)}

        wandb.log({
            'train/epoch_loss': avg_loss,
            'val/wer_clean': wer_clean * 100,
            'val/wer_other': wer_other * 100,
            **layer_weight_log,
            'epoch': epoch,
        }, step=global_step)

        print(f'\nEpoch {epoch}: loss={avg_loss:.4f} | WER clean={wer_clean*100:.2f}% other={wer_other*100:.2f}%')

        if wer_clean < best_wer:
            best_wer = wer_clean
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'best_model.pt'))
            print(f'  → Saved best model (WER clean={best_wer*100:.2f}%)')

        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f'epoch_{epoch:03d}.pt'))

    wandb.finish()
    print(f'\nDone. Best val WER (clean): {best_wer*100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cache_ram', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='ctc-asr')
    parser.add_argument('--wandb_run', type=str, default='w2v2bert-ctc')
    args = parser.parse_args()
    train(args)
