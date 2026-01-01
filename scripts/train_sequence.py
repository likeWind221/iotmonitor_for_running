"""Train a simple GRU-based sequence classifier on window-aggregated CSV data.

Usage examples:
  python scripts/train_sequence.py --train data/splits/train.csv --val data/splits/val.csv --seq-len 10 --epochs 10
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
# robust import for SequenceDataset: support being used as script or package
try:
    # when imported as a package (scripts.train_sequence)
    from .sequence_dataset import SequenceDataset
except Exception:
    # fallback when run as a script (python scripts/train_sequence.py)
    from sequence_dataset import SequenceDataset


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, num_classes=3, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        # x: (B, L, F)
        out, h = self.gru(x)
        # use last hidden state
        last = out[:, -1, :]
        return self.fc(last)


def collate_fn(batch):
    X = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return X, y


def train_epoch(model, loader, opt, criterion, device, amp=False, grad_scaler=None):
    model.train()
    losses = []
    ys, ps = [], []
    for X,y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        if amp and device.type == 'cuda' and grad_scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(X)
                loss = criterion(logits, y)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(opt)
            grad_scaler.update()
        else:
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
        losses.append(loss.item())
        preds = logits.argmax(dim=1).cpu().numpy()
        ys.extend(y.cpu().numpy().tolist())
        ps.extend(preds.tolist())
    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average='macro')
    return np.mean(losses), acc, f1


def eval_epoch(model, loader, criterion, device, amp=False):
    model.eval()
    losses = []
    ys, ps = [], []
    with torch.no_grad():
        for X,y in loader:
            X, y = X.to(device), y.to(device)
            if amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    logits = model(X)
                    loss = criterion(logits, y)
            else:
                logits = model(X)
                loss = criterion(logits, y)
            losses.append(loss.item())
            preds = logits.argmax(dim=1).cpu().numpy()
            ys.extend(y.cpu().numpy().tolist())
            ps.extend(preds.tolist())
    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average='macro')
    return np.mean(losses), acc, f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True)
    p.add_argument('--val', required=False)
    p.add_argument('--seq-len', type=int, default=10)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--save-dir', default='models/seq')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--patience', type=int, default=8, help='early stopping patience in epochs (based on val_f1)')
    p.add_argument('--scheduler', choices=['none','reduce','cosine'], default='none', help='LR scheduler')
    p.add_argument('--amp', action='store_true', help='use automatic mixed precision (requires CUDA)')
    p.add_argument('--save-every', type=int, default=0, help='save intermediate checkpoints every N epochs (0 to disable)')
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print('Loading train dataset...')
    train_ds = SequenceDataset(args.train, seq_len=args.seq_len, mode='train')
    print('Number of train samples:', len(train_ds))
    # pass trained scaler to val/test datasets
    scaler = train_ds.scaler
    if args.val:
        val_ds = SequenceDataset(args.val, seq_len=args.seq_len, mode='val', scaler=scaler)
        print('Number of val samples:', len(val_ds))
    else:
        val_ds = None

    n_classes = len(train_ds.get_label_map())
    input_dim = len(train_ds.features)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn) if val_ds is not None else None

    device = torch.device(args.device)
    model = GRUClassifier(input_dim=input_dim, hidden_dim=args.hidden, num_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # scheduler
    scheduler = None
    if args.scheduler == 'reduce' and val_loader is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=max(1, args.patience//2), min_lr=1e-6)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    # AMP setup
    use_amp = args.amp and device.type == 'cuda'
    grad_scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_f1 = 0.0
    epochs_since_best = 0
    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, opt, criterion, device, amp=use_amp, grad_scaler=grad_scaler)
        if val_loader is not None:
            val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device, amp=use_amp)
            print(f"Epoch {ep}/{args.epochs} - train loss:{tr_loss:.4f} acc:{tr_acc:.4f} f1:{tr_f1:.4f} | val loss:{val_loss:.4f} acc:{val_acc:.4f} f1:{val_f1:.4f}")
            # scheduler step
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1)
            elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                scheduler.step()

            # early stopping / checkpoint
            if val_f1 > best_f1 + 1e-6:
                best_f1 = val_f1
                epochs_since_best = 0
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'gru_best.pth'))
                print('🔖 New best model saved (val_f1={:.4f})'.format(best_f1))
            else:
                epochs_since_best += 1
                print(f'⏳ epochs since best: {epochs_since_best}/{args.patience}')
                if epochs_since_best >= args.patience:
                    print('⚠️ Early stopping triggered')
                    break
        else:
            print(f"Epoch {ep}/{args.epochs} - train loss:{tr_loss:.4f} acc:{tr_acc:.4f} f1:{tr_f1:.4f}")
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                scheduler.step()

        if args.save_every and args.save_every > 0 and ep % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'gru_ep{ep}.pth'))

    # final save
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'gru_final.pth'))
    train_ds.save_scaler(os.path.join(args.save_dir, 'scaler.pkl'))
    # save label map
    import json
    with open(os.path.join(args.save_dir, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(train_ds.get_label_map(), f, ensure_ascii=False)

    print('Done. Model and scaler saved to', args.save_dir)


if __name__ == '__main__':
    main()
