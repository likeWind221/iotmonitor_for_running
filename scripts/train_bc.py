"""
行为克隆（BC）基线训练脚本 - PyTorch MLP
用法示例：
  python scripts/train_bc.py --train data/splits/train.csv --val data/splits/val.csv --test data/splits/test.csv --out models/bc

功能：
- 读取 CSV（每行为一个 5s 窗口特征/标签）
- 特征选择、标准化（保存 scaler）
- MLP 分类器训练（CrossEntropyLoss + class weights）
- 早停（基于 val loss）
- 输出：保存模型（PyTorch）、scaler（pickle）、训练日志（CSV）和评估报告（JSON）

注意：需要安装 torch, scikit-learn, pandas, numpy
"""
import os
import argparse
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


LABEL_MAP = {'acc': 0, 'keep': 1, 'dec': 2}
INV_LABEL_MAP = {v:k for k,v in LABEL_MAP.items()}

FEATURES = ['avg_hr','hr_slope','avg_spo2','speed_mps','speed_trend','angleX_mean','angleX_std','angleY_mean','angleY_std','target_speed','fatigue_score']


class WindowDataset(Dataset):
    def __init__(self, df, scaler=None, fit_scaler=False):
        self.df = df.copy().reset_index(drop=True)
        self.X = self.df[FEATURES].fillna(0.0).values.astype(np.float32)
        self.y = self.df['expert_action'].map(LABEL_MAP).values.astype(np.int64)
        if scaler is None and fit_scaler:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        elif scaler is not None:
            self.scaler = scaler
            self.X = self.scaler.transform(self.X)
        else:
            self.scaler = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128, n_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2]).tolist()
    return {
        'accuracy': float(acc),
        'precision': [float(x) for x in prec.tolist()],
        'recall': [float(x) for x in rec.tolist()],
        'f1': [float(x) for x in f1.tolist()],
        'confusion_matrix': cm
    }


def train(args):
    os.makedirs(args.out, exist_ok=True)
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val) if args.val else None
    test_df = pd.read_csv(args.test) if args.test else None

    # datasets & scalers
    train_ds = WindowDataset(train_df, scaler=None, fit_scaler=True)
    scaler = train_ds.scaler
    val_ds = WindowDataset(val_df, scaler=scaler) if val_df is not None else None
    test_ds = WindowDataset(test_df, scaler=scaler) if test_df is not None else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) if val_ds is not None else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(in_dim=len(FEATURES), hidden=args.hidden, n_classes=3, dropout=args.dropout).to(device)

    # class weights
    labels = train_df['expert_action'].map(LABEL_MAP).values
    class_sample_counts = np.bincount(labels, minlength=3)
    class_weights = torch.tensor(1.0 / (class_sample_counts + 1e-6), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs+1):
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            logits = model(Xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))
        val_loss = None
        val_metrics = None
        if val_loader is not None:
            model.eval()
            losses = []
            ys, yps = [], []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb = Xb.to(device)
                    yb = yb.to(device)
                    logits = model(Xb)
                    loss = criterion(logits, yb)
                    losses.append(loss.item())
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    ys.append(yb.cpu().numpy())
                    yps.append(preds)
            val_loss = float(np.mean(losses))
            ys = np.concatenate(ys)
            yps = np.concatenate(yps)
            val_metrics = compute_metrics(ys, yps)

        history.append({'epoch':epoch,'train_loss':avg_train_loss,'val_loss':val_loss,'val_metrics':val_metrics})
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "None"
        print(f"Epoch {epoch} train_loss={avg_train_loss:.4f} val_loss={val_loss_str}")

        # early stopping
        if val_loss is not None:
            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # save best model
                torch.save(model.state_dict(), os.path.join(args.out, 'bc_model_best.pth'))
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print('Early stopping')
                break

    # final save
    torch.save(model.state_dict(), os.path.join(args.out, 'bc_model_final.pth'))
    with open(os.path.join(args.out, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    # save history
    pd.DataFrame(history).to_csv(os.path.join(args.out, 'train_history.csv'), index=False)

    # evaluate on test
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model.eval()
        ys, yps = [], []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                logits = model(Xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                ys.append(yb.cpu().numpy())
                yps.append(preds)
        ys = np.concatenate(ys)
        yps = np.concatenate(yps)
        test_metrics = compute_metrics(ys, yps)
        print('Test metrics:', test_metrics)
        with open(os.path.join(args.out, 'test_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    print('Training complete. Models and artifacts saved to', args.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True)
    p.add_argument('--val', required=True)
    p.add_argument('--test', required=False)
    p.add_argument('--out', default='models/bc')
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--min-delta', type=float, default=1e-4)
    args = p.parse_args()

    train(args)
