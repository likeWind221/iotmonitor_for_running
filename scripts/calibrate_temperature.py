"""
Calibration script: temperature scaling for BC model.
Usage:
  python scripts/calibrate_temperature.py --val data/splits/val.csv --model models/bc/bc_model_best.pth --scaler models/bc/scaler.pkl --out-dir models/bc --bins 10
Outputs:
 - saves temperature to <out-dir>/temperature.txt
 - saves reliability plot to logs/reliability.png
 - prints ECE before/after and temperature
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
# ensure project root is importable when running the script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train_bc import MLP, FEATURES, LABEL_MAP, INV_LABEL_MAP


def load_val_df(path):
    df = pd.read_csv(path)
    return df


def get_logits_and_labels(model, scaler, df, device='cpu'):
    X = df[FEATURES].fillna(0.0).values.astype(np.float32)
    y = df['expert_action'].map(LABEL_MAP).values.astype(np.int64)
    if scaler is not None:
        X = scaler.transform(X)
    xb = torch.from_numpy(X).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(xb).cpu().numpy()
    return logits, y


def nll_loss_with_temp(logits, targets, temp):
    # logits: numpy ndarray (N, C)
    logits_t = logits / temp
    log_probs = logits_t - np.log(np.sum(np.exp(logits_t), axis=1, keepdims=True))
    nll = - np.mean(log_probs[np.arange(len(targets)), targets])
    return nll


def optimize_temperature(logits, targets, device='cpu'):
    # Implement using PyTorch for numeric stability and gradient
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)
    T = torch.nn.Parameter(torch.ones(1, dtype=torch.float32, device=device))
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled = logits_t / torch.clamp(T, min=1e-6)
        loss = loss_fn(scaled, targets_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_final = float(torch.clamp(T, min=1e-6).item())
    return T_final


def compute_ece(probs, labels, n_bins=10):
    # probs: (N, C), labels: (N,)
    preds = np.argmax(probs, axis=1)
    confidences = probs.max(axis=1)
    accuracies = (preds == labels).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        idx = (confidences > lo) & (confidences <= hi) if i>0 else (confidences >= lo) & (confidences <= hi)
        if not np.any(idx):
            bin_stats.append((i, 0, 0, 0.0, 0.0))
            continue
        avg_conf = float(np.mean(confidences[idx]))
        avg_acc = float(np.mean(accuracies[idx]))
        prop = float(idx.sum()) / len(labels)
        ece += prop * abs(avg_conf - avg_acc)
        bin_stats.append((i, idx.sum(), prop, avg_conf, avg_acc))
    return ece, bin_stats


def plot_reliability(probs, labels, out_png, n_bins=10, title='Reliability diagram'):
    ece, bins = compute_ece(probs, labels, n_bins=n_bins)
    xs = []
    ys = []
    ns = []
    for i, cnt, prop, avg_conf, avg_acc in bins:
        xs.append(avg_conf)
        ys.append(avg_acc)
        ns.append(cnt)
    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1], '--', color='gray')
    plt.scatter(xs, ys, s=np.array(ns)*10 + 5)
    for i,(x,y,n) in enumerate(zip(xs,ys,ns)):
        plt.text(x, y, str(n), fontsize=8, ha='right')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f"{title} (ECE={ece:.4f})")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return ece


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--val', default='data/splits/val.csv')
    p.add_argument('--model', default='models/bc/bc_model_best.pth')
    p.add_argument('--scaler', default='models/bc/scaler.pkl')
    p.add_argument('--out-dir', default='models/bc')
    p.add_argument('--bins', type=int, default=10)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load model & scaler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(in_dim=len(FEATURES))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()

    scaler = None
    if os.path.exists(args.scaler):
        scaler = pickle.load(open(args.scaler, 'rb'))

    val_df = load_val_df(args.val)
    logits, labels = get_logits_and_labels(model, scaler, val_df, device='cpu')

    # before calibration
    import scipy.special as ss
    probs_before = ss.softmax(logits, axis=1)
    ece_before, _ = compute_ece(probs_before, labels, n_bins=args.bins)

    # optimize temperature
    T_opt = optimize_temperature(logits, labels, device='cpu')

    # after calibration
    logits_scaled = logits / T_opt
    probs_after = ss.softmax(logits_scaled, axis=1)
    ece_after, _ = compute_ece(probs_after, labels, n_bins=args.bins)

    # save temperature
    with open(os.path.join(args.out_dir, 'temperature.txt'), 'w', encoding='utf-8') as f:
        f.write(str(T_opt))

    # plot reliability
    import os
    os.makedirs('logs', exist_ok=True)
    plot_reliability(probs_before, labels, 'logs/reliability_before.png', n_bins=args.bins, title='Reliability (before)')
    plot_reliability(probs_after, labels, 'logs/reliability_after.png', n_bins=args.bins, title='Reliability (after)')

    # print summary
    print('Temperature:', T_opt)
    print('ECE before:', ece_before)
    print('ECE after:', ece_after)
    print('Saved plots to logs/reliability_before.png and logs/reliability_after.png')
