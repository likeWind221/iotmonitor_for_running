"""Replay CSV windows and run sequence model inference for smoke testing.

Usage:
  python scripts/sim_inference.py --csv data/splits/train_small.csv --model models/seq_full/gru_best.pth \
      --scaler models/seq_full/scaler.pkl --label-map models/seq_full/label_map.json --seq-len 12 --limit 200
"""
import argparse
import pandas as pd
import numpy as np
import pickle
import json
import torch
from collections import deque, Counter
import os
import sys
# ensure project root is importable for importing scripts.train_sequence
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

SEQ_FEATURES = ['avg_hr','hr_slope','avg_spo2','speed_mps','speed_trend','angleX_mean','angleX_std','angleY_mean','angleY_std','fatigue_score']


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--scaler', required=True)
    p.add_argument('--label-map', required=True)
    p.add_argument('--seq-len', type=int, default=12)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df.sort_values(['session_id','window_start_ms'])
    print('Rows:', len(df))

    # load artifacts
    scaler = pickle.load(open(args.scaler, 'rb'))
    with open(args.label_map, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    inv_label = {int(v): k for k, v in label_map.items()}

    from scripts.train_sequence import GRUClassifier
    n_classes = len(inv_label)
    model = GRUClassifier(input_dim=len(SEQ_FEATURES), hidden_dim=128, num_classes=n_classes)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()

    buf = deque(maxlen=args.seq_len)

    printed = 0
    pred_counts = Counter()
    confs = []

    for i, row in df.iterrows():
        # build feature
        feat = [safe_float(row.get(c, 0.0)) for c in SEQ_FEATURES]
        buf.append(feat)
        if len(buf) == args.seq_len:
            arr = np.array(buf, dtype=np.float32)
            try:
                arr_norm = scaler.transform(arr)
            except Exception as e:
                print('Scaler transform failed:', e)
                arr_norm = arr
            xb = torch.from_numpy(arr_norm.reshape(1, args.seq_len, -1)).float()
            with torch.no_grad():
                logits = model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
            pred = int(probs.argmax())
            conf = float(probs.max())
            pred_label = inv_label.get(pred, str(pred))
            pred_counts[pred_label] += 1
            confs.append(conf)
            # print a sample line with context
            if printed < args.limit or args.limit == 0:
                print(f"window_start_ms={int(row.window_start_ms)} session={int(row.session_id)} true={row.get('current_mode_name','')} pred={pred_label} conf={conf:.3f}")
                printed += 1
            if args.limit > 0 and printed >= args.limit:
                break

    print('\nSummary:')
    print('Pred counts:', dict(pred_counts))
    if confs:
        print('Conf mean/std/min/max:', np.mean(confs), np.std(confs), np.min(confs), np.max(confs))


if __name__ == '__main__':
    main()
