"""
加载已训练的 BC 模型并在指定数据集上评估
用法：
  python scripts/evaluate_bc.py --model models/bc/bc_model_final.pth --scaler models/bc/scaler.pkl --data data/splits/test.csv
"""
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn as nn
from train_bc import MLP, FEATURES, LABEL_MAP, INV_LABEL_MAP


def load_scaler(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_model(path, device='cpu', hidden=128, dropout=0.3):
    model = MLP(in_dim=len(FEATURES), hidden=hidden, n_classes=3, dropout=dropout)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate(model, scaler, df, device='cpu'):
    X = df[FEATURES].fillna(0.0).values.astype(np.float32)
    X = scaler.transform(X)
    y_true = df['expert_action'].map(LABEL_MAP).values
    import torch
    with torch.no_grad():
        Xb = torch.from_numpy(X).to(device)
        logits = model(Xb).cpu().numpy()
        y_pred = np.argmax(logits, axis=1)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2]).tolist()
    return {'accuracy':float(acc),'precision':[float(x) for x in prec],'recall':[float(x) for x in rec],'f1':[float(x) for x in f1],'confusion_matrix':cm}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--scaler', required=True)
    p.add_argument('--data', required=True)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    scaler = load_scaler(args.scaler)
    model = load_model(args.model, device='cpu')

    res = evaluate(model, scaler, df, device='cpu')
    print('Evaluation result:')
    print(json.dumps(res, indent=2))

