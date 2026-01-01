"""
Quick test: compare BC model raw vs temperature-scaled probabilities on mismatch samples.
Usage:
  python scripts/test_apply_temperature.py --mismatch logs/model_mismatch.csv --model models/bc/bc_model_best.pth --scaler models/bc/scaler.pkl --temp models/bc/temperature.txt --out logs/temperature_test.csv --top 100
"""
import argparse
import pandas as pd
import numpy as np
import pickle
import torch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train_bc import MLP, FEATURES, LABEL_MAP, INV_LABEL_MAP


def load_temperature(path):
    if os.path.exists(path):
        try:
            return float(open(path,'r',encoding='utf-8').read().strip())
        except Exception:
            return 1.0
    return 1.0


def infer(model, scaler, feat, temp=1.0):
    X = np.array(feat, dtype=np.float32).reshape(1,-1)
    if scaler is not None:
        X = scaler.transform(X)
    xb = torch.from_numpy(X)
    model.eval()
    with torch.no_grad():
        logits = model(xb).cpu().numpy().ravel()
    from scipy.special import softmax
    probs_raw = softmax(logits)
    probs_scaled = softmax(logits / float(temp))
    return probs_raw, probs_scaled, logits


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mismatch', default='logs/model_mismatch.csv')
    p.add_argument('--model', default='models/bc/bc_model_best.pth')
    p.add_argument('--scaler', default='models/bc/scaler.pkl')
    p.add_argument('--temp', default='models/bc/temperature.txt')
    p.add_argument('--out', default='logs/temperature_test.csv')
    p.add_argument('--top', type=int, default=100)
    p.add_argument('--model-conf-high', type=float, default=0.95, help='模型置信度高阈值，用于模拟运行时是否采信模型')
    args = p.parse_args()

    df = pd.read_csv(args.mismatch)
    df = df.head(args.top)

    model = MLP(in_dim=len(FEATURES))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    scaler = None
    if os.path.exists(args.scaler):
        scaler = pickle.load(open(args.scaler, 'rb'))

    T = load_temperature(args.temp)

    rows = []
    debug_rows = []
    # helper to normalize model_action field (handles Chinese labels)
    MAP_CHINESE = {'减速':'dec', '加速':'acc', '保持':'keep', 'dec':'dec', 'acc':'acc', 'keep':'keep'}
    MAP_RULE = {'减速':'dec', '加速':'acc', '保持':'keep', 'dec':'dec', 'acc':'acc', 'keep':'keep'}

    def safe_float(v, default=0.0):
        try:
            if v is None:
                return default
            s = str(v).strip()
            if s == '' or s.lower() == 'nan' or s.lower() == 'none':
                return default
            return float(s)
        except Exception:
            return default

    for idx, r in df.iterrows():
        feat = [
            safe_float(r.get('avg_hr')),
            safe_float(r.get('hr_slope')),
            safe_float(r.get('avg_spo2')),
            safe_float(r.get('speed_mps')),
            safe_float(r.get('speed_trend')),
            safe_float(r.get('angleX_mean')),
            safe_float(r.get('angleX_std')),
            safe_float(r.get('angleY_mean')),
            safe_float(r.get('angleY_std')),
            safe_float(r.get('target_speed')),
            safe_float(r.get('fatigue_score')),
        ]
        probs_raw, probs_scaled, logits = infer(model, scaler, feat, temp=T)

        # detect NaN/Inf in logits
        logits_arr = np.array(logits, dtype=np.float64)
        has_nan = np.any(np.isnan(logits_arr)) or np.any(np.isinf(logits_arr))
        if has_nan:
            debug_rows.append({
                'idx': idx,
                'window_start_ms': r.get('window_start_ms'),
                'feat': feat,
                'logits': logits_arr.tolist(),
                'temp': float(T)
            })

        pred_raw = int(np.nanargmax(probs_raw)) if not np.all(np.isnan(probs_raw)) else -1
        pred_scaled = int(np.nanargmax(probs_scaled)) if not np.all(np.isnan(probs_scaled)) else -1
        # map original model_action to canonical
        orig = (r.get('model_action') or '').strip()
        orig_map = MAP_CHINESE.get(orig, orig)

        # simulate runtime conservative check: require conf >= model_conf_high and direction consistency
        model_conf_val = float(r.get('model_conf')) if r.get('model_conf') not in [None,'','nan'] else float(np.nan)
        rule_raw = (r.get('rule_action') or '').strip()
        rule_map = MAP_RULE.get(rule_raw, rule_raw)
        cur_speed = safe_float(r.get('speed_mps'), default=np.nan)
        target_speed = safe_float(r.get('target_speed'), default=np.nan)

        # default: don't trust model
        simulated_use = 'rule'
        simulated_rejected_direction = False
        if not np.isnan(model_conf_val) and model_conf_val >= float(args.model_conf_high):
            # direction check
            ma = orig_map
            if ma == 'dec' and not (np.isnan(cur_speed) or np.isnan(target_speed)) and not (cur_speed < 0.9 * target_speed):
                simulated_use = 'model'
            elif ma == 'acc' and not (np.isnan(cur_speed) or np.isnan(target_speed)) and not (cur_speed > 1.1 * target_speed):
                simulated_use = 'model'
            elif ma == 'keep':
                simulated_use = 'model'
            else:
                simulated_rejected_direction = True
                simulated_use = 'rule'

        rows.append({
            'idx': idx,
            'window_start_ms': r.get('window_start_ms'),
            'speed_mps': r.get('speed_mps'),
            'rule_action': rule_raw,
            'model_action_orig': orig,
            'model_action_orig_canon': orig_map,
            'model_conf_orig': safe_float(r.get('model_conf'), default=np.nan),
            'pred_raw': int(pred_raw),
            'conf_raw': float(np.nanmax(probs_raw)) if not np.all(np.isnan(probs_raw)) else np.nan,
            'probs_raw': ','.join([f"{x:.6f}" for x in probs_raw]) if not np.all(np.isnan(probs_raw)) else 'nan',
            'pred_scaled': int(pred_scaled),
            'conf_scaled': float(np.nanmax(probs_scaled)) if not np.all(np.isnan(probs_scaled)) else np.nan,
            'probs_scaled': ','.join([f"{x:.6f}" for x in probs_scaled]) if not np.all(np.isnan(probs_scaled)) else 'nan',
            'temp': float(T),
            'simulated_use': simulated_use,
            'simulated_rejected_direction': simulated_rejected_direction
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False, encoding='utf-8')

    # write debug rows if any
    if debug_rows:
        dbg_df = pd.DataFrame(debug_rows)
        dbg_path = 'logs/temperature_test_debug.csv'
        dbg_df.to_csv(dbg_path, index=False, encoding='utf-8')
        print(f"Detected {len(debug_rows)} rows with NaN/Inf logits; wrote details to {dbg_path}")
    else:
        print('No NaN/Inf logits detected')

    # summary
    changed = (out_df['pred_raw'] != out_df['pred_scaled']).sum()
    mean_conf_raw = float(np.nanmean(out_df['conf_raw'].values))
    mean_conf_scaled = float(np.nanmean(out_df['conf_scaled'].values))
    print(f"Samples: {len(out_df)}; changed predictions after temp scaling: {changed}")
    print(f"Mean conf before: {mean_conf_raw:.4f}; after: {mean_conf_scaled:.4f}")

    # additional summary for simulated runtime use
    by_use = out_df['simulated_use'].value_counts().to_dict()
    print('Simulated decision use counts:', by_use)
    print('Simulated rejected by direction:', out_df['simulated_rejected_direction'].sum())
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False, encoding='utf-8')

    # write debug rows if any
    if debug_rows:
        dbg_df = pd.DataFrame(debug_rows)
        dbg_path = 'logs/temperature_test_debug.csv'
        dbg_df.to_csv(dbg_path, index=False, encoding='utf-8')
        print(f"Detected {len(debug_rows)} rows with NaN/Inf logits; wrote details to {dbg_path}")
    else:
        print('No NaN/Inf logits detected')

    # summary
    changed = (out_df['pred_raw'] != out_df['pred_scaled']).sum()
    mean_conf_raw = float(np.nanmean(out_df['conf_raw'].values))
    mean_conf_scaled = float(np.nanmean(out_df['conf_scaled'].values))
    print(f"Samples: {len(out_df)}; changed predictions after temp scaling: {changed}")
    print(f"Mean conf before: {mean_conf_raw:.4f}; after: {mean_conf_scaled:.4f}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False, encoding='utf-8')
    print('Saved test results to', args.out)
