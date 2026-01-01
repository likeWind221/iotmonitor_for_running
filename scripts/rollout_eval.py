"""
在线策略 Rollout 评估（仿真环境）
用法：
  python scripts/rollout_eval.py --model models/bc/bc_model_final.pth --scaler models/bc/scaler.pkl --data data/splits/test.csv --out results/rollout.json

说明：
- 使用训练好的 BC 模型做推理（可选概率阈值和一致性平滑），并在模拟环境中执行动作
- 简单动力学：speed += delta_acc/dec per window（参数可调）
- 使用 generate_training_data.reward_for_window 来计算每步 reward
- 输出每 session 的累计 reward、安全违例、动作切换率的统计汇总
"""
import os
import sys
# Ensure project root is in sys.path so modules like generate_training_data can be imported when running from /scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
import numpy as np
import pandas as pd
import torch
import pickle
from collections import defaultdict

from train_bc import MLP, FEATURES, LABEL_MAP, INV_LABEL_MAP
from generate_training_data import reward_for_window


def load_scaler(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_model(path, device='cpu', hidden=128, dropout=0.3):
    model = MLP(in_dim=len(FEATURES), hidden=hidden, n_classes=3, dropout=dropout)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def infer_action(model, scaler, features, device='cpu', prob_thresh=0.0):
    x = np.array(features, dtype=np.float32).reshape(1,-1)
    x = scaler.transform(x)
    xb = torch.from_numpy(x).to(device)
    with torch.no_grad():
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
    max_p = float(probs.max())
    act = int(probs.argmax())
    if max_p < prob_thresh:
        # fallback to keep
        act = LABEL_MAP['keep']
    return INV_LABEL_MAP[act], max_p


def run_rollout(df, model, scaler, args):
    results = []
    sessions = df['session_id'].unique()
    if args.limit:
        sessions = sessions[:args.limit]

    dump_ids = set()
    if getattr(args, 'dump_sessions', ''):
        try:
            dump_ids = set(int(x.strip()) for x in args.dump_sessions.split(',') if x.strip())
        except Exception:
            dump_ids = set()

    windows_dump = {}

    for sid in sessions:
        s_df = df[df['session_id']==sid].sort_values('window_start_ms')
        current_speed = float(s_df['speed_mps'].iloc[0]) if not np.isnan(s_df['speed_mps'].iloc[0]) else 1.0
        last_action = None
        last_pred = None
        pred_consec = 0
        last_exec_change_time_ms = None
        cum_reward = 0.0
        safety_violations = 0
        action_changes = 0
        actions = []

        per_window_logs = []

        for i, row in s_df.iterrows():
            # build features: use current speed as speed_mps
            feat = [
                row['avg_hr'],
                row['hr_slope'],
                row['avg_spo2'],
                current_speed,
                0.0, # speed_trend unknown here
                row['angleX_mean'],
                row['angleX_std'],
                row['angleY_mean'],
                row['angleY_std'],
                row['fatigue_score']
            ]
            hr = row['avg_hr']
            spo2 = row['avg_spo2']
            target_speed = row['target_speed']
            max_hr = 220 - args.age
            hr_ratio = None
            if not np.isnan(hr):
                hr_ratio = (hr - args.resting_hr) / max(1, (max_hr - args.resting_hr))

            # safety override
            override_dec = False
            if spo2 is not None and spo2 < args.spo2_threshold:
                override_dec = True
            if hr_ratio is not None and hr_ratio > args.hr_ratio_threshold:
                override_dec = True

            action, conf = infer_action(model, scaler, feat, device=args.device, prob_thresh=args.prob_thresh)
            # apply override
            if override_dec:
                action = 'dec'
                conf = 1.0

            # safety override: immediate dec if thresholds crossed
            if override_dec:
                exec_action = 'dec'
                conf = 1.0
                # allow immediate change on safety
                if last_action is None or exec_action != last_action:
                    last_exec_change_time_ms = int(row['window_start_ms'])
            else:
                # update predicted consecutive count
                if last_pred is None or action == last_pred:
                    pred_consec += 1
                else:
                    pred_consec = 1
                last_pred = action

                # decide executed action based on consensus and min-change-ms
                if last_action is None:
                    # first step: accept action only if consensus satisfied, else fallback to 'keep'
                    if pred_consec >= args.consensus:
                        exec_action = action
                        last_exec_change_time_ms = int(row['window_start_ms'])
                    else:
                        exec_action = 'keep'
                else:
                    if pred_consec >= args.consensus:
                        window_start_ms = int(row['window_start_ms'])
                        if args.min_change_ms > 0 and last_exec_change_time_ms is not None and (window_start_ms - last_exec_change_time_ms) < args.min_change_ms:
                            exec_action = last_action
                        else:
                            exec_action = action
                            if exec_action != last_action:
                                last_exec_change_time_ms = window_start_ms
                    else:
                        exec_action = last_action

            if last_action is not None and exec_action != last_action:
                action_changes += 1
            last_action = exec_action
            actions.append(exec_action)

            # safety violation: action=acc when hr_ratio>threshold or spo2 low
            if exec_action == 'acc' and ((spo2 is not None and spo2 < args.spo2_threshold) or (hr_ratio is not None and hr_ratio > args.hr_ratio_threshold)):
                safety_violations += 1

            # compute reward using reward_for_window (based on current_speed and exec_action)
            r = reward_for_window(current_speed, target_speed, exec_action, spo2, row['fatigue_score'], hr_ratio)
            cum_reward += r

            # collect per-window log if requested
            if sid in dump_ids:
                per_window_logs.append({
                    'window_start_ms': int(row['window_start_ms']),
                    'window_end_ms': int(row['window_end_ms']),
                    'avg_hr': row['avg_hr'],
                    'avg_spo2': row['avg_spo2'],
                    'target_speed': target_speed,
                    'current_speed': round(current_speed,3),
                    'model_action': action,
                    'model_conf': round(conf,3),
                    'exec_action': exec_action,
                    'reward': round(r,3)
                })

            # update simulated speed
            if exec_action == 'acc':
                current_speed = current_speed + args.acc_delta
            elif exec_action == 'dec':
                current_speed = max(0.1, current_speed - args.dec_delta)
            else:
                # keep: small drift toward target
                current_speed = current_speed + (target_speed - current_speed) * 0.05

        # compute per-session metrics
        session_len = len(s_df)
        results.append({
            'session_id': int(sid),
            'session_mode': s_df['session_mode'].iloc[0],
            'cum_reward': float(cum_reward),
            'safety_violations': int(safety_violations),
            'action_changes': int(action_changes),
            'action_change_rate': float(action_changes) / max(1, session_len),
            'mean_speed': float(np.mean([float(x) if not np.isnan(x) else 0.0 for x in s_df['speed_mps']])),
            'actions_counts': {k: int(v) for k,v in pd.Series(actions).value_counts().to_dict().items()}
        })

        if sid in dump_ids and args.save_windows:
            windows_dump[int(sid)] = per_window_logs

    return results, windows_dump


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--scaler', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='results/rollout.json')
    p.add_argument('--limit', type=int, default=None, help='限制会话数量以便快速测试')
    p.add_argument('--device', default='cpu')
    p.add_argument('--prob-thresh', type=float, default=0.0, help='预测置信度阈值，低于阈值 fallback to keep')
    p.add_argument('--consensus', type=int, default=1, help='需要连续几次相同动作才执行（平滑）')
    p.add_argument('--min-change-ms', type=int, default=0, help='最小允许动作切换时间间隔（毫秒），小于该间隔的切换会被忽略')
    p.add_argument('--acc-delta', type=float, default=0.3, help='执行 acc 时每步速度增加值 (m/s)')
    p.add_argument('--dec-delta', type=float, default=0.5, help='执行 dec 时每步速度减少值 (m/s)')
    p.add_argument('--spo2-threshold', type=float, default=92.0)
    p.add_argument('--hr-ratio-threshold', type=float, default=0.95)
    p.add_argument('--age', type=int, default=25)
    p.add_argument('--resting-hr', type=int, default=60)
    p.add_argument('--dump-sessions', type=str, default='', help='逗号分隔的 session_id 列表，用于导出逐窗口日志，例如 "122,20,14"')
    p.add_argument('--save-windows', action='store_true', help='若设置则把逐窗口诊断数据包含到输出 JSON')
    p.add_argument('--diag-dir', default='results/diag', help='导出逐窗口 CSV/PNG 的目录')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.data)
    scaler = load_scaler(args.scaler)
    model = load_model(args.model, device=args.device)

    results, windows_dump = run_rollout(df, model, scaler, args)
    # summarize
    import statistics
    cum_rewards = [r['cum_reward'] for r in results]
    safety = [r['safety_violations'] for r in results]
    action_rates = [r['action_change_rate'] for r in results]
    per_mode = {}
    for r in results:
        per_mode.setdefault(r['session_mode'], []).append(r)

    summary = {
        'n_sessions': len(results),
        'cum_reward_mean': statistics.mean(cum_rewards),
        'cum_reward_std': statistics.pstdev(cum_rewards) if len(cum_rewards)>1 else 0.0,
        'safety_violations_mean': statistics.mean(safety),
        'action_change_rate_mean': statistics.mean(action_rates),
        'per_mode': {}
    }
    for mode, rs in per_mode.items():
        summary['per_mode'][mode] = {
            'n': len(rs),
            'cum_reward_mean': statistics.mean([x['cum_reward'] for x in rs]),
            'safety_violations_mean': statistics.mean([x['safety_violations'] for x in rs]),
            'action_change_rate_mean': statistics.mean([x['action_change_rate'] for x in rs])
        }

    out_content = {'summary': summary, 'sessions': results[:200]}
    if windows_dump:
        out_content['windows_dump'] = windows_dump

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out_content, f, ensure_ascii=False, indent=2)

    # If requested, also persist per-window CSVs and diagnostic plots per session
    if windows_dump and args.save_windows:
        try:
            os.makedirs(args.diag_dir, exist_ok=True)
            import matplotlib.pyplot as plt
            for sid, logs in windows_dump.items():
                try:
                    dfw = pd.DataFrame(logs)
                    csv_path = os.path.join(args.diag_dir, f'session_{sid}_windows.csv')
                    dfw.to_csv(csv_path, index=False)

                    # create plots: speed / hr / spo2 / conf+actions
                    fig, axes = plt.subplots(4,1, figsize=(10,9), sharex=True)
                    axes[0].plot(dfw['window_start_ms'], dfw['target_speed'], linestyle='--', label='target')
                    axes[0].plot(dfw['window_start_ms'], dfw['current_speed'], label='current')
                    axes[0].set_ylabel('speed (m/s)')
                    axes[0].legend()

                    axes[1].plot(dfw['window_start_ms'], dfw['avg_hr'], color='C1')
                    axes[1].set_ylabel('HR')

                    axes[2].plot(dfw['window_start_ms'], dfw['avg_spo2'], color='C2')
                    axes[2].set_ylabel('SpO2')

                    axes[3].plot(dfw['window_start_ms'], dfw['model_conf'], color='C3', label='model_conf')
                    action_map = {'acc':1, 'keep':0, 'dec':-1}
                    axes[3].scatter(dfw['window_start_ms'], [action_map.get(a,0) for a in dfw['exec_action']], c='k', s=18, label='exec_action (marker)')
                    axes[3].set_ylabel('conf / exec_action')
                    axes[3].legend()

                    axes[-1].set_xlabel('window_start_ms')
                    plt.tight_layout()
                    png_path = os.path.join(args.diag_dir, f'session_{sid}_diag.png')
                    fig.savefig(png_path)
                    plt.close(fig)
                except Exception as ee:
                    print('Failed to export diag for session', sid, ee)
        except Exception as e:
            print('Failed to write diagnostic CSV/plots', e)

    print('Rollout summary:')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if windows_dump:
        print('Saved per-window diagnostics for sessions:', list(windows_dump.keys()))
    print('Saved detail to', args.out)
