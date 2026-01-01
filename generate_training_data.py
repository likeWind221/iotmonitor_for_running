"""
生成离线训练数据（5s窗口）
功能：
- 从现有的 CSV（simulated_data/gyro_data_*.csv 与 physio_data_*.csv）读取最新文件
- 按非重叠 5s 窗口聚合特征（avg_hr, hr_slope, avg_spo2, speed_mps, angle stats 等）
- 生成 fatigue_score、quality_flag
- 使用规则化专家策略生成 expert_action（减速/保持/加速）和 reward
- 输出为 CSV，便于离线训练（行为克隆或离线 RL）
- 合成会话现在支持会话内混合模式切换（随机切换与平滑过渡，启用请使用 --synth-mixed）

用法示例：
  python generate_training_data.py --mode file --data-dir simulated_data --out data/train_windows.csv

"""
import os
import glob
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

TARGET_SPEED = {0:1.0, 1:2.5, 2:5.0}
MODE_NAME = {0:'walk',1:'jog',2:'sprint'}


def get_latest_files(data_dir):
    gyro_files = sorted([os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.startswith('gyro_data_') and f.endswith('.csv')])
    physio_files = sorted([os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.startswith('physio_data_') and f.endswith('.csv')])
    if not gyro_files or not physio_files:
        raise FileNotFoundError('未找到数据文件，请确保目录下有 gyro_data_*.csv 和 physio_data_*.csv')
    return gyro_files[-1], physio_files[-1]


def load_and_seconds(gyro_path, physio_path):
    gyro = pd.read_csv(gyro_path)
    physio = pd.read_csv(physio_path)
    # Convert to int seconds
    gyro['timestamp_s'] = (gyro['timestamp_ms'] // 1000).astype(int)
    physio['timestamp_s'] = (physio['timestamp_ms'] // 1000).astype(int)
    return gyro, physio


def make_windows(gyro_df, physio_df, window_s=5):
    # Merge per second as in live_pace_rl
    agg = gyro_df.groupby('timestamp_s').agg({'gyro_x':['mean','std'],'gyro_y':['mean','std']})
    agg.columns = ['gyro_x_mean','gyro_x_std','gyro_y_mean','gyro_y_std']
    agg = agg.reset_index()
    physio_sec = physio_df.sort_values('timestamp_s').drop_duplicates('timestamp_s')
    aligned = physio_sec.merge(agg, on='timestamp_s', how='inner')
    aligned = aligned.sort_values('timestamp_s').reset_index(drop=True)

    start = aligned['timestamp_s'].min()
    end = aligned['timestamp_s'].max()
    windows = []
    t = start
    prev_hr = None
    prev_speed = None
    while t + window_s <= end:
        w = aligned[(aligned['timestamp_s'] >= t) & (aligned['timestamp_s'] < t + window_s)]
        if not w.empty:
            window_start_ms = int(w['timestamp_s'].min() * 1000)
            window_end_ms = int(w['timestamp_s'].max() * 1000 + 999)
            count_mpu = int(w['gyro_x_mean'].count())
            count_data = int(w['heart_rate'].count())
            avg_hr = float(w['heart_rate'].mean()) if count_data>0 else np.nan
            avg_spo2 = float(w['blood_oxygen'].mean()) if count_data>0 else np.nan
            angleX_mean = float(w['gyro_x_mean'].mean()) if count_mpu>0 else np.nan
            angleX_std = float(w['gyro_x_std'].mean()) if count_mpu>0 else np.nan
            angleY_mean = float(w['gyro_y_mean'].mean()) if count_mpu>0 else np.nan
            angleY_std = float(w['gyro_y_std'].mean()) if count_mpu>0 else np.nan
            # speed from distance_m (per-second distance cumulative)
            # use first and last distance_m in window
            first_dist = w['distance_m'].iloc[0]
            last_dist = w['distance_m'].iloc[-1]
            delta_m = last_dist - first_dist
            delta_t = max(1, (w['timestamp_s'].iloc[-1] - w['timestamp_s'].iloc[0]))
            speed_mps = float(delta_m / delta_t) if delta_t>0 else np.nan

            # hr_slope and speed_trend using previous window values
            hr_slope = float(avg_hr - prev_hr) if prev_hr is not None and not np.isnan(avg_hr) and not np.isnan(prev_hr) else 0.0
            speed_trend = float(speed_mps - prev_speed) if prev_speed is not None and not np.isnan(speed_mps) and not np.isnan(prev_speed) else 0.0

            # include mode info when available
            current_mode = w['current_mode'].iloc[0] if 'current_mode' in w.columns else None
            windows.append({
                'window_start_ms':window_start_ms,
                'window_end_ms':window_end_ms,
                'count_mpu':count_mpu,
                'count_data':count_data,
                'avg_hr':None if np.isnan(avg_hr) else round(avg_hr,2),
                'hr_slope':round(hr_slope,3),
                'avg_spo2':None if np.isnan(avg_spo2) else round(avg_spo2,2),
                'speed_mps':None if np.isnan(speed_mps) else round(speed_mps,3),
                'speed_trend':round(speed_trend,3),
                'angleX_mean':None if np.isnan(angleX_mean) else round(angleX_mean,3),
                'angleX_std':None if np.isnan(angleX_std) else round(angleX_std,3),
                'angleY_mean':None if np.isnan(angleY_mean) else round(angleY_mean,3),
                'angleY_std':None if np.isnan(angleY_std) else round(angleY_std,3),
                'current_mode': int(current_mode) if current_mode is not None else None,
                'current_mode_name': MODE_PARAMS[int(current_mode)]['name'] if current_mode is not None else None,
            })
            prev_hr = avg_hr if not np.isnan(avg_hr) else prev_hr
            prev_speed = speed_mps if not np.isnan(speed_mps) else prev_speed
        t += window_s
    return windows


def fatigue_score_from_window(avg_hr, hr_slope, angle_std, age=25, resting_hr=60):
    # Simple normalized fatigue proxy
    max_hr = 220 - age
    hr_rel = 0.0
    if avg_hr is not None:
        hr_rel = (avg_hr - resting_hr) / max(1, (max_hr - resting_hr))
        hr_rel = min(max(hr_rel, 0.0), 1.0)
    # normalize angle_std assuming typical range 0-20
    ang_norm = min(max(angle_std / 20.0, 0.0), 1.0)
    # hr_slope normalized (clamp to -5..5 -> map to 0..1 using positive slope only for fatigue)
    slope_norm = min(max(hr_slope / 5.0, 0.0), 1.0)
    # weights
    w1, w2, w3 = 0.5, 0.3, 0.2
    score = w1 * hr_rel + w2 * slope_norm + w3 * ang_norm
    return round(min(max(score, 0.0), 1.0), 3)


def expert_rule_action(speed, target_speed, spo2, fatigue, hr_ratio):
    # Safety override
    if spo2 is not None and spo2 < 92:
        return 'dec'
    if hr_ratio is not None and hr_ratio > 0.95:
        return 'dec'
    # speed relative rule
    if speed is None:
        return 'keep'
    if speed < target_speed * 0.9:
        return 'acc'
    if speed > target_speed * 1.1:
        return 'dec'
    # fatigue bias
    if fatigue > 0.7 and speed > target_speed*0.95:
        return 'dec'
    return 'keep'


def reward_for_window(speed, target_speed, action, spo2, fatigue, hr_ratio):
    # base reward: closer speed -> higher
    if speed is None:
        r_speed = 0
    else:
        err = abs(speed - target_speed) / max(target_speed, 1e-6)
        r_speed = 10 * (1 - min(1.0, err))
    r = r_speed
    # safety penalties
    if spo2 is not None and spo2 < 92:
        r -= 50
    if hr_ratio is not None and hr_ratio > 0.95 and action == 'acc':
        r -= 30
    # fatigue penalty for acc
    if fatigue > 0.7 and action == 'acc':
        r -= 10
    # prefer keep action small bonus
    if action == 'keep':
        r += 2
    return round(float(r),3)


def process_files(gyro_path, physio_path, out_csv, window_s=5, age=25, resting_hr=60):
    gyro, physio = load_and_seconds(gyro_path, physio_path)
    windows = make_windows(gyro, physio, window_s=window_s)
    rows = []
    for w in windows:
        # infer target_mode by nearest target speed if speed exists
        speed = w['speed_mps']
        if speed is None:
            target_mode = 1
            target_speed = TARGET_SPEED[target_mode]
        else:
            # pick nearest target
            diffs = {m:abs(speed - s) for m,s in TARGET_SPEED.items()}
            target_mode = min(diffs, key=diffs.get)
            target_speed = TARGET_SPEED[target_mode]
        # fatigue
        angle_std = (w['angleX_std'] or 0.0) + (w['angleY_std'] or 0.0)
        fatigue = fatigue_score_from_window(w['avg_hr'], w['hr_slope'], angle_std, age=age, resting_hr=resting_hr)
        # hr_ratio
        max_hr = 220 - age
        hr_ratio = None
        if w['avg_hr'] is not None:
            hr_ratio = (w['avg_hr'] - resting_hr) / max(1, (max_hr - resting_hr))
        action = expert_rule_action(w['speed_mps'], target_speed, w['avg_spo2'], fatigue, hr_ratio)
        reward = reward_for_window(w['speed_mps'], target_speed, action, w['avg_spo2'], fatigue, hr_ratio)
        quality_flag = 0
        if w['count_data'] < 1 or w['count_mpu'] < 1:
            quality_flag = 1
        rows.append({
            **w,
            'target_mode':MODE_NAME[target_mode],
            'target_speed':target_speed,
            'fatigue_score':fatigue,
            'hr_ratio':round(hr_ratio,3) if hr_ratio is not None else None,
            'expert_action':action,
            'reward':reward,
            'quality_flag':quality_flag
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"已生成训练数据：{out_csv}，包含 {len(df)} 个窗口")


# ------------- 合成数据配置（用于快速生成大规模训练集） -------------
MODE_PARAMS = {
    0: {'name':'walk','hr_range':(70,90),'spo2_range':(96,99),'speed':1.0,'gyro_freq':0.5,'x_range':(-18.2,-9.9),'y_range':(152.63,166.91)},
    1: {'name':'jog','hr_range':(100,130),'spo2_range':(94,97),'speed':2.5,'gyro_freq':1.5,'x_range':(-25,-15),'y_range':(145,160)},
    2: {'name':'sprint','hr_range':(150,190),'spo2_range':(90,95),'speed':5.0,'gyro_freq':3.5,'x_range':(-35,-25),'y_range':(130,145)}
}


def lerp(a, b, t):
    return a + (b - a) * t


def get_blended_params(current_mode, target_mode, t):
    if t >= 1.0:
        return MODE_PARAMS[target_mode]
    src = MODE_PARAMS[current_mode]
    dst = MODE_PARAMS[target_mode]
    return {
        'name': dst['name'],
        'hr_range': (
            int(lerp(src['hr_range'][0], dst['hr_range'][0], t)),
            int(lerp(src['hr_range'][1], dst['hr_range'][1], t))
        ),
        'spo2_range': (
            int(lerp(src['spo2_range'][0], dst['spo2_range'][0], t)),
            int(lerp(src['spo2_range'][1], dst['spo2_range'][1], t))
        ),
        'speed': lerp(src['speed'], dst['speed'], t),
        'gyro_freq': lerp(src['gyro_freq'], dst['gyro_freq'], t),
        'x_range': (
            lerp(src['x_range'][0], dst['x_range'][0], t),
            lerp(src['x_range'][1], dst['x_range'][1], t)
        ),
        'y_range': (
            lerp(src['y_range'][0], dst['y_range'][0], t),
            lerp(src['y_range'][1], dst['y_range'][1], t)
        )
    }


def synthesize_session(mode_id, duration_s=300, sample_rate_mpu=10, seed=None, speed_perturb=0.0, mixed=False, switch_min=10, switch_max=30, transition_s=5):
    """生成单次合成会话的原始数据 DataFrames（gyro_df, physio_df）。
    支持 mixed=True 来在会话内随机切换模式，switch_min/switch_max 指切换间隔范围（秒），transition_s 指一次过渡持续秒数。
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    params = MODE_PARAMS[mode_id]
    times = np.arange(duration_s)

    # prepare containers
    hr_vals = []
    spo2_vals = []
    speeds = np.zeros(duration_s, dtype=float)
    mode_by_second = [mode_id] * duration_s

    current_mode = mode_id
    target_mode = mode_id
    next_switch_time = None
    transition_progress = 1.0

    for t in times:
        # decide blending params for this second
        if mixed:
            if next_switch_time is None:
                next_switch_time = t + rng.randint(switch_min, switch_max)
            if t >= next_switch_time and transition_progress >= 1.0:
                available = [m for m in MODE_PARAMS.keys() if m != target_mode]
                current_mode = target_mode
                target_mode = int(rng.choice(available))
                transition_progress = 0.0
                next_switch_time = t + rng.randint(switch_min, switch_max)
            # increment transition progress (per-second step)
            if transition_progress < 1.0:
                transition_progress = min(1.0, transition_progress + 1.0 / max(1, transition_s))
            blended = get_blended_params(current_mode, target_mode, transition_progress)
            params_t = blended
            # mark dominant mode for this second
            if transition_progress >= 0.5:
                mode_by_second[t] = target_mode
            else:
                mode_by_second[t] = current_mode
        else:
            params_t = params
            mode_by_second[t] = current_mode

        # per-second hr and spo2
        hr = rng.randint(params_t['hr_range'][0], params_t['hr_range'][1]+1)
        spo2 = rng.randint(params_t['spo2_range'][0], params_t['spo2_range'][1]+1)
        hr_vals.append(hr)
        spo2_vals.append(spo2)

        # per-second speed with optional perturb
        if speed_perturb and speed_perturb > 0.0:
            sp = params_t['speed'] * (1 + rng.normal(0.0, speed_perturb))
            sp = float(max(0.1, sp))
        else:
            sp = float(params_t['speed'])
        speeds[t] = sp

    # cumulative distance
    distance = np.cumsum(speeds)

    physio_rows = []
    for i, t in enumerate(times):
        physio_rows.append({'timestamp_ms': int(t*1000), 'heart_rate': int(hr_vals[i]), 'blood_oxygen': int(spo2_vals[i]), 'distance_m': float(distance[i]), 'current_mode': int(mode_by_second[i])})
    physio_df = pd.DataFrame(physio_rows)

    # MPU 数据：每秒 sample_rate_mpu 个样本，使用每秒的模式信息在秒内进行额外的线性插值以平滑
    mpu_rows = []
    for i, t in enumerate(times):
        for k in range(sample_rate_mpu):
            frac = k / sample_rate_mpu
            elapsed = t + frac
            if mixed:
                mode_a = mode_by_second[t]
                mode_b = mode_by_second[t+1] if t+1 < duration_s else mode_by_second[t]
                params_fraction = get_blended_params(mode_a, mode_b, frac)
            else:
                params_fraction = params
            angleX = (np.sin(2 * np.pi * params_fraction['gyro_freq'] * elapsed) * ((params_fraction['x_range'][1] - params_fraction['x_range'][0])/4.0)) + ((params_fraction['x_range'][0] + params_fraction['x_range'][1]) / 2.0) + rng.uniform(-0.5,0.5)
            angleY = (np.cos(2 * np.pi * params_fraction['gyro_freq'] * elapsed) * ((params_fraction['y_range'][1] - params_fraction['y_range'][0])/4.0)) + ((params_fraction['y_range'][0] + params_fraction['y_range'][1]) / 2.0) + rng.uniform(-0.8,0.8)
            ts = int((t + frac) * 1000)
            mpu_rows.append({'timestamp_ms': ts, 'gyro_x': float(angleX), 'gyro_y': float(angleY)})
    gyro_df = pd.DataFrame(mpu_rows)
    gyro_df['timestamp_s'] = (gyro_df['timestamp_ms'] // 1000).astype(int)
    physio_df['timestamp_s'] = (physio_df['timestamp_ms'] // 1000).astype(int)
    return gyro_df, physio_df


def generate_synthetic_dataset(out_csv, num_sessions=100, duration_s=300, sample_rate_mpu=10, balance_modes=True, seed=0, window_s=5, age=25, resting_hr=60, speed_perturb=0.0, mixed=False, switch_min=10, switch_max=30, transition_s=5):
    """生成 num_sessions 个合成会话并输出统一的窗口化 CSV 文件。
    当 mixed=True 时，会在会话内随机切换模式（间隔在 switch_min..switch_max 秒间），过渡持续 transition_s 秒。
    """
    rng = np.random.RandomState(seed)
    all_rows = []
    for i in range(num_sessions):
        # choose mode balanced or random
        if balance_modes:
            mode_id = i % len(MODE_PARAMS)
        else:
            mode_id = int(rng.choice(list(MODE_PARAMS.keys())))
        gyro_df, physio_df = synthesize_session(mode_id, duration_s=duration_s, sample_rate_mpu=sample_rate_mpu, seed=seed + i, speed_perturb=speed_perturb, mixed=mixed, switch_min=switch_min, switch_max=switch_max, transition_s=transition_s)
        windows = make_windows(gyro_df, physio_df, window_s=window_s)
        for w in windows:
            speed = w['speed_mps']
            if speed is None:
                target_mode = 1
                target_speed = TARGET_SPEED[target_mode]
            else:
                diffs = {m:abs(speed - s) for m,s in TARGET_SPEED.items()}
                target_mode = min(diffs, key=diffs.get)
                target_speed = TARGET_SPEED[target_mode]
            angle_std = (w['angleX_std'] or 0.0) + (w['angleY_std'] or 0.0)
            fatigue = fatigue_score_from_window(w['avg_hr'], w['hr_slope'], angle_std, age=age, resting_hr=resting_hr)
            max_hr = 220 - age
            hr_ratio = None
            if w['avg_hr'] is not None:
                hr_ratio = (w['avg_hr'] - resting_hr) / max(1, (max_hr - resting_hr))
            action = expert_rule_action(w['speed_mps'], target_speed, w['avg_spo2'], fatigue, hr_ratio)
            reward = reward_for_window(w['speed_mps'], target_speed, action, w['avg_spo2'], fatigue, hr_ratio)
            quality_flag = 0
            if w['count_data'] < 1 or w['count_mpu'] < 1:
                quality_flag = 1
            all_rows.append({
                **w,
                'session_id': i,
                'session_mode': (MODE_PARAMS[mode_id]['name'] if not mixed else 'mixed'),
                'target_mode': MODE_NAME[target_mode],
                'target_speed': target_speed,
                'fatigue_score': fatigue,
                'hr_ratio': round(hr_ratio,3) if hr_ratio is not None else None,
                'expert_action': action,
                'reward': reward,
                'quality_flag': quality_flag
            })
    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"已生成合成训练数据：{out_csv}，包含 {len(df)} 个窗口，来自 {num_sessions} 个会话 (mixed={mixed})")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['file','synth'], default='file', help='数据来源：file（处理 simulated_data） 或 synth（合成会话）')
    p.add_argument('--data-dir', default='simulated_data')
    p.add_argument('--out', default='data/train_windows.csv')
    p.add_argument('--window', type=int, default=5)
    p.add_argument('--age', type=int, default=25)
    p.add_argument('--resting-hr', type=int, default=60)
    # synth options
    p.add_argument('--synth-sessions', type=int, default=100, help='合成会话数量')
    p.add_argument('--synth-duration', type=int, default=300, help='每个会话时长（秒）')
    p.add_argument('--synth-sample-rate', type=int, default=10, help='每秒 MPU 采样次数')
    p.add_argument('--synth-seed', type=int, default=0)
    p.add_argument('--synth-balance', action='store_true', help='是否在模式间平衡样本')
    p.add_argument('--synth-perturb', type=float, default=0.0, help='每秒速度相对扰动标准差（例如0.2表示20%）')
    p.add_argument('--synth-mixed', action='store_true', help='是否在会话内启用混合模式切换')
    p.add_argument('--synth-switch-min', type=int, default=10, help='混合模式切换间隔下限（秒）')
    p.add_argument('--synth-switch-max', type=int, default=30, help='混合模式切换间隔上限（秒）')
    p.add_argument('--synth-transition', type=int, default=5, help='单次过渡持续时间（秒）')
    # Mixed-mode usage example:
    #   python generate_training_data.py --mode synth --synth-mixed --synth-switch-min 8 --synth-switch-max 20 --synth-transition 4 --synth-sessions 50 --out data/mixed.csv
    # By default, switching interval is random in [switch_min, switch_max] seconds and transitions last `synth-transition` seconds.

    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.mode == 'file':
        gyro, physio = get_latest_files(args.data_dir)
        process_files(gyro, physio, args.out, window_s=args.window, age=args.age, resting_hr=args.resting_hr)
    else:
        generate_synthetic_dataset(args.out, num_sessions=args.synth_sessions, duration_s=args.synth_duration, sample_rate_mpu=args.synth_sample_rate, balance_modes=args.synth_balance, seed=args.synth_seed, window_s=args.window, age=args.age, resting_hr=args.resting_hr, speed_perturb=args.synth_perturb, mixed=args.synth_mixed, switch_min=args.synth_switch_min, switch_max=args.synth_switch_max, transition_s=args.synth_transition)
