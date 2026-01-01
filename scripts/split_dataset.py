"""
按 session 分层划分数据集（train/val/test）并保存 CSV
用法示例：
  python scripts/split_dataset.py --input data/train_windows_synth_big_acc.csv --out-dir data/splits --seed 42

说明：
- 按会话（session_id）分割，按照 session 的 session_mode 做分层（保证每个集合包含三种模式的代表）
- 默认按照 80/10/10 划分
- 可选择移除质量差样本（quality_flag==1）
"""
import os
import argparse
import pandas as pd
import numpy as np


def stratified_session_split(df, group_col='session_id', strat_col='session_mode', train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, 'fractions must sum to 1'
    # sessions dataframe
    sessions = df[[group_col, strat_col]].drop_duplicates()
    train_ids = set()
    val_ids = set()
    test_ids = set()

    rng = np.random.RandomState(seed)
    for mode, sub in sessions.groupby(strat_col):
        ids = sub[group_col].unique().tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        # avoid overflow
        n_val = min(n_val, max(0, n - n_train))
        n_test = n - n_train - n_val
        train_ids.update(ids[:n_train])
        val_ids.update(ids[n_train:n_train+n_val])
        test_ids.update(ids[n_train+n_val: n_train+n_val+n_test])

    train = df[df[group_col].isin(train_ids)].reset_index(drop=True)
    val = df[df[group_col].isin(val_ids)].reset_index(drop=True)
    test = df[df[group_col].isin(test_ids)].reset_index(drop=True)
    return train, val, test


def summarize_and_save(df, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}.csv")
    df.to_csv(out_path, index=False)
    # print summary
    print(f"Saved {prefix} -> {out_path} (rows={len(df)})")
    print('  mode distribution:')
    print(df['target_mode'].value_counts())
    print('  action distribution:')
    print(df['expert_action'].value_counts())
    print('  quality_flag sum:', int(df['quality_flag'].sum()))
    return out_path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='输入 CSV（例如 data/train_windows_synth_big_acc.csv）')
    p.add_argument('--out-dir', default='data/splits', help='输出目录')
    p.add_argument('--train-frac', type=float, default=0.8)
    p.add_argument('--val-frac', type=float, default=0.1)
    p.add_argument('--test-frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--remove-quality', action='store_true', help='是否移除 quality_flag==1 的窗口')
    args = p.parse_args()

    df = pd.read_csv(args.input)
    if args.remove_quality:
        before = len(df)
        df = df[df['quality_flag'] == 0].reset_index(drop=True)
        print(f"Removed quality_flag==1: {before - len(df)} rows removed")

    train, val, test = stratified_session_split(df, group_col='session_id', strat_col='session_mode', train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    summarize_and_save(train, args.out_dir, 'train')
    summarize_and_save(val, args.out_dir, 'val')
    summarize_and_save(test, args.out_dir, 'test')

    # quick overall stats
    print('\nOverall:')
    print(' total rows:', len(df))
    print(' train/val/test rows: ', len(train), len(val), len(test))
    print('Done.')
