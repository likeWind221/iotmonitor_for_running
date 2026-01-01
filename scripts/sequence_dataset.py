import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Construct sequences from window-aggregated CSV rows grouped by session_id.

    Each sample is a sequence of shape (seq_len, n_features) and a label (mode index).
    By default, label is the mode at the last time step.
    """

    def __init__(self, csv_path, seq_len=10, features=None, label_col='current_mode', scaler=None, mode='train', stride=1):
        df = pd.read_csv(csv_path)
        # ensure ordering
        df = df.sort_values(['session_id', 'window_start_ms'])
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.features = features or ['avg_hr','hr_slope','avg_spo2','speed_mps','speed_trend','angleX_mean','angleX_std','angleY_mean','angleY_std','fatigue_score']
        # label mapping (ensure native Python types for JSON serialization)
        self.label_col = label_col
        unique_vals = sorted(df[label_col].unique())
        self.label_map = {}
        for i, v in enumerate(unique_vals):
            try:
                key = int(v)
            except Exception:
                key = v
            self.label_map[key] = int(i)
        # inverse mapping: index -> original label
        self.inv_label = {v:k for k,v in self.label_map.items()}

        # per-session sequences
        sequences = []  # (feat_array, label)
        for sid, g in df.groupby('session_id'):
            arr = g[self.features].to_numpy(dtype=np.float32)
            labels = g[self.label_col].to_numpy()
            n = len(arr)
            if n < self.seq_len:
                continue
            for start in range(0, n - self.seq_len + 1, self.stride):
                seq = arr[start:start+self.seq_len]
                lbl = labels[start + self.seq_len - 1]
                sequences.append((seq, self.label_map[lbl]))

        if len(sequences) == 0:
            raise ValueError('No sequences constructed. Try smaller seq_len or check CSV.')

        self.X = np.stack([s[0] for s in sequences])  # (N, L, F)
        self.y = np.array([s[1] for s in sequences], dtype=np.int64)

        # scaler: fit if provided scaler is None and in train mode
        if scaler is None and mode == 'train':
            self.scaler = StandardScaler()
            N, L, F = self.X.shape
            self.scaler.fit(self.X.reshape(-1, F))
        else:
            self.scaler = scaler

        # apply scaler
        N, L, F = self.X.shape
        Xflat = self.X.reshape(-1, F)
        Xflat = self.scaler.transform(Xflat)
        self.X = Xflat.reshape(N, L, F)

    def save_scaler(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load_scaler(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], int(self.y[idx])

    def get_label_map(self):
        return self.label_map
