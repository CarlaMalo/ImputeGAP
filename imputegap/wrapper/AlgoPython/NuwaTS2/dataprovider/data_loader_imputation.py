# ===============================================================================================================
# SOURCE: https://github.com/Chengyui/NuwaTS/tree/master
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://arxiv.org/pdf/2405.15317
# ===============================================================================================================

from torch.utils.data import Dataset
from imputegap.wrapper.AlgoPython.NuwaTS.utils.timefeatures import time_features
import warnings
import numpy as np

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True, timeenc=0, freq='h',  seasonal_patterns=None, percent=10,  train_sensors=None, val_sensors=None, test_sensors=None, tr=None, ts=None, m_tr=None, m_ts=None, ts_m=None, batch_size=None, verbose=True):
        # size [seq_len, label_len, pred_len]
        # info

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.patch_size = size[3]

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.ts_m = ts_m
        self.tr = tr
        self.ts = ts
        self.m_tr = m_tr
        self.m_ts = m_ts
        self.m_tr_heal = None
        self.m_tr_cont = None
        self.verbose = verbose

        self.__read_data__()

    def __read_data__(self):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()


        # Decide which raw data to use depending on split. Expect data shaped (channels, timesteps).
        if self.set_type in (0, 1):
            if self.tr is None:
                raise ValueError("`tr` must be provided for train/val splits")
            df_raw = pd.DataFrame(self.tr)
        else:
            if self.ts is None:
                raise ValueError("`ts` must be provided for test split")
            df_raw = pd.DataFrame(self.ts)

        # Generate synthetic timestamps for columns (time flows along axis=1 i.e. columns)
        sync = df_raw.copy()
        base_time = pd.Timestamp('2025-01-01 00:00:00')
        timestamps = [base_time + pd.Timedelta(hours=i) for i in range(sync.shape[1])]
        sync.columns = timestamps

        # Split sensors for train/validation if applicable. We expect `self.tr` shaped (channels, timesteps).
        if self.set_type != 2:
            M, N = self.tr.shape
            split_idx = M // 2
            cont_tr = self.tr[:split_idx]
            heal_tr = self.tr[split_idx:]

            # Masks split in the same way if provided
            if self.m_tr is not None:
                self.m_tr_cont = self.m_tr[:split_idx]
                self.m_tr_heal = self.m_tr[split_idx:]
            else:
                self.m_tr_cont = None
                self.m_tr_heal = None

            self.train_sensors = heal_tr
            self.val_sensors = cont_tr

            if self.set_type == 0:
                df_data = pd.DataFrame(self.train_sensors)
            else:
                df_data = pd.DataFrame(self.val_sensors)
        else:
            # Test uses its own test sensors and mask
            M, N = self.ts.shape
            self.test_sensors = self.ts
            df_data = pd.DataFrame(self.test_sensors)

        # Apply standard scaling
        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Create timestamp dataframe for time feature extraction
        # sync.columns contains timesteps (one per column), so data_stamp will be indexed by timesteps
        df_stamp = pd.DataFrame({'date': sync.columns})
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        # Encode time features (one row per timestep)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month)
            df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour)
            data_stamp = df_stamp.drop(['date'], axis=1).values  # shape: (num_timesteps, 4)
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)  # shape: (num_timesteps, num_features)
        else:
            # fallback: provide simple positional time index
            data_stamp = np.arange(len(df_stamp)).reshape(-1, 1)  # shape: (num_timesteps, 1)
        self.data_stamp = data_stamp

        # Store processed data
        # Keep shape as (channels, timesteps) internally. When __getitem__ is called we'll slice
        # across the timestep axis and transpose to (seq_len, channels) so callers get (T, C).
        self.data_x = data
        self.data_y = data

        if self.verbose:
            print(f"{self.data_x.shape = }, {self.data_y.shape = }, {self.data_stamp.shape = }")


    def __getitem__(self, index):
        # Window indices are along the time axis (columns). We slice columns [s_begin:s_end)
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # Ensure indices do not exceed data bounds
        num_timesteps = self.data_x.shape[1]
        r_end_clamped = min(r_end, num_timesteps)

        # Internal data shape: (channels, timesteps). Slice columns then transpose to (seq_len, channels).
        seq_x = self.data_x[:, s_begin:s_end].T  # shape: (seq_len, channels)
        seq_y = self.data_y[:, r_begin:r_end_clamped].T  # shape: (actual_len, channels)
        # data_stamp is indexed by timesteps, so slice rows [s_begin:s_end] to get (seq_len, num_time_features)
        seq_x_mark = np.array(self.data_stamp[s_begin:s_end])  # shape: (seq_len, 4) or (seq_len, num_features)
        seq_y_mark = np.array(self.data_stamp[r_begin:r_end_clamped])  # shape: (actual_len, 4) or (actual_len, num_features)
        
        # Pad seq_y, seq_y_mark to maintain consistent shape (label_len + pred_len)
        expected_y_len = self.label_len + self.pred_len
        if seq_y.shape[0] < expected_y_len:
            pad_len = expected_y_len - seq_y.shape[0]
            seq_y = np.pad(seq_y, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
            seq_y_mark = np.pad(seq_y_mark, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)

        # Select corresponding mask slice and transpose to (seq_len, channels)
        if self.set_type == 0:
            mask_raw = self.m_tr_heal
        elif self.set_type == 1:
            mask_raw = self.m_tr_cont
        else:
            mask_raw = self.m_ts
        # Fallback to no-missing mask if none provided
        if mask_raw is None:
            mask_raw = np.ones_like(self.data_x)
        mask = mask_raw[:, s_begin:s_end].T

        if self.verbose:
            print(f"Index {index} shapes: x={seq_x.shape}, y={seq_y.shape}, x_mark={seq_x_mark.shape}, y_mark={seq_y_mark.shape}, mask={mask.shape}, seq_len={self.seq_len}, patch_size={self.patch_size}, pred_len={self.pred_len}, label_len={self.label_len}")

        return seq_x, seq_y, seq_x_mark, seq_y_mark, mask

    def __len__(self):
        # Number of sliding windows along the time axis (columns)
        # We need s_end = s_begin + seq_len and r_end = s_end - label_len + label_len + pred_len = s_begin + seq_len + pred_len
        # So max index is when r_end <= num_timesteps, i.e., s_begin + seq_len + pred_len <= num_timesteps
        # Therefore s_begin <= num_timesteps - seq_len - pred_len, so max s_begin is num_timesteps - seq_len - pred_len
        # Number of valid indices is (num_timesteps - seq_len - pred_len + 1)
        return max(0, self.data_x.shape[1] - self.seq_len - self.pred_len + 1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
