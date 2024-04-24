import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader,Dataset

class StockDataset(Dataset):
    def __init__(self, file_list, feature_columns, target_column, time_step = 60):
        self.file_list = file_list
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.time_step = time_step
        # 存储每个文件的长度，这样我们可以知道如何索引它们
        self.file_lengths = self._get_file_lengths()

    def _get_file_lengths(self):
        lengths = []
        for file_path in self.file_list:
            # 读取.h5文件以获取长度
            with pd.read_hdf(file_path, 'r') as store:
                data_length = store.get_storer('data').nrows
                lengths.append(data_length - (self.time_step - 1))  # 减去时间步长
        return lengths

    def __len__(self):
        return sum(self.file_lengths)

    def __getitem__(self, idx):
        # 找到包含索引的文件
        for i, file_length in enumerate(self.file_lengths):
            if idx < file_length:
                file_path = self.file_list[i]
                break
            idx -= file_length
        
        # 读取文件中的时间窗口
        with pd.read_hdf(file_path) as store:
            start_idx = idx
            end_idx = start_idx + self.time_step
            data = store.select('data', start=start_idx, stop=end_idx)
            X = data[self.feature_columns].values
            y = data[self.target_column].values[-1]  # 假设我们预测窗口最后一个值

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
