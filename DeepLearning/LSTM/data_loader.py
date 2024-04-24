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

class StockDataset:
    def __init__(self, filepath, features, target, time_step, split_ratio=0.8,directory="/"):
        self.filepath = filepath
        self.directory = directory
        self.features = features
        self.target = target
        self.time_step = time_step
        self.split_ratio = split_ratio
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = self.load_data()

    def load_data(self):
        # 加载数据、填充缺失值、特征选择等
        df = pd.read_csv(self.filepath)
        df.fillna(method='bfill', inplace=True)
        X = df[self.features].values.astype(np.float32)
        y = df[self.target].values.astype(np.float32)
        # 归一化
        X_scaled = self.scaler.fit_transform(X)
        # 时间窗口化
        X_scaled, y = self.time_series_window(X_scaled, y)
        return X_scaled, y

    def time_series_window(self, X, y):
    # 窗口化处理
        X_windowed = np.array([X[i - self.time_step:i, :] for i in range(self.time_step, len(X))])
        y_windowed = y[self.time_step:]
        return X_windowed, y_windowed
    
    def get_loaders(self, batch_size):
        # 划分训练测试集并创建DataLoaders
        train_size = int(len(self.data[0]) * self.split_ratio)
        X_train, X_test = self.data[0][:train_size], self.data[0][train_size:]
        y_train, y_test = self.data[1][:train_size], self.data[1][train_size:]
        # 转换为张量
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                      torch.tensor(y_train, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                                     torch.tensor(y_test, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers = 4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers = 4)
        return train_loader, test_loader
    
    def get_data(self):
        train_size = int(len(self.data[0]) * self.split_ratio)
        X_train, X_test = self.data[0][:train_size], self.data[0][train_size:]
        y_train, y_test = self.data[1][:train_size], self.data[1][train_size:]
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test =  torch.tensor(X_test,dtype=torch.float32)
        y_test =  torch.tensor(y_test,dtype=torch.float32)
        return X_train,X_test,y_train,y_test
    
