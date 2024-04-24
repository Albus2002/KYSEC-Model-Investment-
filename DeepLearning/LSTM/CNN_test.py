from data_loader import StockDataset
from model import CNNLSTMModel, Trainer, Predictor
from loss import CustomMSELoss
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 数据集参数
    filepath = 'stock1.csv'
    features = ['BidPrice1', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5',
    'AskPrice1', 'AskPrice2', 'AskPrice3', 'AskPrice4', 'AskPrice5',
    'BidVolume1', 'BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5',
    'AskVolume1', 'AskVolume2', 'AskVolume3', 'AskVolume4', 'AskVolume5',
    'Spread', 'tempVolume', 'tempValue', 'VOI', 'QR1', 'QR2', 'QR3', 'QR4', 'QR5',
    'HR1', 'HR2', 'HR3', 'HR4', 'HR5', 'Press']  # 特征列名
    target = 'Return'   # 目标列名
    time_step = 60      # 时间步长
    split_ratio = 0.8   # 训练/测试数据集划分比例
    batch_size = 32     # 批次大小

    # 模型参数
    input_channels = len(features)
    output_size = 1
    feature_dim = 35
    hidden_size = 50
    num_layers = 2
    dropout_rate = 0.2
    input_size = feature_dim * time_step

    # 加载数据
    dataset = StockDataset(filepath, features, target, time_step, split_ratio)
    train_loader, test_loader = dataset.get_loaders(batch_size)

    # 初始化模型
    model = CNNLSTMModel(input_channels, feature_dim, time_step, output_size, input_size, hidden_size, num_layers, dropout_rate)

    # 损失函数和优化器
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # 训练模型
    trainer = Trainer(model, {'train': train_loader, 'test': test_loader}, criterion, optimizer, num_epochs=10)
    trainer.train()

    # 保存模型
    trainer.save_model('stock_model.pth')

    # 预测
    predictor = Predictor(model, scaler=dataset.scaler,dataset=dataset)
    predictions = predictor.predict()  # 预测

    # 计算测试集的均方根误差（RMSE）
    test_rmse = np.sqrt(np.mean((predictions - dataset.get_data()[3].reshape(-1, 1).numpy())**2))
    print(f'Test RMSE: {test_rmse}')

    # 绘制实际值和预测值
    plt.figure(figsize=(15, 6))
    plt.plot(dataset.get_data()[3].numpy(), label='Actual Test Returns')
    plt.plot(predictions, label='Predicted Test Returns', alpha=0.7)
    plt.title('Test Data - Actual vs Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Return')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()