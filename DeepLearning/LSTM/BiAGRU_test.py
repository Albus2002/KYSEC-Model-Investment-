from data_loader import StockDataset
from model import BiAGRU, Trainer, Predictor
from loss import CustomMSELoss
import tracemalloc
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 启动内存跟踪
    tracemalloc.start()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # BiAGRU模型参数
    input_size = len(features)  # 输入特征的数量
    hidden_size = 64  # GRU隐藏层大小
    num_layers = 2    # GRU层数
    dropout = 0.2    # Dropout比率
    output_size = 1   # 输出大小，对于回归任务通常是1

    # 加载数据
    dataset = StockDataset(filepath, features, target, time_step, split_ratio)
    train_loader, test_loader = dataset.get_loaders(batch_size)

    # 初始化BiAGRU模型
    model = BiAGRU(input_size, hidden_size, num_layers, dropout, output_size)
    model = model.cuda()

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    trainer = Trainer(model, {'train': train_loader, 'test': test_loader}, criterion, optimizer, num_epochs=3,device = device)
    trainer.train()

    # 保存模型
    trainer.save_model('biagru_stock_model.pth')

    # 预测
    predictor = Predictor(model, scaler=dataset.scaler,dataset=dataset,device = device)
    predictions = predictor.predict()  # 预测

    # 计算残差平方和（Residual Sum of Squares, RSS）
    X_test, y_test = dataset.data[0][int(len(dataset.data[0]) * split_ratio):], dataset.data[1][int(len(dataset.data[0]) * split_ratio):]
    res = np.sum((y_test - predictions) ** 2)

    # 计算总平方和（Total Sum of Squares, TSS）
    tot = np.sum((y_test - np.mean(y_test)) ** 2)

    # 计算 R² 值
    r2 = 1 - res / tot

    print(r2)

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
    plt.savefig('BiAGRU_result.png',dpi=350)
    plt.show()

if __name__ == '__main__':
    main()