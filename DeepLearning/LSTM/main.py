from data_loader import StockDataset
from model import BiAGRU, Trainer, Predictor
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt

def main():
    # 检查CUDA是否可用，如果可用则使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集和模型的参数
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

    input_size = len(features)  # 输入特征的数量
    hidden_size = 64  # GRU隐藏层大小
    num_layers = 2    # GRU层数
    dropout = 0.2    # Dropout比率
    output_size = 1   # 输出大小，对于回归任务通常是1

    # 加载数据
    dataset = StockDataset(filepath, features, target, time_step, split_ratio)
    train_loader, test_loader = dataset.get_loaders(batch_size)

    # 初始化Transformer模型并移动到设备上
    model = BiAGRU(input_size, hidden_size, num_layers, dropout, output_size)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        # 包装模型以在多个GPU上并行处理
        model = nn.DataParallel(model)
    model.to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    trainer = Trainer(model, {'train': train_loader, 'test': test_loader}, criterion, optimizer, num_epochs=50, device=device)
    trainer.train()

    # 保存模型
    torch.save(model.state_dict(), 'transformer_stock_model.pth')

    # 预测
    predictor = Predictor(model, scaler=dataset.scaler, device=device)
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
    plt.savefig('Transformer_result.png',dpi=350)
    plt.show()

if __name__ == '__main__':
    main()