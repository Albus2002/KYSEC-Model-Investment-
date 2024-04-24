import torch
import torch.nn as nn

class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, inputs, targets):
        # 计算MSE
        loss = torch.mean(abs(inputs - targets))
        return loss