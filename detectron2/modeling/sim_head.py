import torch
import torch.nn as nn
import torch.nn.functional as F


class SimNet(nn.Module):
    def __init__(self, in_dim):
        super(SimNet, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, 128)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, roi_feat):
        return F.normalize(self.mlp(roi_feat), dim=1)