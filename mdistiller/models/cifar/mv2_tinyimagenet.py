import torch
import torch.nn as nn
import torch.nn.functional as F
from .._base import Lambda, ModelBase


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual

class MobileNetV2(nn.Module):

    def __init__(self, num_classes=100):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, num_classes, 1)
    
    def activate(self, x):
        return F.relu6(x)
        
    def forward_stem(self, x):
        return self.pre(x)

    def get_layers(self):
        return nn.Sequential(
            nn.Sequential(
                self.stage1,
                self.stage2,
            ),
            self.stage3,
            self.stage4,
            nn.Sequential(
                self.stage5,
                self.stage6,
                self.stage7,
                self.conv1,
            ),
        )

    def forward_pool(self, x):
        return F.adaptive_avg_pool2d(x, 1)

    def get_head(self):
        return nn.Sequential(
            self.conv2,
            nn.Flatten(),
        )

    def forward(self, x):
        x = F.relu6(self.pre(x))
        f0 = x
        x = self.stage1(F.relu6(x))
        x = self.stage2(x)
        f1 = x
        x = self.stage3(F.relu6(x))
        f2 = x
        x = self.stage4(F.relu6(x))
        f3 = x
        x = self.stage5(F.relu6(x))
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        f4 = x
        x = F.adaptive_avg_pool2d(x, 1)
        avg = x
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        feats = {}
        feats["preact_feats"] = [f1, f2, f3, f4]
        feats["feats"] = [F.relu6(f1), F.relu6(f2), F.relu6(f3), F.relu6(f4)]
        feats["pooled_feat"] = avg

        return x, feats

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)

def mobilenetv2_tinyimagenet(**kwargs):
    return MobileNetV2(**kwargs)