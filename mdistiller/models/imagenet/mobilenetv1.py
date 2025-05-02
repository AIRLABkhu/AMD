from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from .._base import ModelBase


class MobileNetV1(nn.Module, ModelBase):
    def __init__(self, **kwargs):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride, act: bool=True):
            layers = [
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
            if act:
                layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),  # ...................| 00
            conv_dw(32, 64, 1),  # ..................| 01
            conv_dw(64, 128, 2, act=False),  # ......| 02
            
            conv_dw(128, 128, 1),  # ................| 03
            conv_dw(128, 256, 2, act=False),  # .....| 04
            
            conv_dw(256, 256, 1),  # ................| 05
            conv_dw(256, 512, 2),  # ................| 06
            conv_dw(512, 512, 1),  # ................| 07
            conv_dw(512, 512, 1),  # ................| 08
            conv_dw(512, 512, 1),  # ................| 09
            conv_dw(512, 512, 1, act=False),  # .....| 10
            
            conv_dw(512, 512, 1),  # ................| 11
            conv_dw(512, 1024, 2),  # ...............| 12
            conv_dw(1024, 1024, 1, act=False),  # ...| 13
            
            nn.AvgPool2d(7),  # .....................| 14
        )
        self.fc = nn.Linear(1024, 1000)
        
    def get_arch(self) -> Literal['cnn', 'transformer']:
        return 'cnn'
        
    def forward_stem(self, x):
        return self.model[0](x)
    
    def get_layers(self):
        return nn.Sequential(
            self.model[1:3],
            self.model[3:5],
            self.model[5:11],
            self.model[11:14],
        )
        
    def forward_pool(self, x):
        return self.model[14](F.relu(x)).reshape(-1, 1024)
    
    def get_head(self):
        return self.fc

    def forward(self, x, is_feat=False):
        feat1 = self.model[0:3](x)
        feat2 = self.model[3:5](F.relu(feat1))
        feat3 = self.model[5:11](F.relu(feat2))
        feat4 = self.model[11:14](F.relu(feat3))
        feat5 = self.model[14](F.relu(feat4))
        avg = feat5.reshape(-1, 1024)
        out = self.fc(avg)

        feats = {}
        feats["pooled_feat"] = avg
        feats["feats"] = [F.relu(feat1), F.relu(feat2), F.relu(feat3), F.relu(feat4)]
        feats["preact_feats"] = [feat1, feat2, feat3, feat4]
        return out, feats

    def get_bn_before_relu(self):
        bn1 = self.model[3][-1]
        bn2 = self.model[5][-1]
        bn3 = self.model[11][-1]
        bn4 = self.model[13][-1]
        return [bn1, bn2, bn3, bn4]

    def get_stage_channels(self):
        return [128, 256, 512, 1024]
    

if __name__ == '__main__':
    from .._base import test_model
    model = MobileNetV1()
    x = torch.randn(2, 3, 224, 224)
    assert test_model(model, x)
    print("MobileNetV1 passed")
