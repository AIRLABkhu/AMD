from abc import ABC, abstractmethod
from typing import Literal
import torch
from torch import nn


class Lambda(nn.Module):
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn
        
    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class ModelBase(ABC):
    
    @abstractmethod
    def get_arch(self) -> Literal['cnn', 'transformer']:
        pass
    
    def activate(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(x)
    
    @abstractmethod
    def forward_stem(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_layers(self) -> nn.Sequential:
        pass
    
    @abstractmethod
    def forward_pool(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_head(self) -> nn.Module:
        pass
    
    
def test_model(model: ModelBase, inputs: torch.Tensor, return_elements: bool=False):
    assert isinstance(model, nn.Module)
    y1 = model.forward(inputs)
    
    preact_feats = []
    y2 = model.forward_stem(inputs)
    for layer in model.get_layers():
        y2 = layer(torch.relu(y2))
        preact_feats.append(y2)
    feats = [torch.relu(y) for y in preact_feats]
    pooled_feat = model.forward_pool(y2)
    y2 = model.get_head().forward(pooled_feat)
    
    result = torch.allclose(y1[0], y2), {
        'preact_feats': [
            torch.allclose(y1_, y2_) for y1_, y2_ in 
            zip(y1[1]["preact_feats"], preact_feats)
        ],
        'feats': [
            torch.allclose(y1_, y2_) for y1_, y2_ in 
            zip(y1[1]["feats"], feats)
        ],
        'pooled_feat': torch.allclose(y1[1]["pooled_feat"], pooled_feat),
    }
    if return_elements:
        return result
    else:
        return all([
            result[0],
            *result[1]['preact_feats'],
            *result[1]['feats'],
            result[1]['pooled_feat'],
        ])
