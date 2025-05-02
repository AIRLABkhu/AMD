import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)

class SimpleAdapter(nn.Module):
    def __init__(self, s_features, t_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        in_features = s_features
        out_features = t_features       
        hidden_features = hidden_features or in_features     
        self.fc1 = nn.Linear(in_features, out_features)       # Downconv
        # self.act = act_layer()
        # self.fc2 = nn.Linear(hidden_features, out_features)      # UPconv
        # self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)            
        # x = self.act(x)            
        # x = self.drop(x)
        # x = self.fc2(x)           
        # x = self.drop(x)   
        return x



def get_feat_shapes(student, teacher, input_size):
    data = torch.randn(1, 3, *input_size)
    with torch.no_grad():
        feat_s = None if student is None else student(data)[1]
        feat_t = None if teacher is None else teacher(data)[1]
    feat_s_shapes = None if feat_s is None else [f.shape for f in feat_s["feats"]]
    feat_t_shapes = None if feat_t is None else [f.shape for f in feat_t["feats"]]
    return feat_s_shapes, feat_t_shapes


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
