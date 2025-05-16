from typing import Any
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

    def forward(self, x):
        x = self.fc1(x)            
        return x

@torch.no_grad()
def make_zscore_mask(data: torch.Tensor, threshold: float=5.5, adaptive: bool=True, alpha: float=1.0, except_cls: bool=True):
    '''
    :input:
        data: batch, spatial, channel
        
    :returns:
        a tensor with shape (batch, spatial). 
        contains one on outliers, zero, otherwise.
    '''
    x = data.norm(dim=-1)  # batch, spatial
    median = torch.median(x, dim=1, keepdim=True).values
    mad = torch.median(torch.abs(x - median), dim=1, keepdim=True).values
    mad = torch.where(mad < 1e-6, torch.full_like(mad, 1e-6), mad)

    modified_z = 0.6745 * (x - median) / mad
    if adaptive:
        std_z = modified_z.std(dim=1, keepdim=True)
        threshold = (threshold * (1 + alpha * std_z)).clamp(max=6.0)
        
    outlier_mask = torch.abs(modified_z) > threshold
    if except_cls:
        outlier_mask[:, 0].fill_(0.0)
    return outlier_mask.to(dtype=data.dtype, device=data.device)

@torch.no_grad()
def make_gaussian_std_mask(x: torch.Tensor, threshold: float=2.0, eps: float=1.0E-8, except_cls: bool=True):
    '''
    :params:
        data: batch, spatial, channel
        
    :returns:
        a tensor with shape (batch, spatial). 
        contains one on outliers, zero, otherwise.
    '''
    x = x.norm(dim=-1)
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    x_normalized = (x - mean) / (std + eps)
    outlier_mask = x_normalized > threshold
    if except_cls:
        outlier_mask[:, 0].fill_(0.0)
    return outlier_mask.to(dtype=x.dtype, device=x.device)

@torch.no_grad()
def masked_std_mean(x: torch.Tensor, mask: torch.Tensor, keepdim: bool=False, unbiased: bool=True):
    indices = mask.nonzero()
    mask_nan = torch.sparse_coo_tensor(
        indices=indices.T,
        values=torch.full(size=(len(indices),), fill_value=torch.nan),
        size=mask.shape,
        dtype=mask.dtype,
        device=mask.device,
    ).to_dense()
    x_masked = x + mask_nan.unsqueeze(-1)
    x_mean = x_masked.nanmean(dim=1, keepdim=True)  # B, 1, C
    
    x_centered = torch.square(x_masked - x_mean)  # ..................| B, P, C
    valid_count = (1.0 - mask).sum(dim=1, keepdim=True)  # ...........| B, 1
    squared_sum = torch.nansum(x_centered, dim=1, keepdim=True)  # ...| B, 1, C

    denom = valid_count - (1 if unbiased else 0)
    x_std = torch.sqrt(squared_sum / denom.unsqueeze(-1))
    
    x_mean[torch.isnan(x_std)].fill_(0.0)
    x_std = x_std.nan_to_num(1.0) 

    if keepdim:
        return x_std, x_mean
    else:
        return x_std.squeeze(), x_mean.squeeze()

def normalize_outlier_artifacts(x: torch.Tensor, outlier_mask: torch.Tensor, eps:float=1.0E-8):
    '''
    :input:
        x: torch.Tensor with shape (B, P, C).
        mask: torch.Tensor with shape (B, P).
    
    :returns:
        a tensor with shape (B, P, C) with normalized outliers. 
    '''
    outlier_mask = outlier_mask.clone()
    inlier_mask = 1.0 - outlier_mask
    inlier_mask[:, 0].fill_(0.0)
    outlier_mask[:, 0].fill_(1.0)
    std_out, mean_out = masked_std_mean(x, inlier_mask, keepdim=True)
    std_in, mean_in = masked_std_mean(x, outlier_mask, keepdim=True)
    # normalized outliers only
    outlier_mask = outlier_mask.unsqueeze(-1)
    inlier_mask = inlier_mask.unsqueeze(-1)
    normalized_x = (x - mean_out) * std_in / (std_out + eps) + mean_in
    return (x * inlier_mask) + (normalized_x * outlier_mask)


# reconMHA module
class ReconMHA(nn.MultiheadAttention):
    def __init__(
        self,
        embed_dim: int,
        num_patches: int,
        num_heads: int,
        base_threshold: float=5.5,
        adaptive_threshold: bool=True,
        inlier_recon_rate: float=0.1,
        dropout: float = 0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int|None = None,
        vdim: int|None = None,
        batch_first: bool = True, 
        device: Any|None = None,
        dtype: Any|None = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.arguments = dict(
            embed_dim=embed_dim,
            num_patches=num_patches,
            num_heads=num_heads,
            base_threshold=base_threshold,
            adaptive_threshold=adaptive_threshold,
            inlier_recon_rate=inlier_recon_rate,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
        )
        self.recon_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.recon_token, std=0.02)  # ViT-style initialization
        self.base_threshold = base_threshold
        self.adaptive_threshold = adaptive_threshold
        self.inlier_recon_rate = inlier_recon_rate
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # ViT-style initialization
        
    def forward(self, x: torch.Tensor, outlier_mask: torch.Tensor|None=None, ignore_outliers: bool=True):
        '''
        :parameters:
        
            `x: torch.Tensor`: Tensor with shape of (batch, patch, channel).

        :returns:
        
            `attn: tuple[torch.Tensor, torch.Tensor]`: 
                The first tensor holds the attenion output and the second one holds the attention score matrix.
            
            `outlier_mask: torch.Tensor`: 
                Stands for the binary floating-point tensor consists of 0 and 1 with shape of (batch, patch). 
                One implies that the patch is an outlier.
            
            `random_mask: torch.Tensor`: 
                Stands for the binary floating-point tensor consists of 0 and 1 with shape of (batch, batch). 
                One implies that the patch is randomly selected to be masked. 
                This will be returned only in the training mode. 
        '''
        if outlier_mask is None:
            outlier_mask = make_zscore_mask(
                data=x, 
                threshold=self.base_threshold, 
                adaptive=self.adaptive_threshold,
            )
            
        outlier_mask = outlier_mask.unsqueeze(-1)
        inlier_mask = 1 - outlier_mask
        
        if self.training:
            inlier_indices = inlier_mask.squeeze(-1).nonzero()  # [(bidx, pidx), ...]
            random_indices = inlier_indices[torch.rand(len(inlier_indices)) <= self.inlier_recon_rate]
            random_mask = torch.sparse_coo_tensor(
                random_indices.T, values=torch.ones(len(random_indices)), size=outlier_mask.shape[:2],
                dtype=random_indices.dtype, device=random_indices.device,
            ).to_dense().unsqueeze(-1)
            x_mask = inlier_mask - random_mask
            r_mask = outlier_mask + random_mask
        else:
            x_mask = inlier_mask
            r_mask = outlier_mask
        
        if not ignore_outliers:
            x_mask = 1.0
        x_ready = torch.add(x * x_mask, self.recon_token * r_mask)
        
        x_ready = x_ready + self.pos_embed
        if self.training:
            return super().forward(x_ready, x_ready, x_ready), outlier_mask, random_mask
        else:
            return super().forward(x_ready, x_ready, x_ready), outlier_mask


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
