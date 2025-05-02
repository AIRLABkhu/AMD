import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import get_feat_shapes, SimpleAdapter

@torch.no_grad()
def modified_zscore(data: torch.Tensor, threshold=3.5):
    '''
    data: batch, spatial, channel
    '''
    x = data.norm(dim=-1)  # batch, spatial
    median = torch.median(x, dim=1, keepdim=True).values
    mad = torch.median(torch.abs(x - median), dim=1, keepdim=True).values

    modified_z = 0.6745 * (x - median) / mad
    outlier_mask = torch.abs(modified_z) > threshold
    return outlier_mask


class AMD(Distiller):
    """Artifact Manipulating Distillation"""
    def __init__(self, student, teacher, cfg):
        super(AMD, self).__init__(student, teacher)
        self.align_loss_weight = cfg.AMD.LOSS.ALIGN_WIEGHT
        self.feat_loss_weight = cfg.AMD.LOSS.FEAT_WEIGHT
        self.m_layers = cfg.AMD.M_LAYERS
        self.align_type = cfg.AMD.ALIGN_TYPE
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.AMD.INPUT_SIZE
        )
        # Adapters
        self.adapter_dict = nn.ModuleDict({
            **{
                f"adapter_{m_l:03d}": SimpleAdapter(feat_s_shapes[m_l][-1], feat_t_shapes[m_l][-1])
                for m_l in self.m_layers
            },
            "adapter_fin": SimpleAdapter(feat_s_shapes[-1][-1], feat_t_shapes[-1][-1])
        })

        self.af_enabled = cfg.AMD.AF.ENABLE
        self.af_type = cfg.AMD.AF.CRITERIA.TYPE
        self.af_threshold = cfg.AMD.AF.CRITERIA.THRES

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.adapter_dict.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.adapter_dict.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        final_student, feature_student = self.student.forward_wohead(image)
        with torch.no_grad():
            final_f_t, feature_teacher = self.teacher.forward_wohead(image)
            
        # losses
        final_f_s = self.adapter_dict.adapter_fin(final_student)
        match self.align_type:
            case 'cosine':
                loss_align = self.align_loss_weight * 0.5 * (1 - F.cosine_similarity(final_f_s, final_f_t, dim=-1).mean())
            case 'mse':
                loss_align = self.align_loss_weight * F.mse_loss(final_f_s, final_f_t)
            case 'both':
                loss_align = self.align_loss_weight * (0.5 * (1 - F.cosine_similarity(final_f_s, final_f_t, dim=-1).mean())
                                                        + F.mse_loss(final_f_s, final_f_t))
            case _:
                raise NotImplementedError(self.align_type)

        loss_feat = 0.0
        for m_l in self.m_layers:
            f_s = self.adapter_dict[f"adapter_{m_l:03d}"](feature_student["feats"][m_l])
            f_t = feature_teacher["feats"][m_l]
            if self.af_enabled:
                match self.af_type:
                    case 'zscore':
                        outlier_mask = modified_zscore(f_t, threshold=self.af_threshold)
                    case _:
                        raise NotImplementedError(self.af_type)
                f_t[outlier_mask] = torch.nan
                f_s[outlier_mask] = torch.nan
                loss_feat = loss_feat + torch.square(f_s - f_t).nanmean()
            else:
                loss_feat = loss_feat + F.mse_loss(f_s, f_t)
        loss_feat = self.feat_loss_weight * loss_feat / len(self.m_layers) 

        losses_dict = {
            f"loss_align_{self.align_type}": loss_align,
            "loss_kd": loss_feat,
        }

        return torch.zeros(f_s.size(0), 1000, dtype=f_s.dtype, device=f_s.device), losses_dict

    # def 

