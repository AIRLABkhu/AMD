import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import get_feat_shapes, SimpleAdapter, make_zscore_mask

class AMD_MASK(Distiller):
    """Artifact Manipulating Distillation"""
    def __init__(self, student, teacher, cfg):
        super(AMD_MASK, self).__init__(student, teacher)
        self.feat_loss_weight = cfg.AMD.LOSS.FEAT_WEIGHT
        self.m_layers = cfg.AMD.M_LAYERS + [len(self.teacher.get_layers()) - 1]
        self.align_type = cfg.AMD.ALIGN_TYPE
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.AMD.INPUT_SIZE
        )
        # Adapters from Teacher to Student
        self.adapter_dict = nn.ModuleDict({
            **{
                f"adapter_{m_l:03d}": SimpleAdapter(feat_t_shapes[m_l][-1], feat_s_shapes[m_l][-1])
                for m_l in self.m_layers
            }
        })

        self.af_enabled = cfg.AMD.AF.ENABLE
        self.af_type = cfg.AMD.AF.CRITERIA.TYPE
        self.af_threshold = cfg.AMD.AF.CRITERIA.THRES

    def get_learnable_parameters(self):
        yield from super().get_learnable_parameters()
        yield from self.adapter_dict.parameters()

    def get_extra_parameters(self):
        return sum(
            sum(map(torch.Tensor.numel, module.parameters()))
            for module in [
                self.adapter_dict,
            ]
        )

    def forward_train(self, image, target, **kwargs):
        _, feature_student = self.student.forward_wohead(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher.forward_wohead(image)
            
        loss_feat = 0.0
        for m_l in self.m_layers:
            f_s = feature_student["feats"][m_l]
            f_t = feature_teacher["feats"][m_l]
            if self.af_enabled:
                match self.af_type:
                    case 'zscore':
                        outlier_mask = make_zscore_mask(f_t, threshold=self.af_threshold)
                    case _:
                        raise NotImplementedError(self.af_type)
                inlier_bool_mask = outlier_mask.bool().logical_not()
                f_t = self.adapter_dict[f"adapter_{m_l:03d}"](f_t)
                f_s_inliers = f_s[inlier_bool_mask]
                f_t_inliers = f_t[inlier_bool_mask]
                loss_feat_mse = F.mse_loss(f_s_inliers, f_t_inliers)
                loss_feat = loss_feat + loss_feat_mse
            else:
                f_t = self.adapter_dict[f"adapter_{m_l:03d}"](f_t)
                loss_feat = loss_feat + F.mse_loss(f_s, f_t)
        loss_feat = self.feat_loss_weight * loss_feat / len(self.m_layers) 

        losses_dict = {
            "loss_kd": loss_feat,
        }

        return torch.zeros(f_s.size(0), 1000, dtype=f_s.dtype, device=f_s.device), losses_dict


