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


class FitViT(Distiller):
    """from FitNets: Hints for Thin Deep Nets"""

    def __init__(self, student, teacher, cfg):
        super(FitViT, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.FITNET.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.FITNET.LOSS.FEAT_WEIGHT
        self.hint_layer = cfg.FITNET.HINT_LAYER
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        )
        self.adapter = SimpleAdapter(
            feat_s_shapes[self.hint_layer][-1], feat_t_shapes[self.hint_layer][-1]
        )

        self.af_enabled = cfg.AF.ENABLE
        self.af_type = cfg.AF.CRITERIA.TYPE
        self.af_threshold = cfg.AF.CRITERIA.THRES

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.adapter.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.adapter.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            feature_teacher = self.teacher.forward_partial(image, self.hint_layer)
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        f_s = self.adapter(feature_student["feats"][self.hint_layer])

        if self.af_enabled:
            f_t = feature_teacher["feats"][self.hint_layer]
            match self.af_type:
                case 'zscore':
                    outlier_mask = modified_zscore(f_t, threshold=self.af_threshold)
                case _:
                    raise NotImplementedError(self.af_type)
            f_t[outlier_mask] = torch.nan
            f_s[outlier_mask] = torch.nan
            loss_feat = self.feat_loss_weight * torch.square(f_s - f_t).nanmean()

        else:
            loss_feat = self.feat_loss_weight * F.mse_loss(
                f_s, feature_teacher["feats"][self.hint_layer]
            )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_feat,
        }
        return logits_student, losses_dict

    # def 

