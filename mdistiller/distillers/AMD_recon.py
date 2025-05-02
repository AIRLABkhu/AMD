import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import get_feat_shapes, SimpleAdapter, ReconMHA

class AMD_RECON(Distiller):
    """Artifact Manipulating Distillation"""
    def __init__(self, student, teacher, cfg):
        super(AMD_RECON, self).__init__(student, teacher)
        self.feat_loss_weight = cfg.AMD.LOSS.FEAT_WEIGHT
        self.recon_loss_weight = cfg.AMD.LOSS.RECON_WEIGHT
        self.m_layers = cfg.AMD.M_LAYERS + [len(self.teacher.get_layers()) - 1]
        self.align_type = cfg.AMD.ALIGN_TYPE
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.AMD.INPUT_SIZE
        )
        # af params
        self.af_enabled = cfg.AMD.AF.ENABLE
        self.af_type = cfg.AMD.AF.CRITERIA.TYPE
        self.af_threshold = cfg.AMD.AF.CRITERIA.THRES
        self.af_recon_type = cfg.AMD.AF.RECON.TYPE
        
        # Adapters from Teacher to Student
        self.adapter_dict = nn.ModuleDict({
            **{
                f"adapter_{m_l:03d}": SimpleAdapter(feat_t_shapes[m_l][-1], feat_s_shapes[m_l][-1])
                for m_l in self.m_layers
            }
        })
        # Recon Module
        if (self.af_enabled) & (self.af_recon_type == 'recon_mha'):
            self.recon_module_dict = nn.ModuleDict({
            **{
                f"recon_mha_{m_l:03d}": ReconMHA(embed_dim=self.teacher.embed_dim,
                                                 num_patches=self.teacher.patch_embed.num_patches+1,
                                                 num_heads=int(self.teacher.embed_dim/64),
                                                 base_threshold=self.af_threshold)
                for m_l in self.m_layers
            }
        })

    def get_learnable_parameters(self):
        if self.af_enabled & (self.af_recon_type == 'recon_mha'):
            return super().get_learnable_parameters() + list(self.adapter_dict.parameters()) + list(self.recon_module_dict.parameters())
        else:
            return super().get_learnable_parameters() + list(self.adapter_dict.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.adapter_dict.parameters():
            num_p += p.numel()
        if self.af_enabled & (self.af_recon_type == 'recon_mha'):
            for p in self.recon_module_dict.parameters():
                num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        _, feature_student = self.student.forward_wohead(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher.forward_wohead(image)
        # loss
        # loss for inter feature
        loss_feat = 0.0
        loss_recon = 0.0
        for m_l in self.m_layers:
            f_s = feature_student["feats"][m_l]
            f_t = feature_teacher["feats"][m_l]
            match self.af_recon_type:
                case 'recon_mha':
                    (recon_f_t, _) , outlier_mask, random_mask = self.recon_module_dict[f"recon_mha_{m_l:03d}"](f_t)
                    proj_f_t = self.adapter_dict[f"adapter_{m_l:03d}"](recon_f_t)
                    match self.align_type:
                        case 'cosine':
                            loss_feat = loss_feat + 0.5 * (1 - F.cosine_similarity(f_s, proj_f_t, dim=-1).mean())
                        case 'mse':
                            loss_feat = loss_feat + F.mse_loss(f_s, proj_f_t)
                        case 'both':
                            loss_feat = loss_feat + (0.5 * (1 - F.cosine_similarity(f_s, proj_f_t, dim=-1).mean())
                                                    + F.mse_loss(f_s, proj_f_t))
                        case _:
                            raise NotImplementedError(self.align_type)
                    f_t[outlier_mask] = torch.nan
                    recon_f_t[outlier_mask] = torch.nan
                    loss_recon = loss_recon + torch.square(recon_f_t - f_t).nanmean()
                case _:
                    raise NotImplementedError(self.align_type)
                
        loss_feat = self.feat_loss_weight * loss_feat / len(self.m_layers) 
        loss_recon = self.recon_loss_weight * loss_recon / len(self.m_layers) 

        losses_dict = {
            "loss_kd": loss_feat,
            "loss_recon": loss_recon,
        }

        return torch.zeros(f_s.size(0), 1000, dtype=f_s.dtype, device=f_s.device), losses_dict
    


