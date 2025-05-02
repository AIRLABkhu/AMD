from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .mobilenetv1 import MobileNetV1
from .vit import (
    vit_tiny_patch16_224,
    vit_small_patch16_224,
    vit_base_patch16_224,
    vit_large_patch16_224,
    clip_base_patch16_224,
    clip_base_patch16_224_ft_in,
    clip_xsmall_patch16_224,
    dinov2_tiny_patch14_518,
    dinov2_small_patch14_518,
    dinov2_base_patch14_518,
    dinov2_large_patch14_518,
    deit3_small_patch16_224,
    deit3_medium_patch16_224,
    deit3_base_patch16_224,
    deit3_large_patch16_224,
)


imagenet_model_dict = {
    "ResNet18": resnet18,
    "ResNet34": resnet34,
    "ResNet50": resnet50,
    "ResNet101": resnet101,
    "MobileNetV1": MobileNetV1,
    "vit_tiny": vit_tiny_patch16_224,
    "vit_tiny.unic": vit_tiny_patch16_224,
    "vit_small": vit_small_patch16_224,
    "vit_base": vit_base_patch16_224,
    "vit_base.unic": vit_base_patch16_224,
    "vit_large": vit_large_patch16_224,

    "clip_xsmall": clip_xsmall_patch16_224,
    "clip_base": clip_base_patch16_224,
    "clip_base_ft": clip_base_patch16_224_ft_in,
    
    "dinov2_tiny": dinov2_tiny_patch14_518,
    "dinov2_small": dinov2_small_patch14_518,
    "dinov2_base": dinov2_base_patch14_518,
    "dinov2_large": dinov2_large_patch14_518,
    "deit3_small": deit3_small_patch16_224,
    "deit3_medium": deit3_medium_patch16_224,
    "deit3_base": deit3_base_patch16_224,
    "deit3_large": deit3_large_patch16_224,
}
