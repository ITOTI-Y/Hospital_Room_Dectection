import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Swinv2Config, Swinv2Model, Mask2FormerConfig, Mask2FormerForUniversalSegmentation
from .config import Model_Config, COLOR_MAP
from typing import Dict, Any, Optional

MODEL_CONFIG = Model_Config()

class SwinV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        # 加载Swin Transformer
        self.swin_config = Swinv2Config.from_pretrained(MODEL_CONFIG.PRETRAINED_MODEL)

        # 修改Swin V2配置以适应输入大小
        self.swin_config.image_size = MODEL_CONFIG.IMAGE_SIZE[0]

        # 创建Swin V2模型作为骨干网络
        self.model: Swinv2Model = Swinv2Model.from_pretrained(MODEL_CONFIG.PRETRAINED_MODEL, config=self.swin_config)

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = True) -> Dict[str, Any]:
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=output_hidden_states,)
        return outputs.reshaped_hidden_states[:-1]
    
class UniversalChannelAttention(nn.Module):
    def __init__(self,in_channels, reduction_ratio=16):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // self.reduction_ratio, bias=False)
        self.fc2 = nn.Linear(in_channels // self.reduction_ratio, in_channels, bias=False)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = F.relu(y, inplace=True)
        y = self.fc2(y)
        y = torch.sigmoid(y).view(b, c, 1, 1)

        return x * y.expand_as(x)
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2,use_attention =True, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.use_attention = use_attention

        # upsample layer
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

        # conv layer
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels if use_skip else 0), out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # channel attention layer
        if use_attention:
            self.attention = UniversalChannelAttention(out_channels)

    def forward(self, x:torch.Tensor, skip:torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if self.use_skip and skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv(x)
        if self.use_attention:
            x = self.attention(x)
        
        return x
class SwinV2Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SwinV2Backbone()
        self.upsample1 = UpsampleBlock(768, 384, use_attention=True, use_skip=True)
        self.upsample2 = UpsampleBlock(384, 192, use_attention=True, use_skip=True)
        self.upsample3 = UpsampleBlock(192, 96, use_attention=True, use_skip=True)
        self.final_upsample = UpsampleBlock(96, len(COLOR_MAP), scale_factor=4, use_attention=False, use_skip=False)

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, Any]:
        e1,e2,e3,e4 = self.backbone(pixel_values=pixel_values)
        d1 = self.upsample1(e4, e3)
        m21 = self.upsample2(e3, e2)
        d2 = self.upsample2(d1, self.upsample2(e3, e2))
        m11 = self.upsample3(e2, e1)
        m12 = self.upsample3(m21, m11)
        d3 = self.upsample3(d2, m12)
        y = self.final_upsample(d3, None)

        return y

# class Mask2FormerSegmentation(nn.Module):
#     def __init__(self, backbone_config):
#         super().__init__()
#         self.mask2former_config = Mask2FormerConfig(
#             backbone_config=backbone_config,
#             num_queries=100,
#             num_classes=len(COLOR_MAP),
#             hidden_dim=256,
#             num_feature_levels=len(backbone_config.depths),
#         )
#         self.model = Mask2FormerForUniversalSegmentation(config=self.mask2former_config)

#     def forward(self, 
#                 pixel_values: torch.Tensor, 
#                 labels: Optional[torch.Tensor] = None,
#                 pixel_mask: Optional[torch.LongTensor] = None,
#                 output_hidden_states: Optional[bool] = None,
#                 output_attentions: Optional[bool] = None,
#                 return_dict: Optional[bool] = None):
        
#         if labels is not None:
#             batch_size, num_classes, height, width = labels.shape
            
#             class_labels = labels.argmax(dim=1)
#             mask_labels = labels.transpose(0, 1).reshape(num_classes * batch_size, 1, height, width)
#             class_labels = class_labels.unsqueeze(1).repeat(1, num_classes, 1, 1).reshape(num_classes * batch_size, height, width)
            
#             non_zero_masks = mask_labels.sum(dim=(1, 2, 3)) > 0
#             mask_labels = mask_labels[non_zero_masks]
#             class_labels = class_labels[non_zero_masks]
            
#             mask_labels = [mask for mask in mask_labels]
#             class_labels = [cls_label for cls_label in class_labels]
#         else:
#             mask_labels = None
#             class_labels = None

#         return self.model(
#             pixel_values=pixel_values,
#             mask_labels=mask_labels,
#             class_labels=class_labels,
#             pixel_mask=pixel_mask,
#             output_hidden_states=output_hidden_states,
#             output_attentions=output_attentions,
#             return_dict=return_dict
#         )
    
# class SwinV2Mask2FormerModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = SwinV2Backbone()
#         self.segmentation_head = Mask2FormerSegmentation(self.backbone.swin_config)

#         # 将Swin V2骨干网络设置为Mask2Former的encoder
#         self.segmentation_head.model.model.pixel_level_module.encoder = self.backbone

#     def forward(self, 
#                 pixel_values: torch.Tensor, 
#                 labels: Optional[torch.Tensor] = None,
#                 pixel_mask: Optional[torch.LongTensor] = None,
#                 output_hidden_states: Optional[bool] = None,
#                 output_attentions: Optional[bool] = None,
#                 return_dict: Optional[bool] = None):
        
#         return self.segmentation_head(
#             pixel_values=pixel_values,
#             labels=labels,
#             pixel_mask=pixel_mask,
#             output_hidden_states=output_hidden_states,
#             output_attentions=output_attentions,
#             return_dict=return_dict
#         )
    

def get_model() -> SwinV2Unet:
    return SwinV2Unet()