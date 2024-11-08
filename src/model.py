import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Swinv2Config, Swinv2Model, Mask2FormerConfig, Mask2FormerForUniversalSegmentation
from .config import COLOR_MAP, Mask2Former_Model_Config, Swin_Model_Config
from typing import Dict, Any, Optional

MODEL_CONFIG = Swin_Model_Config()
MASK2FORMER_CONFIG = Mask2Former_Model_Config()

# class SwinV2Backbone(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # 加载Swin Transformer
#         self.swin_config = Swinv2Config.from_pretrained(MODEL_CONFIG.PRETRAINED_MODEL)

#         # 修改Swin V2配置以适应输入大小
#         self.swin_config.image_size = MODEL_CONFIG.IMAGE_SIZE[0]

#         # 创建Swin V2模型作为骨干网络
#         self.model: Swinv2Model = Swinv2Model.from_pretrained(MODEL_CONFIG.PRETRAINED_MODEL, config=self.swin_config)

#     def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = True) -> Dict[str, Any]:
#         outputs = self.model(pixel_values=pixel_values, output_hidden_states=output_hidden_states,)
#         return outputs
    
# class UniversalChannelAttention(nn.Module):
#     def __init__(self,in_channels, reduction_ratio=16):
#         super().__init__()
#         self.reduction_ratio = reduction_ratio
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(in_channels, in_channels // self.reduction_ratio, bias=False)
#         self.fc2 = nn.Linear(in_channels // self.reduction_ratio, in_channels, bias=False)
    
#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc1(y)
#         y = F.relu(y, inplace=True)
#         y = self.fc2(y)
#         y = torch.sigmoid(y).view(b, c, 1, 1)

#         return x * y.expand_as(x)
    
# class UpsampleBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, scale_factor=2,use_attention =True, use_skip=True):
#         super().__init__()
#         self.use_skip = use_skip
#         self.use_attention = use_attention

#         # upsample layer
#         self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

#         # conv layer
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels + (out_channels if use_skip else 0), out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#         # channel attention layer
#         if use_attention:
#             self.attention = UniversalChannelAttention(out_channels)

#     def forward(self, x:torch.Tensor, skip:torch.Tensor) -> torch.Tensor:
#         x = self.upsample(x)
#         if self.use_skip and skip is not None:
#             x = torch.cat([x, skip], dim=1)
        
#         x = self.conv(x)
#         if self.use_attention:
#             x = self.attention(x)
        
#         return x
# class SwinV2Unet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = SwinV2Backbone()
#         self.upsample1 = UpsampleBlock(768, 384, use_attention=True, use_skip=True)
#         self.upsample2 = UpsampleBlock(384, 192, use_attention=True, use_skip=True)
#         self.upsample3 = UpsampleBlock(192, 96, use_attention=True, use_skip=True)
#         self.final_upsample = UpsampleBlock(96, len(COLOR_MAP), scale_factor=4, use_attention=False, use_skip=False)

#     def forward(self, pixel_values: torch.Tensor) -> Dict[str, Any]:
#         e1,e2,e3,e4 = self.backbone(pixel_values=pixel_values)
#         d1 = self.upsample1(e4, e3)
#         m21 = self.upsample2(e3, e2)
#         d2 = self.upsample2(d1, self.upsample2(e3, e2))
#         m11 = self.upsample3(e2, e1)
#         m12 = self.upsample3(m21, m11)
#         d3 = self.upsample3(d2, m12)
#         y = self.final_upsample(d3, None)

#         return y
    
# class SwinV2Wrapper(nn.Module):
#     def __init__(self, swin_model):
#         super().__init__()
#         self.model = swin_model
#         # 修改投影层以匹配Mask2Former期望的通道数和特征图数量
#         self.projections = nn.ModuleList([
#             nn.Conv2d(192, 256, kernel_size=1),
#             nn.Conv2d(384, 256, kernel_size=1),
#             nn.Conv2d(768, 256, kernel_size=1),
#             nn.Conv2d(768, 256, kernel_size=1)
#         ])

#     def forward(self, pixel_values):
#         outputs = self.model(pixel_values, output_hidden_states=True)
#         hidden_states = outputs.hidden_states[1:]  # 跳过第一个hidden state（输入嵌入）
        
#         # 调整特征图的形状和通道数
#         feature_maps = []
#         for i, hidden_state in enumerate(hidden_states):
#             # 调整形状
#             b, hw, c = hidden_state.shape
#             h = w = int(hw**0.5)
#             hidden_state = hidden_state.transpose(1, 2).view(b, c, h, w)
#             # 调整通道数
#             feature_maps.append(self.projections[i](hidden_state))
        
#         # 确保只返回最后四个特征图
#         return feature_maps[-4:]

# class Mask2FormerSegmentation(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # 加载Swin Transformer V2配置
#         self.swin_config = Swinv2Config.from_pretrained(MODEL_CONFIG.PRETRAINED_MODEL)
#         self.swin_config.image_size = MODEL_CONFIG.IMAGE_SIZE[0]
        
#         # 创建Swin V2模型作为骨干网络
#         swin_model = Swinv2Model.from_pretrained(MODEL_CONFIG.PRETRAINED_MODEL, config=self.swin_config)
#         self.backbone = SwinV2Wrapper(swin_model)
        
#         # 配置Mask2Former
#         self.mask2former_config = Mask2FormerConfig(
#             num_channels=3,
#             image_size=MODEL_CONFIG.IMAGE_SIZE[0],
#             num_labels=len(COLOR_MAP),
#             hidden_dim=256,  # 修改为256以匹配backbone输出
#             num_queries=100,
#             no_object_weight=0.1,
#             mask_feature_size=32,
#             backbone_config=self.swin_config,
#         )
        
#         # 创建Mask2Former模型
#         self.mask2former = Mask2FormerForUniversalSegmentation(self.mask2former_config)
        
#         # 替换Mask2Former的骨干网络为我们的Swin V2
#         self.mask2former.model.pixel_level_module.encoder = self.backbone

#     def forward(self, pixel_values: torch.Tensor) -> Dict[str, Any]:
#         outputs = self.mask2former(pixel_values=pixel_values)
        
#         # 调整输出以匹配SwinV2Unet的格式
#         segmentation = outputs.segmentation
        
#         # 你可能需要根据具体需求调整输出格式
#         return {"segmentation": segmentation}

class Mask2FormerWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Mask2FormerForUniversalSegmentation(MASK2FORMER_CONFIG.get_config())
    
    def forward(self, pixel_values: torch.Tensor, **kwargs) -> Dict[str, Any]:
        outputs = self.model(pixel_values=pixel_values, **kwargs)
        return outputs

def get_model() -> nn.Module:
    return Mask2FormerWrapper()