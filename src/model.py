import torch
import torch.nn as nn
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
        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        # 使用 reshaped_hidden_states 作为特征图
        if output_hidden_states:
            feature_maps = outputs.reshaped_hidden_states
        else:
            # 如果没有 reshaped_hidden_states，使用 last_hidden_state 并重塑
            last_hidden_state = outputs.last_hidden_state
            batch_size, _, hidden_size = last_hidden_state.shape
            height = width = int((last_hidden_state.shape[1])**0.5)
            feature_maps = [last_hidden_state.reshape(batch_size, height, width, hidden_size).permute(0, 3, 1, 2)]

        return {"feature_maps": feature_maps}
    
class Mask2FormerSegmentation(nn.Module):
    def __init__(self, backbone_config):
        super().__init__()
        self.mask2former_config = Mask2FormerConfig(
            backbone_config=backbone_config,
            num_queries=100,
            num_classes=len(COLOR_MAP),
            hidden_dim=256,
            num_feature_levels=len(backbone_config.depths),
        )
        self.model = Mask2FormerForUniversalSegmentation(config=self.mask2former_config)

    def forward(self, 
                pixel_values: torch.Tensor, 
                labels: Optional[torch.Tensor] = None,
                pixel_mask: Optional[torch.LongTensor] = None,
                output_hidden_states: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                return_dict: Optional[bool] = None):
        
        if labels is not None:
            batch_size, num_classes, height, width = labels.shape
            
            class_labels = labels.argmax(dim=1)
            mask_labels = labels.transpose(0, 1).reshape(num_classes * batch_size, 1, height, width)
            class_labels = class_labels.unsqueeze(1).repeat(1, num_classes, 1, 1).reshape(num_classes * batch_size, height, width)
            
            non_zero_masks = mask_labels.sum(dim=(1, 2, 3)) > 0
            mask_labels = mask_labels[non_zero_masks]
            class_labels = class_labels[non_zero_masks]
            
            mask_labels = [mask for mask in mask_labels]
            class_labels = [cls_label for cls_label in class_labels]
        else:
            mask_labels = None
            class_labels = None

        return self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
            pixel_mask=pixel_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict
        )
    
class SwinV2Mask2FormerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SwinV2Backbone()
        self.segmentation_head = Mask2FormerSegmentation(self.backbone.swin_config)

        # 将Swin V2骨干网络设置为Mask2Former的encoder
        self.segmentation_head.model.model.pixel_level_module.encoder = self.backbone

    def forward(self, 
                pixel_values: torch.Tensor, 
                labels: Optional[torch.Tensor] = None,
                pixel_mask: Optional[torch.LongTensor] = None,
                output_hidden_states: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                return_dict: Optional[bool] = None):
        
        return self.segmentation_head(
            pixel_values=pixel_values,
            labels=labels,
            pixel_mask=pixel_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict
        )
    
def get_model() -> SwinV2Mask2FormerModel:
    return SwinV2Mask2FormerModel()