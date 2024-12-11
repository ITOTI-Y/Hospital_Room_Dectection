import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict
from transformers import Mask2FormerImageProcessor
from .dataset import *
from ..config import Train_Config, Swin_Model_Config, Loss_Config, COLOR_MAP
from .model import get_model
from .loss import *

TRAIN_CONFIG = Train_Config()
SWIN_MODEL_CONFIG = Swin_Model_Config()

class Train():
    
    def __init__(self, train_dataset , val_dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG.BATCH_SIZE, shuffle=TRAIN_CONFIG.SHUFFLE, num_workers=TRAIN_CONFIG.NUM_WORKERS)
        self.val_loader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG.BATCH_SIZE, shuffle=False, num_workers=TRAIN_CONFIG.NUM_WORKERS)
        self.model = get_model()
        self.model = self.model.to(self.device)
        self.loss = CombineLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=TRAIN_CONFIG.LR, weight_decay=TRAIN_CONFIG.WEIGHT_DECAY)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode=TRAIN_CONFIG.LR_MODE, factor=TRAIN_CONFIG.LR_FACTOR, patience=TRAIN_CONFIG.LR_PATIENCE)
        self.best_val_loss = float('inf')
        self.patience = TRAIN_CONFIG.PATIENCE
        self.patience_counter = 0
        self.num_classes = len(COLOR_MAP)

        self.setup_logging()
        self.setup_tensorboard()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO,  format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def setup_tensorboard(self):
        self.writer = SummaryWriter(log_dir=TRAIN_CONFIG.LOG_DIR)

    def train(self):

        for epoch in range(TRAIN_CONFIG.EPOCHS):
            self.logger.info(f'Epoch {epoch}/{TRAIN_CONFIG.EPOCHS}')
            train_loss = self._train_epoch(epoch)
            val_loss = self._val_epoch(epoch)

            self.logger.info(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning Rate', current_lr, epoch)

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_model(TRAIN_CONFIG.SAVE_MODEL_PATH)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                self.logger.info(f'Early stopping at epoch {epoch}')
                break

            self._save_checkpoint(epoch, TRAIN_CONFIG.SAVE_CHECKPOINT_PATH)

        self.writer.close()

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        with tqdm(self.train_loader, desc='Training', leave=False) as pbar:
            for i, (pixel_values, pixel_mask, mask_labels, class_labels) in enumerate(pbar):
                pixel_values, pixel_mask, mask_labels, class_labels = pixel_values.to(self.device), pixel_mask.to(self.device), mask_labels.to(self.device), class_labels.to(self.device)
                # self._debug_image(pixel_values, pixel_mask, mask_labels, class_labels)
                self.optimizer.zero_grad()
                outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, mask_labels=mask_labels, class_labels=class_labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                # 记录训练损失
                step = epoch * len(self.train_loader) + i
                self.writer.add_scalar('Loss/train', loss.item(), step)

                if i % TRAIN_CONFIG.LOG_INTERVAL == 0:
                    for name, param in self.model.named_parameters():
                        self.writer.add_histogram(f'Parameters/{name}', param.data, step)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def _debug_image(self, pixel_values, pixel_mask, mask_labels, class_labels):
        import cv2
        import os
        # 将pixel_values转换为numpy数组并调整维度顺序
        img = pixel_values.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # [3, 512, 512] -> [512, 512, 3]
        img = ((img - img.min()) * 255 / (img.max() - img.min())).astype(np.uint8)

        # 获取mask_labels和class_labels
        mask_labels = mask_labels.squeeze(0).cpu().numpy()  # [x, 512, 512]
        class_labels = class_labels.squeeze(0).cpu().numpy()  # [x]

        # 创建 class_id 到颜色的映射
        # 假设 class_id 从 1 开始，0 为背景
        color_list = list(COLOR_MAP.keys())
        class_id_to_color = {idx: color for idx, color in enumerate(color_list)}

        # 创建一个空的掩码图像
        combined_mask = np.zeros((mask_labels.shape[1], mask_labels.shape[2]), dtype=np.int32)

        # 将每个掩码层对应的类别ID赋值到 combined_mask 中
        for i in range(mask_labels.shape[0]):
            combined_mask[mask_labels[i] > 0] = class_labels[i]

        # 创建彩色掩码图像
        color_mask = np.zeros((combined_mask.shape[0], combined_mask.shape[1], 3), dtype=np.uint8)
        for class_id, color in class_id_to_color.items():
            color_mask[combined_mask == class_id] = color

        # 拼接原始图像和分割的图像
        concatenated = np.concatenate((img, color_mask), axis=1)

        # 保存图像到当前目录
        save_path = os.path.join('.', 'debug_output.jpg')
        cv2.imwrite(save_path, cv2.cvtColor(concatenated, cv2.COLOR_RGB2BGR))
        self.logger.info(f'Debug image saved to {save_path}')


    
    def _val_epoch(self, epoch):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation', leave=False) as pbar:
                for pixel_values, pixel_mask, mask_labels, class_labels in pbar:
                    pixel_values, pixel_mask, mask_labels, class_labels = pixel_values.to(self.device), pixel_mask.to(self.device), mask_labels.to(self.device), class_labels.to(self.device)
                    outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, mask_labels=mask_labels, class_labels=class_labels)
                    loss = outputs.loss
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar('Loss/val', avg_loss, epoch)

        return avg_loss
    
    def _compute_iou(self, preds, labels):
        intersection = np.logical_and(preds, labels)
        union = np.logical_or(preds, labels)
        iou = np.sum(intersection) / np.sum(union)
        return iou
    
    def _save_model(self, path):
        torch.save(self.model.state_dict(), path)
        self.logger.info(f'Model saved to {path}')

    def _load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.logger.info(f'Model loaded from {path}')

    def _save_checkpoint(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter
        }, path)
        self.logger.info(f'Checkpoint saved to {path}')

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.patience_counter = checkpoint['patience_counter']
        self.logger.info(f'Checkpoint loaded from {path}')
        return checkpoint['epoch']
    
class Predict:
    def __init__(self, model_path:pathlib.Path, image_path:pathlib.Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model().to(self.device)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.image_path = image_path

    def _transform_image(self, image:torch.Tensor):
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.Resize(SWIN_MODEL_CONFIG.IMAGE_SIZE, SWIN_MODEL_CONFIG.IMAGE_SIZE),
            ToTensorV2()
        ])
        result = transform(image=np.array(image))
        return result['image']
    
    def preprocess_image(self, image_path, device):
        # Read the image
        image = Image.open(image_path).convert('RGB')
        image = self._transform_image(image)
        # Add batch dimension and move to device
        image = image.unsqueeze(0).to(device)
        
        return image

    def visualize_results(self, image, outputs):
        import matplotlib.pyplot as plt
        
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 将image转化为可视化格式
        outputs = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()  # 将outputs转化为可视化格式

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        axs[0].imshow(image)
        axs[0].set_title('Input Image')
        axs[0].axis('off')

        axs[1].imshow(outputs, cmap='gray')  # 假设outputs是单通道图像，使用灰度图显示
        axs[1].set_title('Model Output')
        axs[1].axis('off')

        plt.savefig('./data/val_test/1f.jpg')

    def run(self):
        self.model.eval()
        with torch.no_grad():
            image = self.preprocess_image(self.image_path, self.device)
            outputs = self.model(image)
            self.visualize_results(image, outputs)


class Mask2FormerPredict:
    def __init__(self, model_path:pathlib.Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model().to(self.device)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.image_processor = Mask2FormerImageProcessor(
            do_resize=IMAGE_PROCESSOR_CONFIG.DO_RESIZE,
            size=IMAGE_PROCESSOR_CONFIG.IMAGE_SIZE,
            ignore_index=IMAGE_PROCESSOR_CONFIG.IGNORE_INDEX,
            num_labels=IMAGE_PROCESSOR_CONFIG.NUM_LABELS,
        )

    def run(self, image_path:pathlib.Path, save_path:pathlib.Path=None):
        image = Image.open(image_path).convert('RGB')
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predict_result = self.image_processor.post_process_instance_segmentation(outputs)
        if save_path:
            self._save_image(image, predict_result, save_path)
        return predict_result
    
    def _save_image(self, image:Image.Image, predict_result:List[torch.Tensor], save_path:pathlib.Path):
        segmentation = predict_result[0].cpu().numpy()
        original_image = np.array(image)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # 显示原始图像
        axs[0].imshow(original_image)
        axs[0].axis('off')
        axs[0].set_title('Original Image')

        # 显示分割结果
        axs[1].imshow(segmentation)
        axs[1].axis('off')
        axs[1].set_title('Segmentation')

        # 调整布局并保存图像
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()