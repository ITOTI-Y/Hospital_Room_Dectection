import torch
import logging
import torchvision
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .dataset import *
from .config import Train_Config, Model_Config, Loss_Config
from .model import get_model
from .loss import *

TRAIN_CONFIG = Train_Config()
MODEL_CONFIG = Model_Config()

class Train():
    
    def __init__(self, train_dataset: RoomDataset, val_dataset: RoomDataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG.BATCH_SIZE, shuffle=TRAIN_CONFIG.SHUFFLE, num_workers=TRAIN_CONFIG.NUM_WORKERS)
        self.val_loader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG.BATCH_SIZE, shuffle=False, num_workers=TRAIN_CONFIG.NUM_WORKERS)
        self.model = get_model().to(self.device)
        self.loss = CombineLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=TRAIN_CONFIG.LR, weight_decay=TRAIN_CONFIG.WEIGHT_DECAY)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode=TRAIN_CONFIG.LR_MODE, factor=TRAIN_CONFIG.LR_FACTOR, patience=TRAIN_CONFIG.LR_PATIENCE)
        self.best_val_loss = float('inf')
        self.patience = TRAIN_CONFIG.PATIENCE
        self.patience_counter = 0

        self.setup_logging()
        self.setup_tensorboard()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO,  format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def setup_tensorboard(self):
        self.writer = SummaryWriter(log_dir=TRAIN_CONFIG.LOG_DIR)

    def train(self):
        # 添加模型结构到tensorboard
        # sample_input = torch.randn(TRAIN_CONFIG.BATCH_SIZE, 3, *MODEL_CONFIG.IMAGE_SIZE).to(self.device)
        # traced_model = torch.jit.trace(self.model, sample_input)
        # self.writer.add_graph(traced_model, sample_input)

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
            for i, (image, label) in enumerate(pbar):
                image, label = image.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(pixel_values=image)
                loss = self.loss(outputs, label)
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
    
    def _val_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation', leave=False) as pbar:
                for image, label in pbar:
                    image, label = image.to(self.device), label.to(self.device)
                    outputs = self.model(pixel_values=image)
                    loss = self.loss(outputs, label)
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(label.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar('Loss/val', avg_loss, epoch)

        # 计算IOU
        iou = self._compute_iou(all_preds, all_labels)
        self.writer.add_scalar('IOU/val', iou, epoch)

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
        self.model = get_model()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.image_path = image_path

    def _transform_image(self, image:torch.Tensor):
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.Resize(),
            ToTensorV2()
        ])
        result = transform(image=image)
        return result['image']

    def run(self):
        self.model.eval()
        with torch.no_grad():
            image = torchvision.io.read_image(self.image_path).to(self.device)
            outputs = self.model(image.unsqueeze(0))