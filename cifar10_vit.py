import torch as th
import torch.nn.functional as F
import lightning as ltn
import argparse
import lightning.pytorch as pl
import math

from torch import nn
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

import glob
import os
import copy
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("-b", "--batch", type=int, default=64, help="batch size of training")
parser.add_argument("-m", "--model", type=str, default='vit_cifar10', help="model to execute")
opt = parser.parse_args()

# 设置加速器
if th.cuda.is_available():
    accelerator = 'gpu'
    th.set_float32_matmul_precision('medium')
elif th.backends.mps.is_available():
    accelerator = 'mps'
else:
    accelerator = 'cpu'


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Generate a matrix of size (max_len, d_model)
        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len).unsqueeze(1).float()
        div_term = th.exp(th.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # Calculate the divisor term for positional encoding
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        
        # Add a batch dimension to the positional encoding matrix
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Add positional encoding to the input embedding
        return x + self.pe[:, :x.size(1)].detach()


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout after attention and feedforward layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Multihead Attention
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward network
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        # Embedding layer (Note: You might need to adjust this based on how your input is represented)
        # The input to the Transformer is typically already embedded.
        # If your input is raw image patches, you should process them before passing them to the Transformer.
        # self.embedding = nn.Embedding(1000, d_model)  # Assuming input size of 1000, adjust as needed
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # List of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # If using embedding, uncomment the following line:
        # src = self.embedding(src)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        src = self.positional_encoding(src)
        
        # Pass through each encoder layer
        for layer in self.layers:
            src = layer(src, src_mask, src_key_padding_mask)
        
        # Final normalization
        src = self.norm(src)
        
        return src


class CIFAR10Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 1e-3
        self.counter = 0
        self.labeled_loss = 0
        self.labeled_correct = 0

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Use ReduceLROnPlateau scheduler
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        z = self.forward(x)
        loss = F.cross_entropy(z, y)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.forward(x)
        loss = F.cross_entropy(z, y)
        self.log('val_loss', loss, prog_bar=True)

        pred = z.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).sum().item()
        total = y.size(0)
        accuracy = correct / total
        self.log('val_acc', accuracy, prog_bar=True)

        self.labeled_loss += loss.item() * total
        self.labeled_correct += correct
        self.counter += total

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        z = self.forward(x)
        loss = F.cross_entropy(z, y)
        self.log('test_loss', loss, prog_bar=True)

        pred = z.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).sum().item()
        total = y.size(0)
        accuracy = correct / total
        self.log('test_acc', accuracy, prog_bar=True)

    def on_validation_end(self) -> None:
        if self.trainer.sanity_checking:
            return

        avg_loss = self.labeled_loss / self.counter
        avg_acc = self.labeled_correct / self.counter
        logging.info(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_acc:.4f}")

        self.counter = 0
        self.labeled_loss = 0
        self.labeled_correct = 0

    def on_save_checkpoint(self, checkpoint) -> None:
        # This method is now primarily for cleaning up old checkpoints.
        # The actual testing is done in a separate function.

        # 清理旧的检查点，保留最新的5个
        checkpoint_files = sorted(glob.glob('best-*.ckpt'), reverse=True)
        for ix, ckpt in enumerate(checkpoint_files):
            if ix >= 5:
                try:
                    os.unlink(ckpt)
                    logging.info(f"Deleted old checkpoint: {ckpt}")
                except OSError as e:
                    logging.error(f"Error deleting checkpoint {ckpt}: {e}")

        print()


# 定义 Vision Transformer (ViT) 模型
class VisionTransformerModel(CIFAR10Model):  # Inherit from CIFAR10Model
    def __init__(self, image_size=224, patch_size=16, num_classes=10, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1, pretrained=False):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        # 输入图像被切割为多个小块 (patches)
        self.patch_embeddings = nn.Conv2d(in_channels=3, out_channels=self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # 标准的 ViT 使用 Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            d_model=self.dim, 
            nhead=self.heads, 
            num_layers=self.depth,
            dim_feedforward=self.mlp_dim, 
            dropout=self.dropout
        )

        # 分类头
        self.cls_token = nn.Parameter(th.zeros(1, 1, self.dim))  # CLS token
        self.pos_embedding = nn.Parameter(th.zeros(1, (image_size // patch_size) ** 2 + 1, self.dim))  # Position embedding
        nn.init.xavier_uniform_(self.cls_token)
        nn.init.xavier_uniform_(self.pos_embedding)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.mlp_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_dim, self.num_classes)
        )
        
        if pretrained:
            # Placeholder for loading pretrained weights
            logging.warning("Pretrained model loading is not implemented yet.")
            # self.load_pretrained_weights()

    def forward(self, x):
        # 将输入图片切分成 Patch
        x = self.patch_embeddings(x)  # 输出形状 [batch_size, dim, patch_grid_size, patch_grid_size]
        x = x.flatten(2).transpose(1, 2)  # 展平每个 Patch, 输出形状 [batch_size, num_patches, dim]

        # 添加位置编码和 CLS token
        batch_size, num_patches, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # CLS token 重复 batch_size 次
        x = th.cat((cls_tokens, x), dim=1)  # 在第一维上连接 CLS token
        x = x + self.pos_embedding[:, :num_patches + 1]

        # Transformer 编码器
        x = self.dropout_layer(x)
        x = self.transformer_encoder(x)

        # 取 CLS token 作为最终特征
        x = x[:, 0]

        # MLP 分类头
        x = self.mlp_head(x)
        return x

    def load_pretrained_weights(self):
        # Implement logic to load pretrained weights here
        # This might involve downloading weights from a URL or loading from a local file
        # and then loading them into the model using `load_state_dict`
        logging.info("Loading pretrained weights...")
        # Example:
        # pretrained_dict = torch.load("path/to/pretrained_weights.pth")
        # model_dict = self.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # self.load_state_dict(model_dict)
        logging.warning("Loading pretrained weights is not fully implemented in this example.")
        pass


def test_best_vit(model_class=VisionTransformerModel, model_filename='cifar10-vit.pt'):
    checkpoint_files = sorted(glob.glob('best-*.ckpt'), reverse=True)
    if not checkpoint_files:
        logging.error("No checkpoint files found.")
        return

    fname = checkpoint_files[0]
    logging.info(f"Loading checkpoint: {fname}")
    try:
        with open(fname, 'rb') as f:
            model = model_class()
            checkpoint = th.load(f, map_location='cpu')
            
            # Handle potential mismatch in state_dict keys due to inheritance
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('net.'):
                    new_state_dict[k[4:]] = v
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict, strict=False)
            model = model.cpu()
            model.eval()

            logging.info(f"Loaded checkpoint: {fname}")

            with th.no_grad():
                counter, success = 0, 0
                for test_batch in test_loader:
                    x, y = test_batch
                    x, y = x.cpu(), y.cpu()
                    z = model(x)
                    pred = z.data.max(1, keepdim=True)[1]
                    correct = pred.eq(y.data.view_as(pred)).sum().item()
                    success += correct
                    counter += y.size(0)
                    if counter % 1000 == 0:
                        logging.info(f"Processed {counter} samples...")
            logging.info(f"Accuracy: {success / counter:.5f}")
            th.save(model, model_filename)
            logging.info(f"Saved model to {model_filename}")

    except Exception as e:
        logging.error(f"Error loading or testing checkpoint: {e}")


if __name__ == '__main__':
    logging.info('loading data...')
    # 数据增强和标准化
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),  # 调整为ViT需要的尺寸
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    cifar10_train = CIFAR10('datasets', train=True, download=True, transform=transform_train)
    cifar10_test = CIFAR10('datasets', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(cifar10_train, shuffle=True, batch_size=opt.batch, num_workers=8)
    val_loader = DataLoader(cifar10_test, batch_size=opt.batch, num_workers=8)
    test_loader = DataLoader(cifar10_test, batch_size=opt.batch, num_workers=8)

    # 训练
    logging.info('construct trainer...')
    trainer = pl.Trainer(
        accelerator=accelerator,
        precision=16,  # Use mixed precision training
        max_epochs=opt.n_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=30)],
        log_every_n_steps=50
    )

    logging.info('construct model...')
    model = VisionTransformerModel(num_classes=10, pretrained=opt.model.startswith('pretrained'))

    logging.info('training...')
    trainer.fit(model, train_loader, val_loader)

    logging.info('testing...')
    test_best_vit()
