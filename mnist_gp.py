import argparse
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import lightning.pytorch as pl

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy


# ---------------- p-network (示例) ----------------
class PNetwork(nn.Module):
    """简单两层全连接网络, MNIST 784->200->10"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200, bias=True)
        self.fc2 = nn.Linear(200, 10, bias=True)

    def forward(self, x):
        # x: [batch,1,28,28]
        x = x.view(x.size(0), -1)    # flatten -> [batch,784]
        x = F.relu(self.fc1(x))      # [batch,200]
        x = self.fc2(x)              # [batch,10]
        return F.log_softmax(x, dim=1)


# ---------------- g-network (示例) ----------------
class GNetwork(nn.Module):
    """
    对 p-network 的某一部分参数进行压缩/生成。
    这里只演示针对 fc1.weight (784->200) 的压缩, 
    你可扩展到 fc2.weight/bias 等等。
    """
    def __init__(self):
        super().__init__()
        # 输入: (i,j) in [0,784) x [0,200), 简化只用2D->小网络
        # 输出: 对应的单个权重
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, coords: th.Tensor):
        """
        coords: shape [N,2], i.e. (row_idx, col_idx)
        返回: shape [N], 对应每个 (i,j) 的预测权重
        """
        out = self.net(coords)
        return out.squeeze(-1)


# --------------- LightningModule ---------------
class LightningPGModule(pl.LightningModule):
    def __init__(self, alpha=0.5, lr_p=1e-3, lr_g=1e-3):
        """
        alpha: 融合系数, 用来将 p-network 当前权重 和 g-network 预测的权重 融合
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # 关键: 手动优化模式

        # p-network 和 g-network
        self.p_net = PNetwork()
        self.g_net = GNetwork()

        self.generation = 0
        self.alpha = alpha
        self.lr_p  = lr_p
        self.lr_g  = lr_g

        # 用于“压缩”的目标: fc1.weight [200,784]
        # 我们做个简单的 (i,j) 索引, 方便 forward
        w_shape = self.p_net.fc1.weight.shape  # [200,784]
        rows = th.arange(w_shape[0])
        cols = th.arange(w_shape[1])
        # 生成所有 (row, col) 的笛卡尔积
        self.register_buffer("all_coords",
            th.stack(
                th.meshgrid(rows, cols, indexing="ij"),
                dim=-1  # shape [200,784,2]
            ).view(-1,2)  # [200*784,2]
        )

    def configure_optimizers(self):
        """
        返回两个优化器: 先后顺序与 training_step 中相匹配.
        注意: 多优化器 + Lightning 需要手动优化.
        """
        opt_p = th.optim.Adam(self.p_net.parameters(), lr=self.lr_p)
        opt_g = th.optim.Adam(self.g_net.parameters(), lr=self.lr_g)
        return [opt_p, opt_g]

    def forward(self, x):
        return self.p_net(x)

    def training_step(self, batch, batch_idx):
        """
        在每个batch里, 同时更新 p-network 和 g-network:
        1) 使用当前 batch 训练 p-network (监督学习)
        2) 用当前 p-network 的 fc1.weight 训练 g-network (逼近)
        3) (可选) 将 g-network 的输出融合回 p-network
        """
        x, y = batch
        # ---- 1) 更新 p-network ----
        opt_p, opt_g = self.optimizers()
        opt_p.zero_grad()

        pred = self.p_net(x)
        loss_p = F.nll_loss(pred, y)
        self.manual_backward(loss_p)
        opt_p.step()

        # ---- 2) 更新 g-network ----
        # 例: 随机抽一部分 coords, 不要一次全部
        frac = 0.05  # 只抽5%进行训练
        all_num = self.all_coords.size(0)
        n_samp = int(all_num * frac)
        idx = th.randperm(all_num, device=self.device)[:n_samp]
        coords_sample = self.all_coords[idx]  # shape [n_samp,2]
        # 真实权重
        real_w = self.p_net.fc1.weight.data  # shape [200,784]
        # 把 real_w flatten成 [200*784], 与 coords 索引一一对应
        real_w_flat = real_w.view(-1)

        target_w = real_w_flat[idx]

        # forward g_net
        opt_g.zero_grad()
        pred_w = self.g_net(coords_sample.float())
        loss_g = F.mse_loss(pred_w, target_w)
        self.manual_backward(loss_g)
        opt_g.step()

        # ---- 3) (可选) 将 g-network 的输出融合回 p-network 权重 ----
        #    alpha越小, 就越倾向于保留g_net的预测(相当于强制p_net的权重向g_net收缩)
        #    alpha越大, 就更尊重p_net原先的训练值.
        with th.no_grad():
            all_pred_w = self.g_net(self.all_coords.float())
            all_pred_w = all_pred_w.view(200,784)
            old_w = self.p_net.fc1.weight.data
            new_w = self.alpha * old_w + (1 - self.alpha) * all_pred_w
            self.p_net.fc1.weight.data.copy_(new_w)
            self.generation = self.generation + 1
            import math
            t = (self.generation - 10000 / 2000)
            t = math.exp(-t)
            self.alpha = 1 / (1 + t)

        # Logging
        self.log('train_loss_p', loss_p, on_step=True, prog_bar=True)
        self.log('train_loss_g', loss_g, on_step=True, prog_bar=True)

        # 可以返回任意一个loss, 这里随便返回 p-loss
        return loss_p

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.p_net(x)
        loss = F.nll_loss(pred, y)
        acc = (pred.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc,  prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.p_net(x)
        acc = (pred.argmax(dim=1) == y).float().mean()
        self.log('test_acc', acc,  prog_bar=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args()

    # 数据集
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST("./data", train=True, download=True, transform=transform)
    test_dataset  = MNIST("./data", train=False, download=True, transform=transform)

    # 简单分割一部分作验证集
    val_size = 5000
    train_size = len(train_dataset) - val_size
    train_ds, val_ds = th.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # LightningModule
    model = LightningPGModule(alpha=args.alpha, lr_p=1e-3, lr_g=1e-3)

    # Trainer
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="auto",
        max_epochs=args.max_epochs,
        # callbacks=[EarlyStopping(monitor="val_loss", patience=30, mode="min")],
        log_every_n_steps=10
    )

    # 训练
    trainer.fit(model, train_loader, val_loader)

    # 测试
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()
