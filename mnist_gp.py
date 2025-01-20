import argparse
import os, glob
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import lightning.pytorch as pl

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

"""
示例思路:
1) p-network (PNetwork) 为一个两层 MLP, 用于 MNIST 分类.
2) 有 3 个 g-network (GNetworkW1, GNetworkW2, GNetworkB1), 分别生成:
    - 输入层->隐藏层的权重
    - 隐藏层->输出层的权重
    - 隐藏层的偏置
   输出层的偏置(10个)因为参数少, 不做压缩.
3) 使用交替训练 (Intermittent Training):
   - 先对 p-network 做少量 mini-batch 训练 (仿生物中的“自然选择”/学习),
   - 再训练 g-network 让其逼近 p-network 的当前权重 (仿生物中的“基因保留”),
   - 然后将 p-network 的权重替换/融合为 g-network 的输出 (仿生物中的“发育”)。
   - 重复多次 generation.
"""

# ===================== p-network (任务网络) =====================
class PNetwork(nn.Module):
    """
    一个两层全连接 MLP: 784 -> 800 -> 10
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 800, bias=True)
        self.fc2 = nn.Linear(800, 10, bias=True)

    def forward(self, x: th.Tensor):
        # x: [batch, 1, 28, 28]
        x = x.view(x.size(0), -1)       # flatten to [batch, 784]
        x = F.relu(self.fc1(x))        # [batch, 800]
        x = self.fc2(x)                # [batch, 10]
        return F.log_softmax(x, dim=1)

# ===================== g-network (基因网络) - 用于压缩权重 =====================
class GNetworkW1(nn.Module):
    """
    生成输入 -> 隐藏层的权重 (784 * 800)
    输入: (i, j) 的二进制标签, 20 bits
    输出: 对应的实数权重
    """
    def __init__(self, hidden_dim=10):
        super().__init__()
        # 例如(20 -> hidden_dim -> 1)，可以自行扩展层数
        self.net = nn.Sequential(
            nn.Linear(20, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: th.Tensor):
        return self.net(x).squeeze(-1)  # [num_pairs]

class GNetworkW2(nn.Module):
    """
    生成隐藏 -> 输出层的权重 (800 * 10)
    输入: (i, j) 二进制标签, 20 bits
    输出: 对应的实数权重
    """
    def __init__(self, hidden_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: th.Tensor):
        return self.net(x).squeeze(-1)

class GNetworkB1(nn.Module):
    """
    生成隐藏层的偏置 (800)
    输入: 单个神经元的二进制标签, 10 bits
    输出: 对应的实数偏置
    """
    def __init__(self, hidden_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: th.Tensor):
        return self.net(x).squeeze(-1)

# ============== LightningModule 整合 ==============
class LightningPGModule(pl.LightningModule):
    """
    将 p-network 与 g-network 结合在一起:
    - 可以在训练循环里手动执行 "交替训练(Algorithm 1)".
    """
    def __init__(self, fraction_p=0.1, fraction_g=0.1, lr_p=1e-3, lr_g=1e-3, generations=100):
        """
        fraction_p: 每个世代训练 p-network 时, 使用多少比例的数据 (0~1 之间).
        fraction_g: 每个世代训练 g-network 时, 使用多少比例的权重.
        lr_p, lr_g: 学习率
        generations: 迭代次数
        """
        super().__init__()
        # p-network
        self.p_net = PNetwork()

        # g-networks
        self.g_w1 = GNetworkW1(hidden_dim=20)
        self.g_w2 = GNetworkW2(hidden_dim=20)
        self.g_b1 = GNetworkB1(hidden_dim=10)

        self.lr_p = lr_p
        self.lr_g = lr_g
        self.fraction_p = fraction_p
        self.fraction_g = fraction_g
        self.generations = generations

        # 用于记录 val/test 准确率
        self.best_acc = 0.0

    def forward(self, x):
        return self.p_net(x)

    def configure_optimizers(self):
        """
        同时返回对 p-network & g-networks 的优化器 (便于手动在训练循环中区分)
        也可以各自一个 optimizer
        """
        # 分开写更清晰：p-net 用一个优化器，g-net 用另一个
        optimizer_p = th.optim.Adam(self.p_net.parameters(), lr=self.lr_p)
        optimizer_g = th.optim.Adam(
            list(self.g_w1.parameters()) + list(self.g_w2.parameters()) + list(self.g_b1.parameters()),
            lr=self.lr_g
        )
        # lightning 允许返回多个优化器，以便在 training_step 中手动切换
        return optimizer_p, optimizer_g

    def _pnetwork_training_step(self, batch):
        """
        用来训练 p-network, 类似常规 supervised 学习
        """
        x, y = batch
        logits = self.p_net(x)
        loss = F.nll_loss(logits, y)
        return loss

    def _gnetwork_training_step(self, batch_w1, target_w1, batch_w2, target_w2, batch_b1, target_b1):
        """
        用来训练 g-network, 让其输出逼近 p-network 的当前权重
        """
        # 生成预测
        pred_w1 = self.g_w1(batch_w1)  # [num_w1]
        pred_w2 = self.g_w2(batch_w2)  # [num_w2]
        pred_b1 = self.g_b1(batch_b1)  # [num_b1]

        # MSE 损失
        loss_w1 = F.mse_loss(pred_w1, target_w1)
        loss_w2 = F.mse_loss(pred_w2, target_w2)
        loss_b1 = F.mse_loss(pred_b1, target_b1)
        loss = loss_w1 + loss_w2 + loss_b1
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """
        lightning Trainer 会自动调用本函数，并根据 optimizer_idx 告诉我们应该更新哪个优化器.
        但是为了实现文中的 Algorithm 1，我们倾向于在外层手动控制循环 (on_train_epoch_start / end).
        这里保留一个最小实现，使 trainer 不报错即可。
        """
        if optimizer_idx == 0:
            # 训练 p-network
            loss_p = self._pnetwork_training_step(batch)
            self.log("train_loss_p", loss_p)
            return loss_p
        else:
            # 此处仅做一个空操作, 真正训练 g-network 我们会在 on_train_batch_end 手动触发
            return None

    def on_train_epoch_start(self) -> None:
        """
        在每个 epoch 开始时，我们可以进行一次“世代”式训练.
        当然也可以把世代放到 trainer 的 fit 外部，总之思路相似.
        """
        self.train_mode = True  # 标志位

    def on_train_epoch_end(self) -> None:
        self.train_mode = False

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.p_net(x)
        loss = F.nll_loss(logits, y)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {"loss": loss, "acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.p_net(x)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()
        self.log('test_acc', acc, prog_bar=True)

    # ------------------ 以下为手动训练 g-network 的方法示例 ------------------
    def train_gnetwork_once(self, fraction_g=0.1):
        """
        部分训练 g-network: 抽取 fraction_g 比例的权重, 生成 (输入,目标) 来做一次或几次更新.
        """
        # 1) 获取 p-network 当前的权重
        w1 = self.p_net.fc1.weight.data.detach().clone()  # [800, 784]
        b1 = self.p_net.fc1.bias.data.detach().clone()    # [800]
        w2 = self.p_net.fc2.weight.data.detach().clone()  # [10, 800]

        num_w1 = w1.numel()
        num_b1 = b1.numel()
        num_w2 = w2.numel()

        # 2) 准备 (i, j) 索引对的二进制表示
        #    比如对 w1: shape (800, 784), i in [0..799], j in [0..783]
        #    简化：这里直接随机抽样 fraction_g * num_w1 个权重来训练
        #    真实做法可按论文构造 Gray code, 但演示目的下我们可以用简单二进制或直接 idx.
        # ---------------------
        n_samp_w1 = int(num_w1 * fraction_g)
        idx_w1 = th.randint(0, num_w1, size=(n_samp_w1,))
        # batch_w1 的输入是 20 bits (i, j), 此处为了简单演示，我们用 idx // width, idx % width
        # 784 -> 10 bits  (≈ 2^10=1024 >784)
        # 800 -> 10 bits  (≈ 2^10=1024 >800)
        # 这里就演示用 int -> binary, 不做 Gray code
        def idx_to_bin(i, bits=10):
            return [(i >> b) & 1 for b in range(bits)]

        # w1 shape = (800,784), Flatten 后 [800*784] => i in [0..799], j in [0..783]
        h, w = w1.shape
        i_coord = (idx_w1 // w).tolist()
        j_coord = (idx_w1 % w).tolist()

        # 组装 20bits
        input_bits_w1 = []
        for (ii, jj) in zip(i_coord, j_coord):
            ibin = idx_to_bin(ii, 10)
            jbin = idx_to_bin(jj, 10)
            input_bits_w1.append(ibin + jbin)
        batch_w1 = th.tensor(input_bits_w1, dtype=th.float32, device=self.device)
        target_w1 = w1.view(-1)[idx_w1].to(self.device)

        # 3) 同理构造 w2, b1
        n_samp_w2 = int(num_w2 * fraction_g)
        idx_w2 = th.randint(0, num_w2, size=(n_samp_w2,))
        h2, w2_ = w2.shape
        i_coord2 = (idx_w2 // w2_).tolist()
        j_coord2 = (idx_w2 % w2_).tolist()
        input_bits_w2 = []
        for (ii, jj) in zip(i_coord2, j_coord2):
            ibin = idx_to_bin(ii, 10)
            jbin = idx_to_bin(jj, 10)
            input_bits_w2.append(ibin + jbin)
        batch_w2 = th.tensor(input_bits_w2, dtype=th.float32, device=self.device)
        target_w2 = w2.view(-1)[idx_w2].to(self.device)

        # b1 shape = [800], idx 0~799, 只需 10 bits
        n_samp_b1 = int(num_b1 * fraction_g)
        idx_b1 = th.randint(0, num_b1, size=(n_samp_b1,))
        input_bits_b1 = []
        for ii in idx_b1:
            ibin = idx_to_bin(ii.item(), 10)
            input_bits_b1.append(ibin)
        batch_b1 = th.tensor(input_bits_b1, dtype=th.float32, device=self.device)
        target_b1 = b1[idx_b1].to(self.device)

        # 4) 前向 & 反向更新
        optimizer_p, optimizer_g = self.optimizers()
        # 先让 g-optimizer 生效
        optimizer_g.zero_grad()
        loss_g = self._gnetwork_training_step(
            batch_w1, target_w1,
            batch_w2, target_w2,
            batch_b1, target_b1
        )
        self.manual_backward(loss_g)
        optimizer_g.step()

        self.log("train_loss_g", loss_g)

    def apply_gnets(self, alpha=0.0):
        """
        用 g-network 生成所有权重, 再与当前 p-network 的权重融合:
            W_new = alpha * W_old + (1 - alpha) * W_generated
        alpha 可理解为一个衰减或混合系数.
        """
        with th.no_grad():
            # 生成 W1
            w1_shape = self.p_net.fc1.weight.shape
            n_w1 = w1_shape[0]*w1_shape[1]
            coords = th.arange(n_w1, device=self.device)
            # 构造 input bits
            h, w = w1_shape
            i_coord = coords // w
            j_coord = coords % w
            def idx_to_bin(i, bits=10):
                return [(i >> b) & 1 for b in range(bits)]

            input_bits_w1 = []
            for ii, jj in zip(i_coord, j_coord):
                ibin = idx_to_bin(ii.item(), 10)
                jbin = idx_to_bin(jj.item(), 10)
                input_bits_w1.append(ibin+jbin)
            batch_w1 = th.tensor(input_bits_w1, dtype=th.float32, device=self.device)
            gen_w1 = self.g_w1(batch_w1)  # [n_w1]
            gen_w1 = gen_w1.view(*w1_shape)

            # 融合
            old_w1 = self.p_net.fc1.weight.data
            new_w1 = alpha*old_w1 + (1-alpha)*gen_w1
            self.p_net.fc1.weight.data.copy_(new_w1)

            # 同理 w2
            w2_shape = self.p_net.fc2.weight.shape
            n_w2 = w2_shape[0]*w2_shape[1]
            coords2 = th.arange(n_w2, device=self.device)
            h2, w2_ = w2_shape
            i_coord2 = coords2 // w2_
            j_coord2 = coords2 % w2_
            input_bits_w2 = []
            for ii, jj in zip(i_coord2, j_coord2):
                ibin = idx_to_bin(ii.item(), 10)
                jbin = idx_to_bin(jj.item(), 10)
                input_bits_w2.append(ibin+jbin)
            batch_w2 = th.tensor(input_bits_w2, dtype=th.float32, device=self.device)
            gen_w2 = self.g_w2(batch_w2)
            gen_w2 = gen_w2.view(*w2_shape)
            old_w2 = self.p_net.fc2.weight.data
            new_w2 = alpha*old_w2 + (1-alpha)*gen_w2
            self.p_net.fc2.weight.data.copy_(new_w2)

            # b1
            b1_shape = self.p_net.fc1.bias.shape
            n_b1 = b1_shape[0]
            coords_b1 = th.arange(n_b1, device=self.device)
            input_bits_b1 = []
            for ii in coords_b1:
                ibin = idx_to_bin(ii.item(), 10)
                input_bits_b1.append(ibin)
            batch_b1 = th.tensor(input_bits_b1, dtype=th.float32, device=self.device)
            gen_b1 = self.g_b1(batch_b1)
            old_b1 = self.p_net.fc1.bias.data
            new_b1 = alpha*old_b1 + (1-alpha)*gen_b1
            self.p_net.fc1.bias.data.copy_(new_b1)

# ===================== 主函数 / 训练流程 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--g_generations", type=int, default=10, help="多少个外层迭代, 类似Algorithm 1.")
    parser.add_argument("--alpha", type=float, default=0.3, help="融合系数, p-network与g-network输出的权重混合比例")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # 数据
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = MNIST(root="./datasets", train=True, download=True, transform=transform)
    mnist_test  = MNIST(root="./datasets", train=False, download=True, transform=transform)

    # 注意可以定义 validation subset
    train_len = len(mnist_train)
    val_len   = 5000
    val_indices   = list(range(train_len - val_len, train_len))
    train_indices = list(range(0, train_len - val_len))

    train_subset = Subset(mnist_train, train_indices)
    val_subset   = Subset(mnist_train, val_indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_subset,   batch_size=args.batch_size, num_workers=4)
    test_loader  = DataLoader(mnist_test,   batch_size=args.batch_size, num_workers=4)

    # LightningModule
    model = LightningPGModule(
        fraction_p=0.1, fraction_g=0.1,
        lr_p=1e-3, lr_g=1e-3,
        generations=args.g_generations
    )

    # Trainer
    if args.device == "auto":
        accelerator = "gpu" if th.cuda.is_available() else "cpu"
    else:
        accelerator = args.device

    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=args.max_epochs,
        precision=32,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode='min')],
        enable_checkpointing=False,  # 简化演示
        log_every_n_steps=10
    )

    print("==== 进入自定义交替训练循环 ====")
    # 我们手动模拟 Algorithm 1: 迭代 g_generations 次
    for gen in range(args.g_generations):
        print(f"\n[Generation {gen+1}/{args.g_generations}]")

        # 1) 用 fraction_p 的 epoch 训练 p-network (这里直接用 trainer 跑若干 epoch, 或者只跑小步)
        #    简化: 在 lightning 里可以用多个 epoch，但我们可以只做1个小 epoch.
        trainer.fit(model, train_loader, val_loader)
        # 训练结束后, p-network 的权重更新了.

        # 2) 训练 g-network, 使其逼近 p-network 当前权重
        #    这里可以手动写几次 step, 模拟 "fraction_g * total_parameters".
        n_steps_g = 5
        model.train()
        for step in range(n_steps_g):
            model.train_gnetwork_once(fraction_g=model.fraction_g)

        # 3) 用 g-network 的输出覆盖(或部分融合) p-network 的权重
        model.apply_gnets(alpha=args.alpha)

    print("\n==== 最终完整训练若干 epoch (可选) ====")
    # 如果希望在最终再做一个 p-network 的充分微调:
    trainer.fit(model, train_loader, val_loader)

    print("==== 测试集评估 ====")
    trainer.test(model, test_loader)

    print("Done.")

if __name__ == "__main__":
    main()
