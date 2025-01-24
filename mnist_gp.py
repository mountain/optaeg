import argparse
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy

# ---------------- p-network ----------------
class PNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200, bias=True)
        self.fc2 = nn.Linear(200, 10,  bias=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)    
        x = F.relu(self.fc1(x))      
        x = self.fc2(x)              
        return F.log_softmax(x, dim=1)

# ---------------- g-network ----------------
class GNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, coords: th.Tensor):
        # coords: [N,3] (param_idx, row, col)
        out = self.net(coords)
        return out.squeeze(-1)

# --------------- LightningModule ---------------
class LightningPGModule(pl.LightningModule):
    def __init__(self, alpha=0.5, lr_p=1e-3, lr_g=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.p_net = PNetwork()
        self.g_net = GNetwork()
        self.alpha = alpha
        self.lr_p  = lr_p
        self.lr_g  = lr_g

        # 记录所有待生成的参数 (param_idx, tensor)
        self.params_meta = []
        self.params_meta.append(("fc1.weight", self.p_net.fc1.weight))
        self.params_meta.append(("fc1.bias",   self.p_net.fc1.bias))
        self.params_meta.append(("fc2.weight", self.p_net.fc2.weight))
        self.params_meta.append(("fc2.bias",   self.p_net.fc2.bias))

        # 收集所有坐标: (param_idx, row, col)
        all_coords = []
        for i, (name, tensor) in enumerate(self.params_meta):
            shape = tensor.shape
            if len(shape) == 2:  
                rows = th.arange(shape[0])
                cols = th.arange(shape[1])
                mesh = th.stack(th.meshgrid(rows, cols, indexing="ij"), dim=-1)
                mesh = mesh.view(-1,2)
                i_tensor = th.full((mesh.shape[0],1), i)
                coords_i = th.cat([i_tensor, mesh], dim=1)
                all_coords.append(coords_i)
            else:
                # 1D bias
                rows = th.arange(shape[0]).view(-1,1)
                i_tensor = th.full((rows.shape[0],1), i)
                col_zeros = th.zeros(rows.shape[0],1)
                coords_i = th.cat([i_tensor, rows, col_zeros], dim=1)
                all_coords.append(coords_i)

        # 注册为缓冲区 (但用 nn.Parameter(..., requires_grad=False) 也可以)
        self.all_coords = nn.Parameter(
            th.cat(all_coords, dim=0),
            requires_grad=False
        )

        self.generation = 0

    def configure_optimizers(self):
        opt_p = th.optim.Adam(self.p_net.parameters(), lr=self.lr_p)
        opt_g = th.optim.Adam(self.g_net.parameters(), lr=self.lr_g)
        return [opt_p, opt_g]

    def forward(self, x):
        return self.p_net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        opt_p, opt_g = self.optimizers()

        # 1) 更新 p-network
        for _ in range(10):
            opt_p.zero_grad()
            pred = self.p_net(x)
            loss_p = F.nll_loss(pred, y)
            self.manual_backward(loss_p)
            opt_p.step()

        # 2) 随机抽取一部分坐标, 拟合 p-network 的当前权重
        all_num = self.all_coords.size(0)
        n_samp = int(all_num * 0.05)
        idx = th.randperm(all_num, device=self.device)[:n_samp]
        coords_sample = self.all_coords[idx]  # shape [n_samp,3]

        param_idx = coords_sample[:,0].long()
        row_idx   = coords_sample[:,1].long()
        col_idx   = coords_sample[:,2].long()

        target_w = []
        for i in range(coords_sample.shape[0]):
            p_i = param_idx[i].item()
            r   = row_idx[i].item()
            c   = col_idx[i].item()
            _, tensor = self.params_meta[p_i]
            if len(tensor.shape) == 2:
                target_w.append(tensor.data[r,c].unsqueeze(0))
            else:
                target_w.append(tensor.data[r].unsqueeze(0))
        target_w = th.cat(target_w).to(self.device)

        opt_g.zero_grad()
        pred_w = self.g_net(coords_sample.float())
        loss_g = F.mse_loss(pred_w, target_w)
        self.manual_backward(loss_g)
        opt_g.step()

        # 3) alpha 融合
        with th.no_grad():
            all_pred_w = self.g_net(self.all_coords.float())
            for i, (name, tensor) in enumerate(self.params_meta):
                shape = tensor.shape
                coords_i = (self.all_coords[:,0] == i)
                pred_w_i = all_pred_w[coords_i]
                if len(shape) == 2:
                    pred_w_i = pred_w_i.view(shape)
                else:
                    pred_w_i = pred_w_i.view(shape[0])
                old_w = tensor.data
                new_w = self.alpha * old_w + (1 - self.alpha) * pred_w_i
                tensor.data.copy_(new_w)

            self.generation += 1

        self.log('train_loss_p', loss_p, on_step=True, prog_bar=True)
        self.log('train_loss_g', loss_g, on_step=True, prog_bar=True)
        return loss_p

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # ----- (A) 原始 p_net 的验证 -----
        pred = self.p_net(x)
        loss = F.nll_loss(pred, y)
        acc  = (pred.argmax(dim=1) == y).float().mean()

        # ----- (B) 纯 g-net 生成的 p_net2 的验证 -----
        with th.no_grad():
            # 1) 新建一个 p_net2
            p_net2 = PNetwork().to(self.device)

            # 2) 从 g_net 里取出所有预测权重，覆盖 p_net2
            all_pred_w = self.g_net(self.all_coords.float())
            for i, (name, tensor) in enumerate(p_net2.state_dict().items()):
                # 注意: p_net2.state_dict() 里包含 weight/bias, 
                #       顺序可能和 self.params_meta 不完全相同，需要自己对照
                # 这里简化假设顺序一致: [fc1.weight, fc1.bias, fc2.weight, fc2.bias]
                # 如果不一致，就得用 name 匹配
                g_coords_i = (self.all_coords[:,0] == i)
                w_shape = tensor.shape
                w_pred  = all_pred_w[g_coords_i]
                if len(w_shape) == 2:
                    w_pred = w_pred.view(w_shape)
                else:
                    w_pred = w_pred.view(w_shape[0])

                # 用 copy_ 覆盖到 p_net2 当前参数
                p_net2.state_dict()[name].copy_(w_pred)

            # 3) 用 p_net2 做预测
            pred2 = p_net2(x)
            loss2 = F.nll_loss(pred2, y)
            acc2  = (pred2.argmax(dim=1) == y).float().mean()

        # 记录日志
        self.log('val_loss',   loss,  prog_bar=True)
        self.log('val_acc',    acc,   prog_bar=True)
        self.log('val_loss_g', loss2, prog_bar=True)
        self.log('val_acc_g',  acc2,  prog_bar=True)

        # check if we are at rank 0
        if self.global_rank == 0:
            print()

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        p_net2 = PNetwork().to(self.device)

        all_pred_w = self.g_net(self.all_coords.float())
        for i, (name, tensor) in enumerate(p_net2.state_dict().items()):
            g_coords_i = (self.all_coords[:,0] == i)
            w_shape = tensor.shape
            w_pred  = all_pred_w[g_coords_i]
            if len(w_shape) == 2:
                w_pred = w_pred.view(w_shape)
            else:
                w_pred = w_pred.view(w_shape[0])

            p_net2.state_dict()[name].copy_(w_pred)

        pred2 = p_net2(x)
        loss2 = F.nll_loss(pred2, y)
        acc2  = (pred2.argmax(dim=1) == y).float().mean()
        self.log('test_loss_p',   loss2,  prog_bar=True)
        self.log('test_acc_p',    acc2,   prog_bar=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST("./data", train=True,  download=True, transform=transform)
    test_dataset  = MNIST("./data", train=False, download=True, transform=transform)

    val_size = 5000
    train_size = len(train_dataset) - val_size
    train_ds, val_ds = th.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    model = LightningPGModule(alpha=args.alpha, lr_p=1e-3, lr_g=1e-3)

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="auto",
        max_epochs=args.max_epochs,
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()
