
import torch as th
import torch.nn.functional as F
import lightning as ltn
import argparse
import lightning.pytorch as pl

from torch import nn
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("-b", "--batch", type=int, default=256, help="batch size of training")
parser.add_argument("-m", "--model", type=str, default='mnist0', help="model to execute")
opt = parser.parse_args()

if th.cuda.is_available():
    accelerator = 'gpu'
    th.set_float32_matmul_precision('medium')
elif th.backends.mps.is_available():
    accelerator = 'mps'
else:
    accelerator = 'cpu'


# This variant can reach 98.2% accuracy on MNIST with only 702 parameters.
# and the performance is better and quite stable. It is derived from transformer.
class OptAEGV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.vx = nn.Parameter(th.zeros(1, 1, 1))
        self.vy = nn.Parameter(th.ones(1, 1, 1))
        self.wx = nn.Parameter(th.zeros(1, 1, 1))
        self.wy = nn.Parameter(th.ones(1, 1, 1))
        self.afactor = nn.Parameter(th.zeros(1, 1))
        self.mfactor = nn.Parameter(th.ones(1, 1))

    def flow(self, dx, dy, data):
        return data * (1 + dy) + dx

    def forward(self, data):
        shape = data.size()
        data = data.flatten(1)
        data = data - data.mean()
        data = data / data.std()

        b = shape[0]
        v = self.flow(self.vx, self.vy, data.view(b, -1, 1))
        w = self.flow(self.wx, self.wy, data.view(b, -1, 1))

        dx = self.afactor * th.sum(v * th.sigmoid(w), dim=-1)
        dy = self.mfactor * th.tanh(data)
        data = self.flow(dx, dy, data)

        return data.view(*shape)



def batch_aeg_product_optimized(A, B):
    """
    优化后的 batch_aeg_product 函数
    A: (batch_size, rows, features)
    B: (batch_size, features, cols)
    返回: (batch_size, rows, cols) 的结果
    """
    N, rows, features = A.shape
    _, _, cols = B.shape

    # 初始化结果张量
    result = th.zeros(N, rows, cols, device=A.device, dtype=A.dtype)

    # 创建行和列的索引
    i_indices = th.arange(rows, device=A.device).view(rows, 1).expand(rows, cols)  # (rows, cols)
    j_indices = th.arange(cols, device=A.device).view(1, cols).expand(rows, cols)  # (rows, cols)

    for k in range(features):
        # 计算 mask，其中 mask[i,j] = ((i + j + k) % 2 == 0)
        mask = ((i_indices + j_indices + k) % 2 == 0).to(A.device)  # (rows, cols)

        # 获取 A 和 B 中第 k 个特征
        A_k = A[:, :, k].unsqueeze(2)  # (N, rows, 1)
        B_k = B[:, k, :].unsqueeze(1)  # (N, 1, cols)

        # 根据 mask 更新结果
        # 使用广播机制使 mask 适应 (N, rows, cols)
        mask_broadcast = mask.unsqueeze(0)  # (1, rows, cols)
        mask_broadcast = mask_broadcast.expand(N, rows, cols)  # (N, rows, cols)

        # 计算 (result + x) * y 和 (result + y) * x
        option1 = (result + A_k) * B_k  # 当 mask 为 True 时使用
        option2 = (result + B_k) * A_k  # 当 mask 为 False 时使用

        # 使用 torch.where 根据 mask 选择对应的选项
        result = th.where(mask_broadcast, option1, option2)

    return result


class SemiLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(SemiLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(th.Tensor(1, out_features, in_features))
        self.proj = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)

    def forward(self, input):
        expanded_weight = self.weight.expand(input.size(0), -1, -1)  # (batch_size, out_features, in_features)
        reshaped_input = input.view(input.size(0), input.size(1), 1)  # (batch_size, in_features, 1)
        aeg_result = batch_aeg_product_optimized(expanded_weight, reshaped_input)  # (batch_size, out_features, 1)
        aeg_result = aeg_result.squeeze(2)  # (batch_size, out_features)
        return th.tanh(aeg_result) * self.proj(input)


class MNISTModel(ltn.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 1e-3
        self.counter = 0
        self.labeled_loss = 0
        self.labeled_correct = 0

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, 53)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(-1, 1, 28, 28)
        z = self.forward(x)
        loss = F.nll_loss(z, y)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(-1, 1, 28, 28)

        z = self.forward(x)
        loss = F.nll_loss(z, y)
        self.log('val_loss', loss, prog_bar=True)

        pred = z.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).sum() / y.size()[0]
        self.log('correct_rate', correct, prog_bar=True)

        self.labeled_loss += loss.item() * y.size()[0]
        self.labeled_correct += correct.item() * y.size()[0]
        self.counter += y.size()[0]

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x = x.view(-1, 1, 28, 28)
        z = self(x)

        pred = z.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).sum() / y.size()[0]
        self.log('correct_rate', correct, prog_bar=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        import glob, os

        correct = self.labeled_correct / self.counter
        loss = self.labeled_loss / self.counter
        record = '%2.5f-%03d-%1.5f.ckpt' % (correct, checkpoint['epoch'], loss)
        fname = 'best-%s' % record
        with open(fname, 'bw') as f:
            th.save(checkpoint, f)
        for ix, ckpt in enumerate(sorted(glob.glob('best-*.ckpt'), reverse=True)):
            if ix > 5:
                os.unlink(ckpt)

        self.counter = 0
        self.labeled_loss = 0
        self.labeled_correct = 0

        print()


class MNIST_CNN(MNISTModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.fc = SemiLinear(3 * 3 * 3, 10)
        self.act01 = OptAEGV3()
        self.act02 = OptAEGV3()
        self.act03 = OptAEGV3()
        self.act04 = OptAEGV3()

    def forward(self, x):
        x = self.act01(self.conv1(x))
        x = self.act02(self.conv2(x))
        x = self.pool(x)
        x = self.act03(x)
        x = self.pool(x)
        x = self.act04(x)
        x = self.pool(x)
        x = x.view(-1, 3 * 3 * 3)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def test_best():
    import glob
    fname = sorted(glob.glob('best-*.ckpt'), reverse=True)[0]
    with open(fname, 'rb') as f:
        model = MNIST_CNN()
        checkpoint = th.load(f)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model = model.cpu()
        model.eval()

        print('')
        with th.no_grad():
            counter, success = 0, 0
            for test_batch in test_loader:
                x, y = test_batch
                x, y = x.cpu(), y.cpu()
                x = x.view(-1, 1, 28, 28)
                z = model(x)
                pred = z.data.max(1, keepdim=True)[1]
                correct = pred.eq(y.data.view_as(pred)).sum() / y.size()[0]
                print('.', end='', flush=True)
                success += correct.item()
                counter += 1
                if counter % 100 == 0:
                    print('')
        print('')
        print('Accuracy: %2.5f' % (success / counter))
        th.save(model, 'mnist-optaeg-v3.pt')


if __name__ == '__main__':
    print('loading data...')
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision import transforms

    mnist_train = MNIST('datasets', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    mnist_test = MNIST('datasets', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    train_loader = DataLoader(mnist_train, shuffle=True, batch_size=opt.batch, num_workers=8)
    val_loader = DataLoader(mnist_test, batch_size=opt.batch, num_workers=8)
    test_loader = DataLoader(mnist_test, batch_size=opt.batch, num_workers=8)

    # training
    print('construct trainer...')
    trainer = pl.Trainer(accelerator=accelerator, precision=32, max_epochs=opt.n_epochs,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=30)])

    print('construct model...')
    model = MNIST_CNN()

    print('training...')
    trainer.fit(model, train_loader, val_loader)

    print('testing...')
    test_best()
