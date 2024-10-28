# with width multiplier 1.0
# * Original MobileNet:                                88.3% (3.2m parameters)
# * MobileNet + OptAEGV3 + SemiLinear:                 87.3% (3.2m parameters)

import torch as th
import torch.nn.functional as F
import lightning as ltn
import argparse
import lightning.pytorch as pl
import torch_optimizer as optim

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
        return th.sigmoid(aeg_result) * self.proj(input)


class CIFAR10Model(ltn.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 1e-3
        self.counter = 0
        self.labeled_loss = 0
        self.labeled_correct = 0

    def configure_optimizers(self):
        base_optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = optim.Lookahead(base_optimizer, k=5, alpha=0.5)
        optimizer._optimizer_state_dict_pre_hooks = {}
        optimizer._optimizer_state_dict_post_hooks = {}
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(-1, 3, 32, 32)
        y = y.view(-1)
        z = self.forward(x)
        loss = F.nll_loss(z, y)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(-1, 3, 32, 32)
        y = y.view(-1)

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
        x = x.view(-1, 3, 32, 32)
        y = y.view(-1)
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


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, width_multiplier=1.0):
        super(DepthwiseSeparableConv, self).__init__()
        # Apply width multiplier to adjust the number of channels
        in_channels = int(in_channels * width_multiplier)
        out_channels = int(out_channels * width_multiplier)

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                                   groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # AEG activation functions
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.act1(self.bn1(self.depthwise(x)))
        x = self.act2(self.bn2(self.pointwise(x)))
        return x


class MobileNet(nn.Module):
    def __init__(self, num_classes=10, width_multiplier=1.0):
        super(MobileNet, self).__init__()

        # Initial standard conv layer
        self.conv1 = nn.Conv2d(3, int(32 * width_multiplier), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32 * width_multiplier))

        # Depthwise separable conv blocks
        self.conv2 = DepthwiseSeparableConv(32, 64, stride=1, width_multiplier=width_multiplier)  # No downsampling
        self.conv3 = DepthwiseSeparableConv(64, 128, stride=2, width_multiplier=width_multiplier)  # Downsample
        self.conv4 = DepthwiseSeparableConv(128, 128, stride=1, width_multiplier=width_multiplier)  # No downsampling
        self.conv5 = DepthwiseSeparableConv(128, 256, stride=2, width_multiplier=width_multiplier)  # Downsample
        self.conv6 = DepthwiseSeparableConv(256, 256, stride=1, width_multiplier=width_multiplier)  # No downsampling
        self.conv7 = DepthwiseSeparableConv(256, 512, stride=2, width_multiplier=width_multiplier)  # Downsample

        # Multiple depthwise separable conv layers with the same size
        self.conv_blocks = nn.Sequential(
            DepthwiseSeparableConv(512, 512, stride=1, width_multiplier=width_multiplier),
            DepthwiseSeparableConv(512, 512, stride=1, width_multiplier=width_multiplier),
            DepthwiseSeparableConv(512, 512, stride=1, width_multiplier=width_multiplier),
            DepthwiseSeparableConv(512, 512, stride=1, width_multiplier=width_multiplier),
            DepthwiseSeparableConv(512, 512, stride=1, width_multiplier=width_multiplier)
        )

        self.conv8 = DepthwiseSeparableConv(512, 1024, stride=2, width_multiplier=width_multiplier)  # Downsample
        self.conv9 = DepthwiseSeparableConv(1024, 1024, stride=1, width_multiplier=width_multiplier)  # No downsampling

        # Global Average Pooling + Fully Connected Layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(1024 * width_multiplier), num_classes)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv_blocks(x)  # 5 repeated blocks
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.avg_pool(x)
        x = th.flatten(x, 1)
        x = self.fc(x)
        return x


class CIFAR10_Mobile(CIFAR10Model):
    def __init__(self):
        super().__init__()
        self.model = MobileNet(num_classes=10, width_multiplier=0.17)

    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x


def test_best():
    import glob
    fname = sorted(glob.glob('best-*.ckpt'), reverse=True)[0]
    with open(fname, 'rb') as f:
        model = CIFAR10_Mobile()
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
                x = x.view(-1, 3, 32, 32)
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
        th.save(model, 'cifar10-mobile.pt')


if __name__ == '__main__':
    print('loading data...')
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10
    from torchvision import transforms

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar10_train = CIFAR10('datasets', train=True, download=True, transform=transform_train)
    cifar1_test = CIFAR10('datasets', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(cifar10_train, shuffle=True, batch_size=opt.batch, num_workers=8)
    val_loader = DataLoader(cifar1_test, batch_size=opt.batch, num_workers=8)
    test_loader = DataLoader(cifar1_test, batch_size=opt.batch, num_workers=8)

    # training
    print('construct trainer...')
    trainer = pl.Trainer(accelerator=accelerator, precision=32, max_epochs=opt.n_epochs,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=30)])

    print('construct model...')
    model = CIFAR10_Mobile()

    print('training...')
    trainer.fit(model, train_loader, val_loader)

    print('testing...')
    test_best()
