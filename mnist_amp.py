import torch as th
import torch.nn.functional as F
import lightning as ltn
import argparse
import lightning.pytorch as pl

from torch import nn
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from mnist_cnn import SemiLinear

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


def aeg_integrate(i, j, A_row, B_col, initial=0):
    """
    A_row: a row of matrix A
    B_col: a column of matrix B
    return the result of the integration of A_row and B_col
    """
    result = initial  # initialize the result
    for k, (x, y) in enumerate(zip(A_row, B_col)):
        if (i + j + k) % 2 == 0:
            result = result + x
            result = result * y
        else:
            result = result + y
            result = result * x

    return result


def aeg_product(A, B):
    """
    A: matrix A
    B: matrix B
    return the result of product of A and B, where the product is defined as the integration of each row of A and each column of B
    """
    result = th.zeros(A.size(0), B.size(1))
    for i in range(A.size(0)):
        for j in range(B.size(1)):
            result[i, j] = aeg_integrate(i, j, A[i], B[:, j])

    return result


def batch_aeg_product(A, B):
    """
    A: a batch of matrices A
    B: a batch of matrices B
    return the result of product of A and B, where the product is defined as the integration of each row of A and each column of B
    """
    result = th.zeros(A.size(0), A.size(1), B.size(2))
    for i in range(A.size(0)):
        result[i] = aeg_product(A[i], B[i])

    return result


def conv2d_aeg(input, kernel, stride=1, padding=0):
    """
    input: 输入图像，形状为 (N, C, H, W)，其中 N 是批大小，C 是通道数，H 是高度，W 是宽度
    kernel: 卷积核，形状为 (out_channels, in_channels, KH, KW)
    stride: 卷积步幅
    padding: 填充的像素数量
    返回卷积操作的结果
    """
    # 获取输入和卷积核的尺寸
    N, C_in, H, W = input.shape
    out_channels, in_channels, KH, KW = kernel.shape

    # 检查通道数是否匹配
    assert C_in == in_channels, "输入图像的通道数必须和卷积核通道数一致"

    # 进行 padding
    if padding > 0:
        input_padded = th.nn.functional.pad(input, (padding, padding, padding, padding), mode='constant', value=0)
    else:
        input_padded = input

    # 计算输出特征图的尺寸
    out_height = (H - KH + 2 * padding) // stride + 1
    out_width = (W - KW + 2 * padding) // stride + 1

    # 初始化输出张量
    output = th.zeros((N, out_channels, out_height, out_width), device=input.device)

    # 执行卷积操作
    for n in range(N):  # 遍历批大小
        for oc in range(out_channels):  # 遍历输出通道
            for i in range(0, out_height):
                for j in range(0, out_width):
                    # 提取当前的局部区域
                    h_start = i * stride
                    w_start = j * stride
                    h_end = h_start + KH
                    w_end = w_start + KW

                    # 遍历输入通道，进行逐元素卷积
                    for ic in range(in_channels):
                        region = input_padded[n, ic, h_start:h_end, w_start:w_end]
                        output[n, oc, i, j] += aeg_integrate(i, j, region.flatten(), kernel[oc, ic, :, :].flatten())

    return output


def conv2d_aeg_optimized(input, kernel, stride=1, padding=0):
    """
    优化后的 conv2d_aeg 函数，支持批次化操作。
    input: 输入图像，形状为 (N, C_in, H, W)
    kernel: 卷积核，形状为 (out_channels, in_channels, KH, KW)
    stride: 卷积步幅
    padding: 填充的像素数量
    返回: 卷积操作的结果，形状为 (N, out_channels, out_height, out_width)
    """
    N, C_in, H, W = input.shape
    out_channels, in_channels, KH, KW = kernel.shape

    assert C_in == in_channels, "输入图像的通道数必须和卷积核通道数一致"

    # 应用 padding
    if padding > 0:
        input_padded = F.pad(input, (padding, padding, padding, padding), mode='constant', value=0)
    else:
        input_padded = input

    # 计算输出特征图的尺寸
    out_height = (H - KH + 2 * padding) // stride + 1
    out_width = (W - KW + 2 * padding) // stride + 1
    L = out_height * out_width  # 总的空间位置数

    # 使用 unfold 提取所有滑动窗口，形状为 (N, C_in * KH * KW, L)
    input_unfolded = F.unfold(input_padded, kernel_size=(KH, KW), stride=stride)  # (N, C_in*KH*KW, L)

    # 重塑为 (N, C_in, KH*KW, L)
    input_unfolded = input_unfolded.view(N, C_in, KH * KW, L)  # (N, C_in, K, L)

    # 重塑 kernel 为 (out_channels, C_in, K)
    kernel_flat = kernel.view(out_channels, C_in, KH * KW)  # (out_channels, C_in, K)

    # 初始化 result 为零，形状为 (N, out_channels, C_in, L)
    result = th.zeros(N, out_channels, C_in, L, device=input.device, dtype=input.dtype)

    # 计算 i 和 j 的索引
    l_indices = th.arange(L, device=input.device)
    i_indices = (l_indices // out_width).view(1, 1, 1, L)  # (1, 1, 1, L)
    j_indices = (l_indices % out_width).view(1, 1, 1, L)   # (1, 1, 1, L)

    # 计算 k 的索引
    k_indices = th.arange(KH * KW, device=input.device).view(1, 1, KH * KW, 1)  # (1, 1, K, 1)

    # 计算 mask，其中 mask[i,j,k,l] = ((i + j + k) % 2 == 0)
    mask = ((i_indices + j_indices + k_indices) % 2 == 0)  # (1, 1, K, L)

    # 扩展 mask 至 (N, out_channels, C_in, K, L)
    # 只需要两次 unsqueeze，确保最终 mask 具有 5 个维度
    mask = mask.unsqueeze(0)  # (1, 1, 1, K, L)
    mask = mask.expand(N, out_channels, C_in, KH * KW, L)  # (N, out_channels, C_in, K, L)

    # 扩展 input_unfolded 和 kernel_flat 以匹配形状
    # input_unfolded: (N, C_in, K, L) -> (N, 1, C_in, K, L) -> (N, out_channels, C_in, K, L)
    input_expanded = input_unfolded.unsqueeze(1).expand(N, out_channels, C_in, KH * KW, L)  # (N, out_channels, C_in, K, L)

    # kernel_flat: (out_channels, C_in, K) -> (1, out_channels, C_in, K, 1) -> (N, out_channels, C_in, K, L)
    kernel_expanded = kernel_flat.unsqueeze(0).unsqueeze(-1).expand(N, out_channels, C_in, KH * KW, L)  # (N, out_channels, C_in, K, L)

    # 在每个 k 步骤中更新 result
    for k in range(KH * KW):
        # 当前 mask: (N, out_channels, C_in, L)
        current_mask = mask[:, :, :, k, :]  # (N, out_channels, C_in, L)

        # 当前 x 和 y: (N, out_channels, C_in, L)
        x = input_expanded[:, :, :, k, :]  # (N, out_channels, C_in, L)
        y = kernel_expanded[:, :, :, k, :]  # (N, out_channels, C_in, L)

        # 计算 (result + x) * y 和 (result + y) * x
        option1 = (result + x) * y  # (N, out_channels, C_in, L)
        option2 = (result + y) * x  # (N, out_channels, C_in, L)

        # 使用 mask 选择对应的选项
        result = th.where(current_mask, option1, option2)  # (N, out_channels, C_in, L)

    # 对 in_channels 求和，得到最终输出形状为 (N, out_channels, L)
    output = result.sum(dim=2).view(N, out_channels, out_height, out_width)  # (N, out_channels, H_out, W_out)

    return output


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


class FullConection(nn.Module):
    def __init__(self, in_features, out_features):
        super(FullConection, self).__init__()
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


class AEGConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(AEGConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(th.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.reset_parameters()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)

    def forward(self, input):
        return th.sigmoid(conv2d_aeg_optimized(
            input, self.weight, self.stride, self.padding
        )) * self.conv2d(input)


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


class MNIST_AMP(MNISTModel):
    def __init__(self):
        super().__init__()
        self.conv1 = AEGConv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = AEGConv2d(6, 6, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(3)
        self.fc = SemiLinear(6 * 3 * 3, 10)
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
        x = x.view(-1, 6 * 3 * 3)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def test_best():
    import glob
    fname = sorted(glob.glob('best-*.ckpt'), reverse=True)[0]
    with open(fname, 'rb') as f:
        model = MNIST_AMP()
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
        th.save(model, 'mnist-amp.pt')


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

    train_loader = DataLoader(mnist_train, shuffle=True, batch_size=opt.batch, num_workers=8, persistent_workers=True)
    val_loader = DataLoader(mnist_test, batch_size=opt.batch, num_workers=8, persistent_workers=True)
    test_loader = DataLoader(mnist_test, batch_size=opt.batch, num_workers=8, persistent_workers=True)

    # training
    print('construct trainer...')
    trainer = pl.Trainer(accelerator=accelerator, precision=32, max_epochs=opt.n_epochs,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=30)])

    print('construct model...')
    model = MNIST_AMP()

    print('training...')
    trainer.fit(model, train_loader, val_loader)

    print('testing...')
    test_best()
