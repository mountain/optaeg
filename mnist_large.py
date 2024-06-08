import torch as th
import torch.nn.functional as F
import lightning as ltn
import argparse
import lightning.pytorch as pl

from torch import Tensor
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
    accelerator = 'cpu'
else:
    accelerator = 'cpu'


# This variant can reach 99.6% accuracy on MNIST with only 160K parameters.
class OptAEGV1(nn.Module):

    def __init__(self, points=3779):
        super().__init__()
        self.points = points
        self.iscale = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))
        self.weight = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))
        self.bias = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))
        self.a = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))
        self.b = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))
        self.c = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))
        self.d = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))

    @th.compile
    def forward(self, data: Tensor) -> Tensor:
        shape = data.size()
        data = (data - data.mean()) / data.std() * self.iscale
        data = data.flatten(0)

        dx = th.e * th.tanh(self.weight * data + self.bias)
        dy = th.e * th.tanh(data)

        data1 = data * th.exp(dy) + dx
        data2 = data * th.exp(dy) - dx
        data3 = data * th.exp(- dy) + dx
        data4 = data * th.exp(- dy) - dx

        data = self.a * data1 + self.b * data2 + self.c * data3 + self.d * data4
        return data.view(*shape)


class OptAEGV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.uxr = nn.Parameter(th.zeros(1, 1))
        self.uyr = nn.Parameter(th.ones(1, 1))
        self.uxi = nn.Parameter(th.zeros(1, 1))
        self.uyi = nn.Parameter(th.ones(1, 1))
        self.vxr = nn.Parameter(th.zeros(1, 1))
        self.vyr = nn.Parameter(th.ones(1, 1))
        self.vxi = nn.Parameter(th.zeros(1, 1))
        self.vyi = nn.Parameter(th.ones(1, 1))
        self.wxr = nn.Parameter(th.zeros(1, 1))
        self.wyr = nn.Parameter(th.ones(1, 1))
        self.wxi = nn.Parameter(th.zeros(1, 1))
        self.wyi = nn.Parameter(th.ones(1, 1))
        self.afactor = nn.Parameter(th.zeros(1, 1))
        self.mfactor = nn.Parameter(th.ones(1, 1))
        self.reduce = nn.Linear(3, 3, bias=False)

    def flow(self, dx, dy, data):
        return data * (1 + dy) + dx

    def forward(self, data):
        shape = data.size()
        data = data.flatten(1)

        ur = self.flow(self.uxr, self.uyr, data)
        ui = self.flow(self.uxi, self.uyi, data)
        vr = self.flow(self.vxr, self.vyr, data)
        vi = self.flow(self.vxi, self.vyi, data)
        wr = self.flow(self.wxr, self.wyr, data)
        wi = self.flow(self.wxi, self.wyi, data)

        dxr = self.afactor * (vr * th.sigmoid(wr))
        dxi = self.afactor * (vi * th.sigmoid(wi))
        dyr = self.mfactor * th.tanh(ur)
        dyi = self.mfactor * th.tanh(ui)
        dx = dxr + 1j * dxi
        dy = dyr + 1j * dyi

        flow = self.flow(dx, dy, data)
        flowr = th.real(flow).unsqueeze(-1)
        flowi = th.imag(flow).unsqueeze(-1)
        reduce = self.reduce(th.cat((data.unsqueeze(-1), flowr, flowi), dim=-1))

        base = reduce[:, :, 0]
        base = (base - base.mean()) / base.std()
        result = self.flow(reduce[:, :, 1], reduce[:, :, 2], base)

        return result.view(*shape)


class Conv2d(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=True, kernels_per_layer=1):
        super(Conv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size, padding=padding,
                                   stride=stride, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, 1, padding=0,
                                   stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.lnon = OptAEGV1()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.lnon(x)
        x = self.pointwise(x)
        return x


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


class MNIST_OptAEGV1(MNISTModel):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv0 = nn.Conv2d(1, 16, kernel_size=5, padding=1, bias=False)
        self.lnon0 = OptAEGV4()
        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.lnon1 = OptAEGV4()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.lnon2 = OptAEGV4()
        self.fc = nn.Linear(16 * 3 * 3, 10, bias=False)

    def forward(self, x):
        x = self.conv0(x)
        x = self.lnon0(x)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.lnon1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.lnon2(x)
        x = self.pool(x)
        x = th.flatten(x, 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def test_best():
    import glob
    fname = sorted(glob.glob('best-*.ckpt'), reverse=True)[0]
    with open(fname, 'rb') as f:
        model = MNIST_OptAEGV1()
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
        th.save(model, 'mnist-optaeg-v1.pt')


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
    model = MNIST_OptAEGV1()

    print('training...')
    trainer.fit(model, train_loader, val_loader)

    print('testing...')
    test_best()
