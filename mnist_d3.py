# We propose a new architecture for AEG Networks
# We can reach 98.2% accuracy on MNIST with only 794 parameters.
#
#   variant      accuracy      paramters
#      aeg
#      v3         98.2%           702
#      v2         97.8%           693
#      v1         97.3%           687

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


class OptAEGD3(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(th.zeros(1, 1))
        self.ux = nn.Parameter(th.zeros(1, 1))
        self.uy = nn.Parameter(th.zeros(1, 1))
        self.uz = nn.Parameter(th.zeros(1, 1))
        self.vx = nn.Parameter(th.zeros(1, 1))
        self.vy = nn.Parameter(th.zeros(1, 1))
        self.vz = nn.Parameter(th.zeros(1, 1))
        self.wx = nn.Parameter(th.zeros(1, 1))
        self.wy = nn.Parameter(th.zeros(1, 1))
        self.wz = nn.Parameter(th.zeros(1, 1))
        self.afactor = nn.Parameter(th.zeros(1, 1))
        self.mfactor = nn.Parameter(th.zeros(1, 1))
        self.pfactor = nn.Parameter(th.zeros(1, 1))

    @th.compile
    def flow(self, a, dx, dy, dz):
        log = th.log(th.abs(a))
        return a * (1 + log * dz + (log * log + log) * dz * dz / 2.0) * (1 + dy + dy * dy / 2.0) + dx

    @th.compile
    def forward(self, data):
        shape = data.size()
        data = data.flatten(1)
        data = th.sigmoid(self.alpha * data)

        u = self.flow(data, self.ux, self.uy, self.uz)
        v = self.flow(data, self.vx, self.vy, self.vz)
        w = self.flow(data, self.wx, self.wy, self.wz)

        dx = self.afactor * th.tanh(data) * th.sigmoid(u)
        dy = self.mfactor * th.tanh(data) * th.sigmoid(v)
        dz = self.pfactor * th.tanh(data) * th.sigmoid(w)
        data = self.flow(data, dx, dy, dz)

        return data.view(*shape)


class OptAEGD2(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(th.zeros(1, 1))
        self.ux = nn.Parameter(th.zeros(1, 1))
        self.uy = nn.Parameter(th.zeros(1, 1))
        self.vx = nn.Parameter(th.zeros(1, 1))
        self.vy = nn.Parameter(th.zeros(1, 1))
        self.wx = nn.Parameter(th.zeros(1, 1))
        self.wy = nn.Parameter(th.zeros(1, 1))
        self.afactor = nn.Parameter(th.zeros(1, 1))
        self.mfactor = nn.Parameter(th.zeros(1, 1))

    def flow(self, a, dx, dy):
        return a * (1 + dy + dy * dy / 2.0) + dx + 0.25 * dx * dy

    def forward(self, data):
        shape = data.size()
        data = data.flatten(1)
        data = th.tanh(self.alpha * data)

        u = self.flow(data, self.ux, self.uy)
        v = self.flow(data, self.ux, self.uy)
        w = self.flow(data, self.wx, self.wy)

        dx = self.afactor * u * th.sigmoid(v)
        dy = self.mfactor * th.tanh(data) * th.sigmoid(w)
        data = self.flow(data, dx, dy)

        return data.view(*shape)


class OptAEGV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.vx = nn.Parameter(th.zeros(1, 1, 1))
        self.vy = nn.Parameter(th.ones(1, 1, 1))
        self.wx = nn.Parameter(th.zeros(1, 1, 1))
        self.wy = nn.Parameter(th.ones(1, 1, 1))
        self.afactor = nn.Parameter(th.zeros(1, 1))
        self.mfactor = nn.Parameter(th.ones(1, 1))

    @th.compile
    def flow(self, dx, dy, data):
        return data * (1 + dy) + dx

    @th.compile
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


class MNISTModel(ltn.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 1e-3
        self.counter = 0
        self.labeled_loss = 0
        self.labeled_correct = 0

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
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
        correct = pred.eq(y.data.view_as(pred)).sum()
        self.log('accuracy', correct / z.size()[0], prog_bar=True)

        self.labeled_loss += loss.item() * y.size()[0]
        self.labeled_correct += correct.item()
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

        # to avoid ddp issues, only process with rank 0 write the checkpoint
        if self.global_rank == 0:
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


class MNIST_AEGConv(MNISTModel):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv0 = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
        self.lnon0 = OptAEGD2()
        self.conv1 = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)
        self.lnon1 = OptAEGD2()
        self.conv2 = nn.Conv2d(4 , 4, kernel_size=3, padding=1, bias=False)
        self.lnon2 = OptAEGD2()
        self.fc = nn.Linear(4 * 3 * 3, 10, bias=False)

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
        model = MNIST_AEGConv()
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
                correct = pred.eq(y.data.view_as(pred)).sum()
                success += correct.item()
                counter += y.size(0)
                if counter % 100 == 0:
                    print('.', end='', flush=True)
        print('')
        print('Accuracy: %2.5f' % (success / counter))
        th.save(model, 'mnist-optaeg-aeg.pt')


if __name__ == '__main__':
    print('loading data...')
    from torch.utils.data import DataLoader, random_split
    from torchvision.datasets import MNIST
    from torchvision import transforms

    dataset = MNIST('datasets', train=True, download=True, transform=transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    mnist_train, mnist_val = random_split(dataset, [train_size, val_size])

    mnist_test = MNIST('datasets', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    train_loader = DataLoader(mnist_train, shuffle=True, batch_size=opt.batch, num_workers=8)
    val_loader = DataLoader(mnist_test, batch_size=opt.batch, num_workers=8)
    test_loader = DataLoader(mnist_test, batch_size=opt.batch, num_workers=8)

    # training
    print('construct trainer...')
    trainer = pl.Trainer(
        strategy='ddp_find_unused_parameters_true',
        accelerator=accelerator, precision=32, max_epochs=opt.n_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=30)])

    print('construct model...')
    model = MNIST_AEGConv()

    print('training...')
    trainer.fit(model, train_loader, val_loader)

    print('testing...')
    test_best()
