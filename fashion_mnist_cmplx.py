
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
    th.set_float32_matmul_precision('high')
elif th.backends.mps.is_available():
    accelerator = 'mps'
else:
    accelerator = 'cpu'


class OptAEGV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.vx = nn.Parameter(th.rand(1, 1, 1, dtype=th.complex64) / 100)
        self.vy = nn.Parameter(th.rand(1, 1, 1, dtype=th.complex64) / 100)
        self.wx = nn.Parameter(th.rand(1, 1, 1, dtype=th.complex64) / 100)
        self.wy = nn.Parameter(th.rand(1, 1, 1, dtype=th.complex64) / 100)
        self.afactor = nn.Parameter(th.rand(1, 1, dtype=th.complex64) / 100)
        self.mfactor = nn.Parameter(th.rand(1, 1, dtype=th.complex64) / 100)

    def flow(self, dx, dy, data):
        return data * (1 + dy) + dx

    def normalize(self, z):
        rho = th.abs(z)
        theta = th.atan2(th.imag(z), th.real(z))
        return th.tanh(rho) * (th.cos(theta) + 1.0j * th.sin(theta))

    def forward(self, data):
        shape = data.size()
        data = data.flatten(1)
        data = self.normalize(data)

        b = shape[0]
        v = self.flow(self.vx, self.vy, data.view(b, -1, 1))
        w = self.flow(self.wx, self.wy, data.view(b, -1, 1))

        dx = self.afactor * th.sum(v * th.tanh(w), dim=-1)
        dy = self.mfactor * th.tanh(data)
        data = self.flow(dx, dy, data)

        data = self.normalize(data)
        return data.view(*shape)


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
        self.conv1 = nn.Conv2d(1, 40, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=3, padding=1, bias=False, dtype=th.complex64)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(20 * 7 * 7, 10, bias=False, dtype=th.complex64)
        self.fc2 = nn.Linear(20, 10, bias=False)
        self.act01 = OptAEGV3()
        self.act02 = OptAEGV3()
        self.act03 = OptAEGV3()
        self.act04 = OptAEGV3()

    def pooling(self, z):
        return self.pool(th.real(z)) + 1.0j * self.pool(th.imag(z))

    def forward(self, x):
        x = self.conv1(x + th.rand_like(x) * 1e-4)
        x = self.act01(x[:, :20] + 1.0j * x[:, 20:])
        x = self.act02(self.conv2(x))
        x = self.pooling(x)
        x = self.act03(x)
        x = self.pooling(x)
        x = x.view(-1, 20 * 7 * 7)
        x = self.fc1(x)
        x = th.cat((th.real(x), th.imag(x)), dim=1)
        x = self.fc2(x)
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
        th.save(model, 'fashion-mnist-optaeg.pt')


if __name__ == '__main__':
    print('loading data...')
    from torch.utils.data import DataLoader
    from torchvision.datasets import FashionMNIST
    from torchvision import transforms

    mnist_train = FashionMNIST('datasets', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    mnist_test = FashionMNIST('datasets', train=False, download=True, transform=transforms.Compose([
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
    model = MNIST_CNN()

    print('training...')
    trainer.fit(model, train_loader, val_loader)

    print('testing...')
    test_best()
