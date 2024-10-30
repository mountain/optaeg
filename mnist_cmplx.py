# We give four variants of learnable activitions based on AEG theory
# V3 is the best which can reach 98.2% accuracy on MNIST with only 702 parameters.
#
#   variant      accuracy      paramters      comments
#    v3-cmplx     98.2%           645         best performance
#      v3         98.2%           702
#      v4         98.2%           735
#      v2         97.8%           693
#      v1         97.3%           687
#
# AEG stands for arithmetical expression geometry, which is a new theory studying the geometry of arithmetical expressions.
# It opens a new optimization space for neural networks, and can be used to construct a new kind of neural network.
# For the details of AEG theory, please refer to the draft paper:
# * https://github.com/mountain/aeg-paper : Can arithmetical expressions form a geometry space?
# * https://github.com/mountain/aeg-invitation : an invitation to AEG theory
# * https://github.com/mountain/aeg-invitation/blob/main/slides2/aeg.pdf : introductory slides


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
parser.add_argument("-b", "--batch", type=int, default=128, help="batch size of training")
parser.add_argument("-m", "--model", type=str, default='mnist0', help="model to execute")
opt = parser.parse_args()

if th.cuda.is_available():
    accelerator = 'gpu'
    th.set_float32_matmul_precision('high')
elif th.backends.mps.is_available():
    accelerator = 'cpu'
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


class MNIST_OptAEGV3(MNISTModel):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1, bias=False)
        self.lnon0 = OptAEGV3()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=4, stride=2, padding=1, bias=False, dtype=th.complex64)
        self.lnon1 = OptAEGV3()
        self.conv2 = nn.Conv2d(3, 3, kernel_size=4, stride=2, padding=1, bias=False, dtype=th.complex64)
        self.lnon2 = OptAEGV3()
        self.conv3 = nn.Conv2d(3, 3, kernel_size=4, stride=2, padding=1, bias=False, dtype=th.complex64)
        self.lnon3 = OptAEGV3()
        self.fc = nn.Linear(3 * 3 * 3, 5, bias=False, dtype=th.complex64)

    def forward(self, x):
        x = self.conv0(x + th.rand_like(x) * 0.000001) # random noise against numerical instability
        x = x[:, :3] + 1.0j * x[:, 3:]
        x = self.lnon0(x)
        x = self.conv1(x)
        x = self.lnon1(x)
        x = self.conv2(x)
        x = self.lnon2(x)
        x = self.conv3(x)
        x = self.lnon3(x)
        x = th.flatten(x, 1)
        x = self.fc(x)
        x = th.cat((th.real(x), th.imag(x)), dim=1)
        x = F.log_softmax(x, dim=1)
        return x


def test_best():
    import glob
    fname = sorted(glob.glob('best-*.ckpt'), reverse=True)[0]
    with open(fname, 'rb') as f:
        model = MNIST_OptAEGV3()
        checkpoint = th.load(f, weights_only=True)
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
                         callbacks=[EarlyStopping(monitor="correct_rate", mode="max", patience=30)])

    print('construct model...')
    model = MNIST_OptAEGV3()

    print('training...')
    trainer.fit(model, train_loader, val_loader)

    print('testing...')
    test_best()
