import torch as th
import torch.nn.functional as F
import torch.nn as nn
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
elif th.backends.mps.is_available():
    accelerator = 'cpu'
else:
    accelerator = 'cpu'


class OptAEGV1(nn.Module):

    def __init__(self, points=29):
        super().__init__()
        self.points = points
        self.iscale = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))
        self.oscale = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))
        self.theta = th.linspace(-th.pi, th.pi, points)
        self.velocity = th.linspace(0, th.e, points)
        self.weight = nn.Parameter(th.normal(0, 1, (points, points)))

    @th.compile
    def integral(self, param, index):
        return th.sum(param[index].view(-1, 1) * th.softmax(self.weight, dim=1)[index, :], dim=1)

    @th.compile
    def interplot(self, param, index):
        lmt = param.size(0) - 1

        p0 = index.floor().long()
        p1 = p0 + 1
        pos = index - p0
        p0 = p0.clamp(0, lmt)
        p1 = p1.clamp(0, lmt)

        v0 = self.integral(param, p0)
        v1 = self.integral(param, p1)

        return (1 - pos) * v0 + pos * v1

    @th.compile
    def forward(self, data: Tensor) -> Tensor:
        if self.theta.device != data.device:
            self.theta = self.theta.to(data.device)
            self.velocity = self.velocity.to(data.device)
        shape = data.size()
        data = (data - data.mean()) / data.std() * self.iscale
        data = data.flatten(0)

        theta = self.interplot(self.theta, th.sigmoid(data) * (self.points - 1))
        ds = self.interplot(self.velocity, th.abs(th.tanh(data) * (self.points - 1)))

        dx = ds * th.cos(theta)
        dy = ds * th.sin(theta)
        data = data * th.exp(dy) + dx

        data = (data - data.mean()) / data.std() * self.oscale
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
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, 37)
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


class FashionMNIST_OptAEGV1(MNISTModel):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv0 = nn.Conv2d(1, 32, kernel_size=7, padding=3, bias=False)
        self.lnon0 = OptAEGV1()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.lnon1 = OptAEGV1()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.lnon2 = OptAEGV1()
        self.fc1 = nn.Linear(32 * 3 * 3, 50)
        self.lnon3 = OptAEGV1()
        self.fc2 = nn.Linear(50, 10, bias=False)
 
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
        x = self.fc1(x)
        x = self.lnon3(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


def test_best():
    import glob
    fname = sorted(glob.glob('best-*.ckpt'), reverse=True)[0]
    with open(fname, 'rb') as f:
        checkpoint = th.load(f)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.eval()

        with th.no_grad():
            counter, success = 0, 0
            for test_batch in test_loader:
                x, y = test_batch
                x = x.view(-1, 1, 28, 28)
                z = model(x)
                pred = z.data.max(1, keepdim=True)[1]
                correct = pred.eq(y.data.view_as(pred)).sum() / y.size()[0]
                print('.', end='', flush=True)
                if counter % 100 == 0:
                    print('')
                success += correct.item()
                counter += 1
        print('')
        print('Accuracy: %2.5f' % (success / counter))
        th.save(model, 'fashion-mnist-optaeg-v1.pt')


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

    train_loader = DataLoader(mnist_train, shuffle=True, batch_size=opt.batch, num_workers=8)
    val_loader = DataLoader(mnist_test, batch_size=opt.batch, num_workers=8)
    test_loader = DataLoader(mnist_test, batch_size=opt.batch, num_workers=8)

    # training
    print('construct trainer...')
    trainer = pl.Trainer(accelerator=accelerator, precision=32, max_epochs=opt.n_epochs,
                         callbacks=[EarlyStopping(monitor="correct_rate", mode="max", patience=74)])

    print('construct model...')
    model = FashionMNIST_OptAEGV1()

    print('training...')
    trainer.fit(model, train_loader, val_loader)

    print('testing...')
    test_best()
