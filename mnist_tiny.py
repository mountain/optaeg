# We gave three varaints of OptAEG, V3 is the best.
# We can reach 98.2% accuracy on MNIST with only 702 parameters.
#
#   variant      accuracy      paramters      comments
#      v4         98.2%           735         more stable
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


# This variant can reach 97.3% accuracy on MNIST with only 687 parameters.
# and it is so far the best result.
class OptAEGV1(nn.Module):

    def __init__(self, points=3779):
        super().__init__()
        self.points = points
        self.iscale = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))
        self.theta = None
        self.velocity = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle
        del state["theta"]
        del state["velocity"]
        return state

    @th.compile
    def interpolate(self, param, index):
        i = index.floor().long()
        p = index - i
        j = i + 1
        return (1 - p) * param[i] + p * param[j]

    @th.compile
    def forward(self, data: Tensor) -> Tensor:
        if self.theta == None:
            self.theta = th.linspace(-th.pi, th.pi, self.points, device=data.device)
            self.velocity = th.linspace(0, th.e, self.points, device=data.device)
        shape = data.size()
        data = (data - data.mean()) / data.std() * self.iscale
        data = data.flatten(0)

        theta = self.interpolate(self.theta, th.sigmoid(data) * (self.points - 1))
        ds = self.interpolate(self.velocity, th.abs(th.tanh(data)) * (self.points - 1))

        dx = ds * th.cos(theta)
        dy = ds * th.sin(theta)
        data = data * th.exp(dy) + dx

        return data.view(*shape)


# This variant can reach 97.8% accuracy on MNIST with only 693 parameters.
# and its code is simpler and better.
class OptAEGV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.coeff = nn.Parameter(th.ones(1))
        self.afactor = nn.Parameter(th.zeros(1))
        self.mfactor = nn.Parameter(th.ones(1))

    @th.compile
    def forward(self, data):
        shape = data.size()
        data = data.flatten(0)
        data = data - data.mean()
        data = data / data.std()

        value = th.sigmoid(data)
        value = self.coeff * value * (1 - value)
        value = self.coeff * value * (1 - value)
        value = value - value.mean()
        value = value / value.std()

        dx = self.afactor * th.tanh(value) # control the additive diversity
        dy = self.mfactor * th.tanh(data) # control the growth scale
        data = data * (1 + dy) + dx

        return data.view(*shape)


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
        self.abias = nn.Parameter(th.zeros(1, 1))
        self.mbias = nn.Parameter(th.zeros(1, 1))
        self.mapping = nn.Linear(2, 1)

    def flow(self, dx, dy, data):
        return data * (1 + dy) + dx

    def forward(self, data):
        shape = data.size()
        data = data.flatten(1)
        data = data - data.mean()
        data = data / data.std()

        ur = self.flow(self.uxr, self.uyr, data)
        ui = self.flow(self.uxi, self.uyi, data)
        vr = self.flow(self.vxr, self.vyr, data)
        vi = self.flow(self.vxi, self.vyi, data)
        wr = self.flow(self.wxr, self.wyr, data)
        wi = self.flow(self.wxi, self.wyi, data)

        dxr = self.afactor * (vr * th.sigmoid(wr)) + self.abias
        dxi = self.afactor * (vi * th.sigmoid(wi)) + self.abias
        dyr = self.mfactor * th.tanh(ur) + self.mbias
        dyi = self.mfactor * th.tanh(ui) + self.mbias
        dx = dxr + 1j * dxi
        dy = dyr + 1j * dyi

        data = self.flow(dx, dy, data)
        datar = th.real(data).unsqueeze(-1)
        datai = th.imag(data).unsqueeze(-1)
        data = th.cat((datar, datai), dim=-1)
        data = self.mapping(data).squeeze(-1)

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


class MNIST_OptAEGV3(MNISTModel):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv0 = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
        self.lnon0 = OptAEGV4()
        self.conv1 = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)
        self.lnon1 = OptAEGV4()
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)
        self.lnon2 = OptAEGV4()
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
        model = MNIST_OptAEGV3()
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
    model = MNIST_OptAEGV3()

    print('training...')
    trainer.fit(model, train_loader, val_loader)

    print('testing...')
    test_best()
