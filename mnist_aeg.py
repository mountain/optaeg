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


def hyperbolic_inner_product(x1: Tensor, y1: Tensor, x2: Tensor, y2: Tensor) -> Tensor:
    numerator = (x1 - x2) ** 2 + (y1 - y2) ** 2
    denominator = 4 * y1 * y2
    inner_product = numerator / denominator

    return inner_product


class AEGNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linkage_size = input_size * output_size
        self.input_x = nn.Parameter(2 * th.rand(1, output_size, input_size) - 1)
        self.input_y = nn.Parameter(th.rand(1, output_size, input_size)) + 0.1
        self.output_x = nn.Parameter(2 * th.rand(1, output_size, input_size) - 1)
        self.output_y = nn.Parameter(th.ones(1, output_size, input_size)) + 0.1
        self.linkage_add = nn.Parameter(th.rand(1, output_size, input_size))
        self.linkage_mul = nn.Parameter(th.rand(1, output_size, input_size))

    def forward(self, data):
        shape = data.size()
        data = data.view(-1, 1, self.input_size)
        input = self.input_x / self.input_y
        output = self.output_x / self.output_y
        tests = (input + self.linkage_add) * (1 + self.linkage_mul)
        error = (tests - output) * th.sigmoid(data)
        expect = data * th.softmax(1 / th.sqrt(error * error + 1e-7), dim=1)
        return th.sum(expect, dim=2).view(*shape)


class AEGConv(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.input_x = nn.Parameter(2 * th.rand(1, channel_out, channel_in, 1, 1) - 1)
        self.input_y = nn.Parameter(th.rand(1, channel_out, channel_in, 1, 1)) + 0.1
        self.output_x = nn.Parameter(2 * th.rand(1, channel_out, channel_in, 1, 1) - 1)
        self.output_y = nn.Parameter(th.ones(1, channel_out, channel_in, 1, 1)) + 0.1
        self.linkage_add = nn.Parameter(th.rand(1, channel_out, channel_in, 1, 1))
        self.linkage_mul = nn.Parameter(th.rand(1, channel_out, channel_in, 1, 1))

    def forward(self, data):
        b, c, w, h = data.size()
        data = data.view(-1, 1, c, w, h)
        coeff = hyperbolic_inner_product(self.input_x, self.input_y, data, th.ones_like(data))
        inputs = self.input_x / self.input_y
        output = self.output_x / self.output_y
        tests = (inputs + self.linkage_add) * (1 + self.linkage_mul)
        error = (tests - output) * th.sigmoid(data)
        prob = th.softmax(1 / th.sqrt(error * error + 1e-7), dim=1)
        expect = th.sum(coeff * prob, dim=2)
        return expect


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
        correct = pred.eq(y.data.view_as(pred)).sum()
        self.log('accuracy', correct / z.size()[0], prog_bar=True)
        img = x[0].to('cpu').numpy()
        guess = pred[0, 0].item()
        self.logger.experiment.add_image(f'image_{guess}', img)

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
        self.lnon0 = OptAEGV3()
        self.conv1 = AEGConv(4, 4)
        self.lnon1 = OptAEGV3()
        self.conv2 = AEGConv(4 , 4)
        self.lnon2 = OptAEGV3()
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
    trainer = pl.Trainer(accelerator=accelerator, precision=32, max_epochs=opt.n_epochs,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)])

    print('construct model...')
    model = MNIST_AEGConv()

    print('training...')
    trainer.fit(model, train_loader, val_loader)

    print('testing...')
    test_best()
