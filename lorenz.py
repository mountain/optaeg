import numpy as np
import torch as th
import torch.nn.functional as F
import lightning as ltn
import argparse
import lightning.pytorch as pl

from torch import Tensor
from torch import nn
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from dysts.flows import Lorenz, Rossler, Thomas

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


class DynamicModel(ltn.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 1e-3
        self.counter = 0
        self.labeled_loss = 0

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, 53)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch[:, :30, :], train_batch[:, 30:, :]
        z = self.forward(x)
        loss = F.mse_loss(z, y)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch[:, :30, :], val_batch[:, 30:, :]
        z = self.forward(x)
        loss = F.mse_loss(z, y)
        self.log('val_loss', loss, prog_bar=True)

        self.labeled_loss += loss.item() * y.size()[0]
        self.counter += y.size()[0]

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch[:, :30, :], test_batch[:, 30:, :]
        z = self.forward(x)
        loss = F.mse_loss(z, y)
        self.labeled_loss += loss.item() * y.size()[0]
        self.counter += y.size()[0]

    def on_save_checkpoint(self, checkpoint) -> None:
        import glob, os

        loss = self.labeled_loss / self.counter
        record = '%09.4f-%03d.ckpt' % (loss, checkpoint['epoch'])
        fname = 'best-%s' % record
        with open(fname, 'bw') as f:
            th.save(checkpoint, f)
        for ix, ckpt in enumerate(sorted(glob.glob('best-*.ckpt'), reverse=False)):
            if ix > 5:
                os.unlink(ckpt)

        self.counter = 0
        self.labeled_loss = 0

        print()


class Dynamic_OptAEGV1(DynamicModel):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(300, 2 * 300, bias=False)
        self.nn0 = OptAEGV1()
        self.fc1 = nn.Linear(2 * 300, 2 * 300)
        self.nn1 = OptAEGV1()
        self.fc2 = nn.Linear(2 * 300, 2 * 300)
        self.nn2 = OptAEGV1()
        self.fc3 = nn.Linear(2 * 300, 300, bias=False)

    def forward(self, x):
        x = x.reshape(-1, 30 * 10)
        x = self.fc0(x)
        x = self.nn0(x)
        x = self.fc1(x)
        x = self.nn1(x)
        x = self.fc2(x)
        x = self.nn2(x)
        x = self.fc3(x)
        return x.view(-1, 30, 10)


def test_best():
    import glob
    fname = sorted(glob.glob('best-*.ckpt'), reverse=True)[0]
    with open(fname, 'rb') as f:
        model = Dynamic_OptAEGV1()
        checkpoint = th.load(f)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model = model.cpu().double()
        model.eval()

        print('')
        with th.no_grad():
            counter, labeled_loss = 0, 0
            for test_batch in test_loader:
                x, y = test_batch[:, :5, :], test_batch[:, 5:, :]
                x, y = x.cpu(), y.cpu()
                z = model(x)
                loss = F.mse_loss(z, y)
                print('.', end='', flush=True)
                labeled_loss += loss.item()
                counter += 1
                if counter % 100 == 0:
                    print('')
        print('')
        print('Loss: %2.5f' % (labeled_loss / counter))
        th.save(model, 'lorenz-optaeg-v1.pt')


class DynamicDataset(th.utils.data.Dataset):
    def __init__(self, dataset_name):
        import os
        dataset_name = 'data/lorenz-%s.npy' % dataset_name
        if os.path.exists(dataset_name):
            print('load... %s' % dataset_name)
            data = np.load(dataset_name)
        else:
            print('create... %s' % dataset_name)
            print('lorenz')
            dyn = Lorenz()
            data1 = dyn.make_trajectory(6000000)
            print(data1.shape)
            print('rossler')
            dyn = Rossler()
            data2 = dyn.make_trajectory(6000000)
            print(data2.shape)
            print('thomas')
            dyn = Thomas()
            data3 = dyn.make_trajectory(6000000)
            print(data3.shape)
            data1 = data1.reshape(100000, 60, 3)
            data2 = data2.reshape(100000, 60, 3)
            data3 = data3.reshape(100000, 60, 3)
            posi = np.arange(60).reshape(1, 60, 1) * np.ones_like(data1[:, :, 0:1]) / 60
            data = np.concatenate((data1, data2, data3, posi), axis=-1)
            np.save(dataset_name, data)
            print('save... %s' % dataset_name)
        self.data = data

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    print('loading data...')
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision import transforms

    ds_train = DynamicDataset('train')
    ds_valid = DynamicDataset('valid')
    ds_test = DynamicDataset('test')

    train_loader = DataLoader(ds_train, shuffle=True, batch_size=opt.batch, num_workers=8)
    val_loader = DataLoader(ds_valid, batch_size=opt.batch, num_workers=8)
    test_loader = DataLoader(ds_test, batch_size=opt.batch, num_workers=8)

    # training
    print('construct trainer...')
    trainer = pl.Trainer(accelerator=accelerator, precision=64, max_epochs=opt.n_epochs,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=30)])

    print('construct model...')
    model = Dynamic_OptAEGV1()

    print('training...')
    trainer.fit(model, train_loader, val_loader)

    print('testing...')
    test_best()