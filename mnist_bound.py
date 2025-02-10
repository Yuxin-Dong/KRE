import argparse
import numpy as np
import torch
import pickle
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from backpack import extend, backpack
from backpack.extensions import BatchGrad, Variance
from pyhessian import hessian

parser = argparse.ArgumentParser(description='MLP-MNIST')
parser.add_argument('--mlp_dim', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--randlabel', type=int, default=0)
parser.add_argument('--subset', type=int, default=5000)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batchsize', type=int, default=50)
parser.add_argument('--sigma', type=float, default=0.00001)
parser.add_argument('--gpu', type=int, default=3)
flags = parser.parse_args()

device = torch.device('cuda:%d' % flags.gpu)


class IndexedDataset(Dataset):
    def __init__(self, dataset, size):
        self.dataset = dataset
        self.subset = np.random.choice(len(dataset), size, False)

    def __getitem__(self, index):
        data, target = self.dataset[self.subset[index]]
        return data, target, index

    def __len__(self):
        return len(self.subset)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((14, 14), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Normalize((0.1307,), (0.3081,)),
])
mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=flags.batchsize, shuffle=False)

flip_size = len(mnist_train.targets) * flags.randlabel // 10
flip_index = np.random.choice(len(mnist_train.targets), flip_size, False)
targets = np.array(mnist_train.targets)
targets[flip_index] = torch.randint(0, 10, (flip_size,))
mnist_train.targets = targets


class MLP(nn.Module):
    def __init__(self, dim=128):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(14 * 14, dim)
        self.lin2 = nn.Linear(dim, 10)
        for lin in [self.lin1, self.lin2]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = extend(nn.Sequential(self.lin1, nn.ReLU(), self.lin2))

    def forward(self, x):
        out = x.view(x.shape[0], 14 * 14)
        out = self._main(out)
        return out


mlp_var = MLP(flags.mlp_dim).to(device)


print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))


def pretty_print(*values):
    col_width = 13

    def format_val(v):
        if not isinstance(v, str):
            v = '%.5f' % v
        return v.ljust(col_width)

    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


def train():
    mlp = MLP(flags.mlp_dim).to(device)
    criterion = extend(nn.CrossEntropyLoss())
    optimizer = optim.SGD(mlp.parameters(), lr=flags.lr)
    train_dataset = IndexedDataset(mnist_train, flags.subset)
    train_loader = DataLoader(train_dataset, batch_size=flags.batchsize, shuffle=True)

    minR, maxR = 1e4, 0
    train_record, test_record = [], []

    for epoch in range(flags.epochs):
        mlp.train()
        train_loss, train_total, train_correct = 0, 0, 0
        pretty_print('epoch', 'step', 'loss', 'var', 'cov', 'R', 'acc')
        for step, (inputs, targets, index) in enumerate(train_loader):
            x, y = inputs.to(device), targets.to(device)
            z = mlp(x)
            loss = criterion(z, y)
            minR = min(minR, loss.item() / x.shape[0])
            maxR = max(maxR, loss.item() / x.shape[0])

            optimizer.zero_grad()
            with backpack(BatchGrad()):
                loss.backward()
            optimizer.step()

            _, predicted = z.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()
            train_loss += loss.item()
            if step % 10 == 0:
                var, cov = 0, 0
                for p in mlp.parameters():
                    mat = p.grad_batch.detach().reshape(x.shape[0], -1)
                    mat = torch.cov(mat.T * x.shape[0] * flags.lr) / flags.sigma
                    var += torch.trace(mat).item()
                    torch.diagonal(mat)[:] += 1
                    cov += torch.logdet(mat).item() / 2

                train_record.append({
                    'V': var,
                    'C': cov,
                })

                pretty_print(epoch, step, train_loss / (step + 1), var, cov, maxR - minR, train_correct / train_total)

        print('train epoch %d loss %f acc %f' % (epoch, train_loss / len(train_loader), train_correct / train_total))

        mlp.eval()
        with torch.no_grad():
            test_loss, test_total, test_correct = 0, 0, 0
            for step, (inputs, targets) in enumerate(test_loader):
                x, y = inputs.to(device), targets.to(device)
                z = mlp(x)
                loss = criterion(z, y)

                _, predicted = z.max(1)
                test_total += y.size(0)
                test_correct += predicted.eq(y).sum().item()
                test_loss += loss.item()

            print('test epoch %d loss %f acc %f' % (epoch, test_loss / len(test_loader), test_correct / test_total))

        test_record.append({
            'L': (train_loss / len(train_loader), test_loss / len(test_loader)),
            'acc': (train_correct / train_total, test_correct / test_total),
        })

    val_loader = DataLoader(mnist_test, batch_size=1000, shuffle=False)
    hessian_comp = hessian(mlp, criterion, dataloader=val_loader, device=device)
    hes = hessian_comp.trace()

    with open('record/' + 'SGD_mnist1_%d_%d.pkl' % (flags.randlabel, flags.mlp_dim), 'wb') as fout:
        pickle.dump({
            'train': train_record,
            'test': test_record,
            'R': (maxR - minR) / 2,
            'H': np.mean(hes),
        }, fout)


if __name__ == "__main__":
    train()

