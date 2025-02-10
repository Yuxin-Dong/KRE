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

parser = argparse.ArgumentParser(description='CNN-CIFAR10')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--randlabel', type=int, default=0)
parser.add_argument('--subset', type=int, default=5000)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--batchsize', type=int, default=50)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--sigma', type=float, default=0.000001)
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
    transforms.Resize((16, 16), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(cifar_test, batch_size=flags.batchsize, shuffle=False)

flip_size = len(cifar_train.targets) * flags.randlabel // 10
flip_index = np.random.choice(len(cifar_train.targets), flip_size, False)
targets = np.array(cifar_train.targets)
targets[flip_index] = torch.randint(0, 10, (flip_size,))
cifar_train.targets = targets


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.lin1 = nn.Linear(48, 48)
        self.lin2 = nn.Linear(48, 10)
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.lin1, self.lin2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        self._cnn = extend(nn.Sequential(self.conv1, nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
                                         self.conv2, nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
                                         self.conv3, nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
                                         self.conv4, nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)))
        self._lin = extend(nn.Sequential(self.lin1, nn.ReLU(), self.lin2))

    def forward(self, x):
        x = self._cnn(x)
        out = x.view(x.shape[0], 48)
        out = self._lin(out)
        return out


cnn_var = CNN().to(device)


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
    cnn = CNN().to(device)
    criterion = extend(nn.CrossEntropyLoss())
    optimizer = optim.SGD(cnn.parameters(), lr=flags.lr)
    train_dataset = IndexedDataset(cifar_train, flags.subset)
    train_loader = DataLoader(train_dataset, batch_size=flags.batchsize, shuffle=True)

    minR, maxR = 1e4, 0
    train_record, test_record = [], []

    for epoch in range(flags.epochs):
        cnn.train()
        train_loss, train_total, train_correct = 0, 0, 0
        pretty_print('epoch', 'step', 'loss', 'var', 'cov', 'R', 'acc')
        for step, (inputs, targets, index) in enumerate(train_loader):
            x, y = inputs.to(device), targets.to(device)
            z = cnn(x)
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
                for p in cnn.parameters():
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

        cnn.eval()
        with torch.no_grad():
            test_loss, test_total, test_correct = 0, 0, 0
            for step, (inputs, targets) in enumerate(test_loader):
                x, y = inputs.to(device), targets.to(device)
                z = cnn(x)
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

    val_loader = DataLoader(cifar_test, batch_size=1000, shuffle=False)
    hessian_comp = hessian(cnn, criterion, dataloader=val_loader, device=device)
    hes = hessian_comp.trace()

    with open('record/' + 'SGD_cifar_randlabel_%d.pkl' % flags.randlabel, 'wb') as fout:
        pickle.dump({
            'train': train_record,
            'test': test_record,
            'R': (maxR - minR) / 2,
            'H': np.mean(hes),
        }, fout)


if __name__ == "__main__":
    train()

