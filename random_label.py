import argparse
import pickle
import numpy as np
from backpack import extend, backpack
from backpack.extensions import BatchGrad, Variance
from pyhessian import hessian
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser(description='AlexNet-CIFAR10')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--randlabel', type=int, default=0)
parser.add_argument('--epochs', type=int, default=0)
parser.add_argument('--batchsize', type=int, default=50)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--sgld', action='store_true', default=False)
parser.add_argument('--sigma', type=float, default=0.000001)
flags = parser.parse_args()

device = torch.device('cuda:%d' % flags.gpu)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
cifar_train = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
cifar_test = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
train_loader = DataLoader(cifar_train, batch_size=flags.batchsize, shuffle=True)
test_loader = DataLoader(cifar_test, batch_size=flags.batchsize, shuffle=False)

flip_size = 1000 * flags.randlabel
flip_index = np.random.choice(10000, flip_size, False)
targets = np.array(cifar_train.targets)
targets[flip_index] = torch.randint(0, 10, (flip_size,))
cifar_train.targets = targets


class AlexNet(nn.Module):
    def __init__(self) -> None:
        super(AlexNet, self).__init__()
        self.features = extend(nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ))
        self.classifier = extend(nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.zeros_(m.bias)


net = AlexNet().to(device)
net.apply(init_weights)

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


def batch_cov(points: torch.Tensor):
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)


def train():
    criterion = extend(nn.CrossEntropyLoss())
    optimizer = optim.SGD(net.parameters(), lr=flags.lr)

    minR, maxR = 1e4, 0
    train_record, test_record = [], []

    for epoch in range(flags.epochs):
        net.train()
        train_loss, train_total, train_correct = 0, 0, 0
        pretty_print('epoch', 'step', 'loss', 'var', 'cov', 'R', 'acc')
        for step, (inputs, targets) in enumerate(train_loader):
            x, y = inputs.to(device), targets.to(device)
            z = net(x)
            loss = criterion(z, y)
            minR = min(minR, loss.item())
            maxR = max(maxR, loss.item())

            optimizer.zero_grad()
            with backpack(BatchGrad()):
                loss.backward()
            optimizer.step()

            var, cov, d = 0, 0, 0
            if (step + 1) % 1000 == 0:
                for p in net.parameters():
                    grad = p.grad_batch.detach().reshape(x.shape[0], -1)
                    d += grad.shape[1]
                    print(grad.shape)
                    if grad.shape[1] < 8192:
                        mat = torch.cov(grad.T * x.shape[0] * flags.lr) / flags.sigma
                        var += mat.trace().item()
                        torch.diagonal(mat)[:] += 1
                        _, det = torch.slogdet(mat)
                        cov += det.item() / 2
                    else:
                        for i in range(0, grad.shape[1], 8192):
                            mat = torch.cov(grad[:, i:i + 8192].T * x.shape[0] * flags.lr) / flags.sigma
                            var += mat.trace().item()
                            torch.diagonal(mat)[:] += 1
                            _, det = torch.slogdet(mat)
                            cov += det.item() / 2
                var = d / 2 * np.log(1 + var / d)

            # optimizer.zero_grad()
            # with backpack(Variance()):
            #     loss.backward()
            # optimizer.step()
            #
            # var, cov, d = 0, 0, 0
            # for p in net.parameters():
            #     vt = p.variance.detach().reshape(-1) * (x.shape[0] * flags.lr) ** 2 / flags.sigma
            #     d += vt.shape[0]
            #     var += vt.sum().item()
            #     cov += torch.log(vt + 1).sum().item() / 2
            # var = d / 2 * np.log(1 + var / d)

            if flags.sgld:
                for p in net.parameters():
                    noise = p.data.new(
                        p.data.size()).normal_(mean=0, std=1) * np.sqrt(flags.sigma)
                    p.data.add_(noise)

            _, predicted = z.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()
            train_loss += loss.item()
            pretty_print(epoch, step, train_loss / (step + 1), var, cov, maxR - minR, train_correct / train_total)

            train_record.append({
                'V': var,
                'C': cov,
            })

        print('train epoch %d loss %f acc %f' % (epoch, train_loss / len(train_loader), train_correct / train_total))

        net.eval()
        with torch.no_grad():
            test_loss, test_total, test_correct = 0, 0, 0
            for step, (inputs, targets) in enumerate(test_loader):
                x, y = inputs.to(device), targets.to(device)
                z = net(x)
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

    hes = []
    if not flags.sgld:
        val_loader = DataLoader(cifar_test, batch_size=1000, shuffle=False)
        hessian_comp = hessian(net, criterion, dataloader=val_loader, device=device)
        hes.append(hessian_comp.trace())

    with open('record/' + ('SGLD' if flags.sgld else 'SGD') + '_cifar_rand_%d.pkl' % flags.randlabel, 'wb') as fout:
        pickle.dump({
            'train': train_record,
            'test': test_record,
            'R': (maxR - minR) / 2,
            'H': np.mean(hes),
        }, fout)


if __name__ == "__main__":
    train()
