import argparse
import numpy as np
import torch
import pickle
import scipy.stats
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from backpack import extend, backpack
from backpack.extensions import BatchGrad
from pyhessian import hessian

parser = argparse.ArgumentParser(description='LIN-REGRESS')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--restarts', type=int, default=100)
parser.add_argument('--subset', type=int, default=100)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batchsize', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--sgld', action='store_true', default=False)
parser.add_argument('--sigma', type=float, default=0.001)
flags = parser.parse_args()

device = torch.device('cuda:%d' % flags.gpu)


def gen_mix_norm(miu, sigma):
    sigma = np.sqrt(sigma)

    def gen(size):
        length = len(miu)
        size = tuple(np.atleast_1d(size))
        data = np.zeros((length, *size))
        for i in range(length):
            data[i, :] = scipy.stats.norm.rvs(miu[i], sigma[i], size)
        choice = np.random.choice(length, size)
        return np.choose(choice, data)

    return gen


class RegressionDataset(Dataset):
    def __init__(self, size):
        self.beta = torch.Tensor([-3, -2, 2, 3, 1, 2, -1, -3, 0, 3])
        self.x = torch.Tensor(gen_mix_norm([-3, 3], [4, 10])((size, 10)))

        self.noise = torch.Tensor(scipy.stats.norm.rvs(0, 0.1, size))
        self.y = torch.mm(self.x, self.beta.unsqueeze(1)).squeeze()
        self.y += self.noise

    def __getitem__(self, index):
        return self.x[index, :], self.y[index], index

    def __len__(self):
        return len(self.y)


regress_test = RegressionDataset(flags.subset)
test_loader = DataLoader(regress_test, batch_size=flags.batchsize, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(10, 10)
        self.lin2 = nn.Linear(10, 1)
        for lin in [self.lin1, self.lin2]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = extend(nn.Sequential(self.lin1, nn.ReLU(), self.lin2))

    def forward(self, x):
        return self._main(x).squeeze()


net_var = Net().to(device)


print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))


def train(run):
    net = Net().to(device)
    criterion = extend(nn.MSELoss())
    optimizer = optim.SGD(net.parameters(), lr=flags.lr)
    train_dataset = RegressionDataset(flags.subset)
    train_loader = DataLoader(train_dataset, batch_size=flags.batchsize, shuffle=True)

    minR, maxR = 1e4, 0
    train_record, test_record = [], []
    W0 = [p.data.detach().cpu().numpy() for p in net.parameters()]

    for epoch in range(flags.epochs):
        net.train()
        train_loss = 0
        for step, (inputs, targets, index) in enumerate(train_loader):
            x, y = inputs.to(device), targets.to(device)
            z = net(x)
            loss = criterion(z, y)
            minR = min(minR, loss.item())
            maxR = max(maxR, loss.item())
            train_loss += loss.item()

            optimizer.zero_grad()
            with backpack(BatchGrad()):
                loss.backward()
            optimizer.step()

            cov = torch.concat([p.grad_batch.detach().view(x.shape[0], -1) for p in net.parameters()], dim=1)
            cov = torch.cov(cov.T * x.shape[0] * flags.lr) / flags.sigma
            var = torch.trace(cov).item()
            torch.diagonal(cov)[:] += 1
            cov = torch.logdet(cov)

            if flags.sgld:
                for p in net.parameters():
                    noise = p.data.new(
                        p.data.size()).normal_(mean=0, std=1) * np.sqrt(flags.sigma)
                    p.data.add_(noise)

            W = [p.data.detach().cpu().numpy() for p in net.parameters()]
            train_record.append({
                'W': W,
                'B': index,
                'V': var,
                'C': cov.item() / 2,
            })

        print('train epoch %d loss %f' % (epoch, train_loss / len(train_loader)))

        net.eval()
        with torch.no_grad():
            test_loss = 0
            for step, (inputs, targets, index) in enumerate(test_loader):
                x, y = inputs.to(device), targets.to(device)
                z = net(x)
                loss = criterion(z, y)
                test_loss += loss.item()

            print('test epoch %d loss %f' % (epoch, test_loss / len(test_loader)))

        hes = []
        if not flags.sgld:
            hessian_comp = hessian(net, criterion, dataloader=[(regress_test.x, regress_test.y)], cuda=True)
            hes.append(hessian_comp.trace())

        test_record.append({
            'L': (train_loss / len(train_loader), test_loss / len(test_loader)),
            'H': np.mean(hes),
        })

    with open('record/' + ('SGLD' if flags.sgld else 'SGD') + '_regress_%d.pkl' % run, 'wb') as fout:
        pickle.dump({
            'train': train_record,
            'test': test_record,
            'R': (maxR - minR) / 2,
            'S': (train_dataset.x, train_dataset.y),
            'W0': W0,
        }, fout)


if __name__ == "__main__":
    for restart in range(flags.restarts):
        train(restart)

