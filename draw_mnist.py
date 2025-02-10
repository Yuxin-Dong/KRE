import argparse
import concurrent.futures
import numpy as np
import pickle
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics

parser = argparse.ArgumentParser(description='MLP-MNIST')
parser.add_argument('--mlp_dim', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--restarts', type=int, default=100)
parser.add_argument('--subset', type=int, default=5000)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=50)
parser.add_argument('--sgld', action='store_true', default=False)
parser.add_argument('--sigma', type=float, default=0.00001)
flags = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((14, 14), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Normalize((0.1307,), (0.3081,)),
])
mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)

record = []
executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)


class IndexedDataset(Dataset):
    def __init__(self, dataset, index):
        self.dataset = dataset
        self.subset = index

    def __getitem__(self, index):
        return self.dataset[self.subset[index]]

    def __len__(self):
        return len(self.subset)


def init(mnt, mxt):
    record.clear()
    for run in range(flags.restarts):
        with open('record/%s_mnist_%d.pkl' % ('SGLD' if flags.sgld else 'SGD', run), 'rb') as fin:
            r = pickle.load(fin)
            for t in range(0, mnt):
                del r['train'][t]['W']
            for t in range(mxt, len(r['train'])):
                del r['train'][t]['W']
            record.append(r)
            print('loading ', run)


def kernel(data, sigma2=0.0):
    Kd = metrics.pairwise.euclidean_distances(data, squared=True)
    if sigma2 < 1e-10:
        sigma2 = np.mean(np.sort(Kd[:, :15], 1))
    # print(np.mean(np.sort(Kd[:, :15], 1)))
    if sigma2 < 1e-10:
        return 1, np.ones((data.shape[0], data.shape[0]))
    return np.sqrt(np.pi * sigma2), np.exp(-Kd / (2 * sigma2))


def entropy(C, K):
    e = np.linalg.eigvalsh(K)
    e = e[e > 1e-10]
    e /= np.sum(e)
    return -np.sum(e * np.log(e)) / C


def mutual_info(C1, K1, C2, K2):
    return entropy(C1, K1) + entropy(C2, K2) - entropy(C1 * C2, K1 * K2)


def kernel_index(index, sigma):
    X, Y = np.zeros((index.shape[0], index.shape[1] * 14 * 14)), np.zeros((index.shape[0], index.shape[1] * 10))
    def load_data(i):
        X[i, :] = np.reshape(mnist_train.data[index[i, :]][:, ::2, ::2], -1)
        Y[i, :] = np.reshape(np.array(np.eye(10)[mnist_train.targets[index[i, :]]]), -1)
    futures = [executor.submit(load_data, i) for i in range(index.shape[0])]
    concurrent.futures.wait(futures)
    (CX, KX), (CY, KY) = kernel(X, 13000 * sigma), kernel(Y, 85 * sigma)
    return CX * CY, KX * KY


def calc_KS():
    index = np.zeros((flags.restarts, flags.subset), np.int32)
    for run in range(flags.restarts):
        index[run, :] = record[run]['S']
    return kernel_index(index, 100)


def calc_KB(t):
    index = np.zeros((flags.restarts, flags.batchsize), np.int32)
    for run in range(flags.restarts):
        index[run, :] = record[run]['S'][record[run]['train'][t]['B']]
    return kernel_index(index, 1)


def calc_KW(t):
    A1, A2 = np.zeros((flags.restarts, 14 * 14 * flags.mlp_dim)), np.zeros((flags.restarts, flags.mlp_dim * 10))
    B1, B2 = np.zeros((flags.restarts, flags.mlp_dim)), np.zeros((flags.restarts, 10))
    for run, r in enumerate(record):
        W = r['train'][t]['W'] if t >= 0 else r['W0']
        A1[run, :] = np.reshape(W[0], -1)
        B1[run, :] = W[1]
        A2[run, :] = np.reshape(W[2], -1)
        B2[run, :] = W[3]
    (CA1, KA1), (CA2, KA2), (CB1, KB1), (CB2, KB2) = kernel(A1, 200), kernel(A2, 40), kernel(B1, 0.02), kernel(B2, 0.003)
    return CA1 * CA2 * CB1 * CB2, KA1 * KA2 * KB1 * KB2


def main():
    steps = flags.epochs * flags.subset // flags.batchsize
    d = 14 * 14 * flags.mlp_dim + flags.mlp_dim + flags.mlp_dim * 10 + 10
    V = np.zeros((flags.restarts, steps))
    C = np.zeros(steps)
    H = np.zeros(flags.epochs)
    R = np.zeros(flags.restarts)
    L = np.zeros((flags.restarts, flags.epochs, 2))
    
    init(0, 2000)
    with open('record/%s_mnist_cov_0.pkl' % ('SGLD' if flags.sgld else 'SGD'), 'rb') as fin:
        cov_record = pickle.load(fin)
    if not flags.sgld:
        with open('record/SGD_mnist_hes_0.pkl', 'rb') as fin:
            hes_record = pickle.load(fin)

    for t in range(steps):
        C[t] = cov_record['train'][t]['C']
    if not flags.sgld:
        for t in range(flags.epochs):
            H[t] = hes_record['test'][t]['H']
    for run, r in enumerate(record):
        R[run] = r['R']
        for t in range(steps):
            V[run, t] = r['train'][t]['V']
        for t in range(flags.epochs):
            L[run, t, :] = r['test'][t]['L']

    V = np.mean(V, 0)
    R = np.mean(R) * flags.batchsize
    L = np.mean(L, 0)

    ftrain = open('mnist_%s_train.csv' % ('sgld' if flags.sgld else 'sgd'), 'w')
    ftest = open('mnist_%s_test.csv' % ('sgld' if flags.sgld else 'sgd'), 'w')

    S = np.zeros(4)
    CS, KS = calc_KS()
    CWl, KWl = calc_KW(-1)

    for i in range(5):
        if i > 0:
            init(i * 2000, i * 2000 + 2000)

        for j in range(2000):
            t = i * 2000 + j
            CB, KB = calc_KB(t)
            CWc, KWc = calc_KW(t)
            IWS = mutual_info(CWc, KWc, CS, KS)
            IWB_W = mutual_info(CWc * CWl, KWc * KWl, CB, KB)
            UB = d / 2 * np.log(1 + V[t] / d * flags.lr ** 2 / flags.sigma)
            S[0] = IWS
            S += np.array([0, IWB_W, UB, C[t]])
            G = np.sqrt(2 * R ** 2 * S / flags.subset)
            print(t, V[t], R, IWS, IWB_W, UB, C[t], G[0], G[1], G[2], G[3])
            ftrain.write(','.join(map(str, [t, V[t], R, IWS, IWB_W, UB, C[t], G[0], G[1], G[2], G[3]])) + '\n')
            ftrain.flush()
            CWl, KWl = CWc, KWc

            if (t + 1) % (flags.subset // flags.batchsize) == 0:
                epoch = (t + 1) // (flags.subset // flags.batchsize) - 1
                h_tr = H[epoch] * flags.sigma * (t + 1) / 2
                print(epoch, L[epoch, 0], L[epoch, 1], h_tr, G[0], G[1], G[2], G[3])
                ftest.write(','.join(map(str, [epoch, L[epoch, 0], L[epoch, 1], h_tr, G[0], G[1], G[2], G[3]])) + '\n')
                ftest.flush()


if __name__ == "__main__":
    main()
