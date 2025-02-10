import argparse
import numpy as np
import pickle
from sklearn import metrics

parser = argparse.ArgumentParser(description='LIN-REGRESS')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--restarts', type=int, default=100)
parser.add_argument('--subset', type=int, default=100)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batchsize', type=int, default=10)
parser.add_argument('--sgld', action='store_true', default=False)
parser.add_argument('--sigma', type=float, default=0.001)
flags = parser.parse_args()


record = []


def init():
    for run in range(flags.restarts):
        with open('record/%s_regress_%d.pkl' % ('SGLD' if flags.sgld else 'SGD', run), 'rb') as fin:
            r = pickle.load(fin)
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


def calc_KS():
    X, Y = np.zeros((flags.restarts, flags.subset * 10)), np.zeros((flags.restarts, flags.subset))
    for run, r in enumerate(record):
        X[run, :] = r['S'][0].view(-1)
        Y[run, :] = r['S'][1].view(-1)
    (CX, KX), (CY, KY) = kernel(X, 30000), kernel(Y, 150000)
    return CX * CY, KX * KY


def calc_KB(t):
    X, Y = np.zeros((flags.restarts, flags.batchsize * 10)), np.zeros((flags.restarts, flags.batchsize))
    for run, r in enumerate(record):
        X[run, :] = r['S'][0][r['train'][t]['B'], :].view(-1)
        Y[run, :] = r['S'][1][r['train'][t]['B']].view(-1)
    (CX, KX), (CY, KY) = kernel(X, 3000), kernel(Y, 15000)
    return CX * CY, KX * KY


def calc_KW(t):
    A1, A2 = np.zeros((flags.restarts, 100)), np.zeros((flags.restarts, 10))
    B1, B2 = np.zeros((flags.restarts, 10)), np.zeros((flags.restarts, 1))
    for run, r in enumerate(record):
        W = r['train'][t]['W'] if t >= 0 else r['W0']
        A1[run, :] = np.reshape(W[0], -1)
        B1[run, :] = W[1]
        A2[run, :] = np.reshape(W[2], -1)
        B2[run, :] = W[3]
    (CA1, KA1), (CA2, KA2), (CB1, KB1), (CB2, KB2) = kernel(A1, 28), kernel(A2, 16), kernel(B1, 0.1), kernel(B2, 0.03)
    return CA1 * CA2 * CB1 * CB2, KA1 * KA2 * KB1 * KB2


def main():
    steps = flags.epochs * flags.subset // flags.batchsize
    d = 100 + 10 + 10 + 1
    V = np.zeros((flags.restarts, steps))
    C = np.zeros((flags.restarts, steps))
    R = np.zeros(flags.restarts)
    L = np.zeros((flags.restarts, flags.epochs, 2))
    H = np.zeros((flags.restarts, flags.epochs))

    init()
    for run, r in enumerate(record):
        R[run] = r['R']
        for t in range(steps):
            V[run, t] = r['train'][t]['V']
            C[run, t] = r['train'][t]['C']
        for t in range(flags.epochs):
            L[run, t, :] = r['test'][t]['L']
            H[run, t] = r['test'][t]['H']

    V = np.mean(V, 0)
    C = np.mean(C, 0)
    R = np.mean(R)
    L = np.mean(L, 0)
    H = np.mean(H, 0)

    ftrain = open('regress_%s_train.csv' % ('sgld' if flags.sgld else 'sgd'), 'w')
    ftest = open('regress_%s_test.csv' % ('sgld' if flags.sgld else 'sgd'), 'w')

    S = np.zeros(4)
    CS, KS = calc_KS()
    CWl, KWl = calc_KW(-1)

    for t in range(steps):
        CB, KB = calc_KB(t)
        CWc, KWc = calc_KW(t)
        IWS = mutual_info(CWc, KWc, CS, KS)
        IWB_W = mutual_info(CWc * CWl, KWc * KWl, CB, KB)
        UB = d / 2 * np.log(1 + V[t] / d)
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
