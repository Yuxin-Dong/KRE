import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def read_csv(filename):
    result = []
    with open(filename) as fin:
        for row in fin.readlines():
            result.append(row.split(','))
    return result


def plot_tight():
    fig, axs = plt.subplots(1, 2, figsize=(5, 1.8))

    b = np.zeros((2, 4, 100))
    for i in range(2):
        data = read_csv('%s_sgld_test.csv' % ('mnist' if i == 0 else 'cifar'))
        for j in range(100):
            b[i, 0, j] = float(data[j][2]) - float(data[j][1])
            b[i, 1, j] = float(data[j][6])
            b[i, 2, j] = float(data[j][5])
            b[i, 3, j] = float(data[j][7])

    for i in range(2):
        axs[i].set_yscale('log')
        axs[i].plot(np.arange(100), b[i, 0, :], label='Generalization Gap')
        axs[i].plot(np.arange(100), b[i, 1, :], label='Wang\'s Bound')
        axs[i].plot(np.arange(100), b[i, 2, :], label='Intermediate Bound')
        axs[i].plot(np.arange(100), b[i, 3, :], label='Ours Bound')
        axs[i].set_xlabel(r'epoch')
        axs[i].set_ylim(bottom=2e-2)
        axs[i].set_xlim(left=0, right=100)
        axs[i].grid(linewidth=0.5, axis='both', alpha=0.3)

    axs[0].set_title('MNIST')
    axs[1].set_title('CIFAR10')
    # legend = axs[1].legend(loc='lower right', bbox_to_anchor=(1, 0.0), framealpha=0.5, fontsize=6)
    legend = fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=2, bbox_to_anchor=(0., 0., 1.0, -0.1))
    for line in legend.get_lines():
        line.set_linewidth(0.8)

    fig.savefig('figures/tight.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_regress():
    fig, axs = plt.subplots(2, 2, figsize=(5, 4))

    d = 121
    b = np.zeros((2, 5, 50))
    v = np.zeros((2, 2, 500))
    for i in range(2):
        data = read_csv('regress_%s_test.csv' % ('sgd' if i == 1 else 'sgld'))
        for j in range(50):
            b[i, 0, j] = float(data[j][2]) - float(data[j][1])
            b[i, 1, j] = (float(data[j][6]) / np.sqrt(3) + float(data[j][3]) * 3 if i == 1 else float(data[j][6]))
            b[i, 2, j] = float(data[j][4])
            b[i, 3, j] = float(data[j][5])
            b[i, 4, j] = (float(data[j][7]) / np.sqrt(2) + float(data[j][3]) * 2 if i == 1 else float(data[j][7]))
        data = read_csv('regress_%s_train.csv' % ('sgd' if i == 1 else 'sgld'))
        for j in range(500):
            v[i, 0, j] = float(data[j][1])
            v[i, 1, j] = float(data[j][6])
    v[:, 0, :] = d / 2 * np.log(1 + v[:, 0, :] / d)

    for i in range(2):
        axs[i, 0].set_yscale('log')
        axs[i, 0].plot(np.arange(50), b[i, 0, :], label='True Gap')
        axs[i, 0].plot(np.arange(50), b[i, 1, :], label='Lemma 2.12 / Lemma 2.15')
        axs[i, 0].plot(np.arange(50), b[i, 2, :], label='IWZ')
        axs[i, 0].plot(np.arange(50), b[i, 3, :], label='IWB|W')
        axs[i, 0].plot(np.arange(50), b[i, 4, :], label='Theorem 2.13 / Theorem 2.16')
        axs[i, 0].set_xlim(left=0, right=50)
        axs[i, 0].grid(linewidth=0.5, axis='both', alpha=0.3)

        axs[i, 1].set_yscale('log')
        axs[i, 1].plot(np.arange(500), v[i, 0, :], label=r'$\theta_v(V_t)$')
        axs[i, 1].plot(np.arange(500), v[i, 1, :], label=r'$\theta_c(\mathbb{V}_t)$')
        axs[i, 1].set_xlim(left=0, right=500)
        axs[i, 1].grid(linewidth=0.5, axis='both', alpha=0.3)

    axs[1, 0].set_xlabel(r'epoch')
    axs[1, 1].set_xlabel(r'step')
    axs[0, 0].set_ylabel(r'SGLD')
    axs[1, 0].set_ylabel(r'SGD')
    # legend = axs[0, 0].legend(loc='lower right', bbox_to_anchor=(1, 0.08), fontsize=6)
    # for line in legend.get_lines():
    #     line.set_linewidth(0.8)
    # legend = axs[1, 0].legend(loc='lower right', bbox_to_anchor=(1, 0.08), fontsize=6)
    legend = fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='upper center', ncol=3, bbox_to_anchor=(0., 0., 1.0, 0.))
    for line in legend.get_lines():
        line.set_linewidth(0.8)
    legend = axs[1, 1].legend(loc='upper right')
    for line in legend.get_lines():
        line.set_linewidth(0.8)

    fig.savefig('figures/regress.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_IWZ():
    fig, axs = plt.subplots(1, 2, figsize=(5, 2))

    v = np.zeros((2, 500))
    for i in range(2):
        data = read_csv('regress_%s_train.csv' % ('sgd' if i == 1 else 'sgld'))
        for j in range(500):
            v[i, j] = float(data[j][3])

    for i in range(2):
        axs[i].plot(np.arange(500), v[i, :], label='IWZ')
        axs[i].set_xlabel(r'step')
        axs[i].set_xlim(left=0, right=500)
        axs[i].grid(linewidth=0.5, axis='both', alpha=0.3)

    axs[0].set_title('SGLD')
    axs[0].annotate('%.3f' % v[0, 50], xy=(50, v[0, 50]), xycoords='data',
                    xytext=(0.2, 0.6), textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=0.7, headwidth=4, headlength=10))
    axs[0].annotate('%.3f' % v[0, 499], xy=(499, v[0, 499]), xycoords='data',
                    xytext=(0.7, 0.6), textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=0.7, headwidth=4, headlength=10))
    axs[1].set_title('SGD')
    axs[1].annotate('%.3f' % v[1, 102], xy=(102, v[1, 102]), xycoords='data',
                    xytext=(0.3, 0.6), textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=0.7, headwidth=4, headlength=10))
    axs[1].annotate('%.3f' % v[1, 499], xy=(499, v[1, 499]), xycoords='data',
                    xytext=(0.7, 0.6), textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=0.7, headwidth=4, headlength=10))

    fig.savefig('figures/IWZ.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_realdata():
    fig, axs = plt.subplots(2, 4, figsize=(10, 4))

    d = np.array([26506, 47642])
    b = np.zeros((2, 2, 4, 100))
    v = np.zeros((2, 2, 2, 10000))
    for i in range(2):
        for k in range(2):
            data = read_csv('%s_%s_test.csv' % ('mnist' if k == 0 else 'cifar', 'sgd' if i == 1 else 'sgld'))
            for j in range(100):
                b[i, k, 0, j] = float(data[j][2]) - float(data[j][1])
                b[i, k, 1, j] = (float(data[j][6]) / np.sqrt(2 if k == 0 else 0.5) + float(data[j][3]) * (2 if k == 0 else 0.5) if i == 1 else float(data[j][6]))
                b[i, k, 2, j] = float(data[j][5])
                b[i, k, 3, j] = (float(data[j][7]) / np.sqrt(2 if k == 0 else 0.2) + float(data[j][3]) * (2 if k == 0 else 0.2) if i == 1 else float(data[j][7]))
            data = read_csv('%s_%s_train.csv' % ('mnist' if k == 0 else 'cifar', 'sgd' if i == 1 else 'sgld'))
            for j in range(10000):
                v[i, k, 0, j] = float(data[j][1])
                v[i, k, 1, j] = float(data[j][6])
            v[i, k, 1, :] = gaussian_filter(v[i, k, 1, :], sigma=3)
    v[:, :, 0, :] = d[None, :, None] / 2 * np.log(1 + v[:, :, 0, :] / d[None, :, None] * 10)

    for i in range(2):
        for k in range(2):
            axs[i, k * 2].set_yscale('log')
            axs[i, k * 2].plot(np.arange(100), b[i, k, 0, :], label='True Gap')
            axs[i, k * 2].plot(np.arange(100), b[i, k, 1, :], label='Lemma 2.12 / Lemma 2.15')
            axs[i, k * 2].plot(np.arange(100), b[i, k, 2, :], label='IWB|W')
            axs[i, k * 2].plot(np.arange(100), b[i, k, 3, :], label='Theorem 2.13 / Theorem 2.16')
            # legend = axs[i, k * 2].legend(loc='lower right', bbox_to_anchor=(1, 0.0), fontsize=6)
            # for line in legend.get_lines():
            #     line.set_linewidth(0.8)
            axs[i, k * 2].set_ylim(bottom=2e-2)
            axs[i, k * 2].set_xlim(left=0, right=100)
            axs[i, k * 2].grid(linewidth=0.5, axis='both', alpha=0.3)

            axs[i, k * 2 + 1].set_yscale('log')
            axs[i, k * 2 + 1].plot(np.arange(10000), v[i, k, 0, :], label=r'$\theta_v(V_t)$')
            axs[i, k * 2 + 1].plot(np.arange(10000), v[i, k, 1, :], label=r'$\theta_c(\mathbb{V}_t)$')
            # axs[i, k * 2 + 1].set_ylim(bottom=0)
            # axs[i, k * 2 + 1].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
            axs[i, k * 2 + 1].get_xaxis().get_offset_text().set_position((0.5, 0))
            axs[i, k * 2 + 1].set_xlim(left=0, right=10000)
            axs[i, k * 2 + 1].grid(linewidth=0.5, axis='both', alpha=0.3)
            # axs[i, k * 2 + 1].xaxis.set_major_formatter(lambda x, pos: r'$%d \times 10^3$' % (x / 1000))

        axs[1, i * 2].set_xlabel(r'epoch')
        axs[1, i * 2 + 1].set_xlabel(r'step')
        legend = axs[0, i * 2 + 1].legend(loc='lower right', bbox_to_anchor=(1, 0.4))
        for line in legend.get_lines():
            line.set_linewidth(0.8)

    legend = fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(0., 0., 1.0, 0.))
    for line in legend.get_lines():
        line.set_linewidth(0.8)

    axs[0, 0].set_ylabel(r'SGLD')
    axs[1, 0].set_ylabel(r'SGD')
    axs[0, 0].set_title(r'MNIST')
    axs[0, 2].set_title(r'CIFAR10')

    fig.savefig('figures/realdata.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_label_mnist():
    fig, axs = plt.subplots(2, 2, figsize=(5, 4))

    epochs = 1000
    steps = epochs * 10
    d = 14 * 14 * 128 + 128 + 128 * 10 + 10
    V = np.zeros((4, steps))
    C = np.zeros((4, steps))
    L = np.zeros((4, 2, epochs))
    A = np.zeros((4, 2, epochs))
    R, H = np.zeros(4), np.zeros(4)
    for i in range(4):
        r = (0, 2, 4, 6)[i]
        data = pickle.load(open('SGD_mnist_%d_128.pkl' % r, 'rb'))
        # r = (8, 16, 32, 64)[i]
        # data = pickle.load(open('SGD_mnist_0_%d.pkl' % r, 'rb'))
        V[i, :] = np.array([t['V'] for t in data['train']])
        C[i, :] = np.array([t['C'] for t in data['train']])
        for j in range(2):
            L[i, j, :] = np.array([t['L'][j] for t in data['test']])
            A[i, j, :] = np.array([t['acc'][j] for t in data['test']])
        R[i] = data['R'] * 50
        H[i] = data['H']
    V = d / 2 * np.log(1 + V / d * 0.02 ** 2 / 5e-6)
    HB = H[:, None] * np.arange(10000)[None, :] * 10 / 2
    VB = np.sqrt(2 * R[:, None] ** 2 * np.cumsum(V * 10, axis=1) / 5000) + HB * 5e-6
    CB = np.sqrt(2 * R[:, None] ** 2 * np.cumsum(C * 10 * 10, axis=1) / 5000) + HB * 1e-6

    for i in range(4):
        axs[0, 0].plot(np.arange(epochs), A[i, 0, :], label=r'$\rho$=%.1f' % (i * 0.2))
        axs[0, 1].plot(np.arange(epochs), L[i, 1, :] - L[i, 0, :])
        axs[1, 0].plot(np.arange(epochs), VB[i, ::10])
        axs[1, 1].plot(np.arange(epochs), CB[i, ::10])

    axs[0, 0].set_ylabel(r'Training Accuracy')
    axs[0, 1].set_ylabel(r'Generalization Error')
    axs[1, 0].set_ylabel('Wang\'s Bound')
    axs[1, 1].set_ylabel(r'Ours Bound', labelpad=1)
    axs[0, 0].set_ylim(bottom=0.3)
    axs[0, 0].set_yticks([0.4, 0.6, 0.8, 1.0])
    axs[0, 0].set_xlim(left=0, right=1000)
    axs[0, 0].grid(linewidth=0.5, axis='both', alpha=0.3)
    axs[0, 1].set_ylim(bottom=0)
    axs[0, 1].set_xlim(left=0, right=1000)
    axs[0, 1].grid(linewidth=0.5, axis='both', alpha=0.3)
    axs[1, 0].set_ylim(bottom=0)
    axs[1, 0].set_xlim(left=0, right=1000)
    axs[1, 0].grid(linewidth=0.5, axis='both', alpha=0.3)
    axs[1, 1].set_ylim(bottom=0)
    axs[1, 1].set_xlim(left=0, right=1000)
    axs[1, 1].grid(linewidth=0.5, axis='both', alpha=0.3)
    legend = axs[0, 0].legend(loc='lower right', bbox_to_anchor=(1, 0.0))
    for line in legend.get_lines():
        line.set_linewidth(0.8)
    axs[1, 0].set_xlabel(r'epoch')
    axs[1, 1].set_xlabel(r'epoch')

    fig.subplots_adjust(wspace=0.25)
    fig.savefig('figures/label_mnist.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_width_mnist():
    fig, axs = plt.subplots(2, 2, figsize=(5, 4))

    epochs = 1000
    steps = epochs * 10
    dim = np.array([16, 32, 64, 128])
    d = 14 * 14 * dim + dim + dim * 10 + 10
    V = np.zeros((4, steps))
    C = np.zeros((4, steps))
    L = np.zeros((4, 2, epochs))
    A = np.zeros((4, 2, epochs))
    R, H = np.zeros(4), np.zeros(4)
    for i in range(4):
        r = dim[i]
        data = pickle.load(open('SGD_mnist_0_%d.pkl' % r, 'rb'))
        V[i, :] = np.array([t['V'] for t in data['train']])
        C[i, :] = np.array([t['C'] for t in data['train']])
        for j in range(2):
            L[i, j, :] = np.array([t['L'][j] for t in data['test']])
            A[i, j, :] = np.array([t['acc'][j] for t in data['test']])
        R[i] = data['R'] * 50
        H[i] = data['H']
    R[:] = np.mean(R)
    V = d[:, None] / 2 * np.log(1 + V / d[:, None] * 0.02 ** 2 / 2e-5)
    HB = H[:, None] * np.arange(10000)[None, :] * 10 / 2
    VB = np.sqrt(2 * R[:, None] ** 2 * np.cumsum(V * 10, axis=1) / 5000) + HB * 2e-5
    CB = np.sqrt(2 * R[:, None] ** 2 * np.cumsum(C * 10 * 1e-5 / 3e-6, axis=1) / 5000) + HB * 3e-6

    for i in range(4):
        axs[0, 0].plot(np.arange(epochs), A[i, 0, :], label=r'dim=%d' % dim[i])
        axs[0, 1].plot(np.arange(epochs), L[i, 1, :] - L[i, 0, :])
        axs[1, 0].plot(np.arange(epochs), VB[i, ::10])
        axs[1, 1].plot(np.arange(epochs), CB[i, ::10])

    axs[0, 0].set_ylabel(r'Training Accuracy')
    axs[0, 1].set_ylabel(r'Generalization Error', labelpad=-1)
    axs[1, 0].set_ylabel('Wang\'s Bound')
    axs[1, 1].set_ylabel(r'Ours Bound', labelpad=2)
    axs[0, 0].set_ylim(bottom=0.88, top=1.01)
    axs[0, 0].set_yticks([0.88, 0.92, 0.96, 1.00])
    axs[0, 0].set_xlim(left=0, right=1000)
    axs[0, 0].grid(linewidth=0.5, axis='both', alpha=0.3)
    axs[0, 1].set_ylim(bottom=0)
    axs[0, 1].set_xlim(left=0, right=1000)
    axs[0, 1].grid(linewidth=0.5, axis='both', alpha=0.3)
    axs[1, 0].set_ylim(bottom=0)
    axs[1, 0].set_xlim(left=0, right=1000)
    axs[1, 0].grid(linewidth=0.5, axis='both', alpha=0.3)
    axs[1, 1].set_ylim(bottom=0)
    axs[1, 1].set_xlim(left=0, right=1000)
    axs[1, 1].grid(linewidth=0.5, axis='both', alpha=0.3)
    legend = axs[0, 0].legend(loc='lower right', bbox_to_anchor=(1, 0.0))
    for line in legend.get_lines():
        line.set_linewidth(0.8)
    axs[1, 0].set_xlabel(r'epoch')
    axs[1, 1].set_xlabel(r'epoch')

    fig.subplots_adjust(wspace=0.25)
    fig.savefig('figures/width_mnist.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_label_cifar():
    fig, axs = plt.subplots(2, 2, figsize=(5, 4))

    epochs = 400
    steps = epochs * 10
    d = 3 * 32 * 9 + 32 + 32 * 32 * 9 + 32 + 32 * 48 * 9 + 48 + 48 * 48 * 9 + 48 + 48 * 48 + 48 + 48 * 10 + 10
    V = np.zeros((4, steps))
    C = np.zeros((4, steps))
    L = np.zeros((4, 2, epochs))
    A = np.zeros((4, 2, epochs))
    R, H = np.zeros(4), np.zeros(4)
    for i in range(4):
        r = (0, 2, 4, 6)[i]
        data = pickle.load(open('SGD_cifar_randlabel_%d.pkl' % r, 'rb'))
        V[i, :] = np.array([t['V'] for t in data['train']])
        C[i, :] = np.array([t['C'] for t in data['train']])
        for j in range(2):
            L[i, j, :] = np.array([t['L'][j] for t in data['test']])
            A[i, j, :] = np.array([t['acc'][j] for t in data['test']])
        R[i] = data['R'] * 50
        H[i] = data['H']
    R[0] = 1.4
    V = d / 2 * np.log(1 + V / d * 0.02 ** 2 / 4e-7)
    HB = H[:, None] * np.arange(4000)[None, :] * 10 / 2
    VB = np.sqrt(2 * R[:, None] ** 2 * np.cumsum(V * 10, axis=1) / 5000) + HB * 4e-7
    CB = np.sqrt(2 * R[:, None] ** 2 * np.cumsum(C * 10 * 1e-6 / 1.5e-7, axis=1) / 5000) + HB * 1.5e-7

    for i in range(4):
        axs[0, 0].plot(np.arange(epochs), A[i, 0, :], label=r'$\rho$=%.1f' % (i * 0.2))
        axs[0, 1].plot(np.arange(epochs), L[i, 1, :] - L[i, 0, :])
        axs[1, 0].plot(np.arange(epochs), VB[i, ::10])
        axs[1, 1].plot(np.arange(epochs), CB[i, ::10])

    axs[0, 0].set_ylabel(r'Training Accuracy')
    axs[0, 1].set_ylabel(r'Generalization Error')
    axs[1, 0].set_ylabel('Wang\'s Bound')
    axs[1, 1].set_ylabel(r'Ours Bound', labelpad=1)
    axs[0, 0].set_ylim(bottom=0.1)
    axs[0, 0].set_xlim(left=0, right=400)
    axs[0, 0].grid(linewidth=0.5, axis='both', alpha=0.3)
    axs[0, 1].set_ylim(bottom=0)
    axs[0, 1].set_xlim(left=0, right=400)
    axs[0, 1].grid(linewidth=0.5, axis='both', alpha=0.3)
    axs[1, 0].set_ylim(bottom=0)
    axs[1, 0].set_xlim(left=0, right=400)
    axs[1, 0].grid(linewidth=0.5, axis='both', alpha=0.3)
    axs[1, 1].set_ylim(bottom=0)
    axs[1, 1].set_yticks([0, 100, 200, 300])
    axs[1, 1].set_xlim(left=0, right=400)
    axs[1, 1].grid(linewidth=0.5, axis='both', alpha=0.3)
    legend = axs[0, 0].legend(loc='lower right', bbox_to_anchor=(1, 0.0))
    for line in legend.get_lines():
        line.set_linewidth(0.8)
    axs[1, 0].set_xlabel(r'epoch')
    axs[1, 1].set_xlabel(r'epoch')

    fig.subplots_adjust(wspace=0.25)
    fig.savefig('figures/label_cifar.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_ppt_1():
    fig, axs = plt.subplots(1, 2, figsize=(5, 1.8))

    b = np.zeros((2, 4, 100))
    for i in range(2):
        data = read_csv('%s_sgld_test.csv' % ('mnist' if i == 0 else 'cifar'))
        for j in range(100):
            b[i, 0, j] = float(data[j][2]) - float(data[j][1])
            b[i, 1, j] = float(data[j][6])
            b[i, 2, j] = float(data[j][5])
            b[i, 3, j] = float(data[j][7])

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i in range(2):
        axs[i].set_yscale('log')
        axs[i].plot(np.arange(100), b[i, 0, :], label='Generalization Gap', color=colors[0])
        axs[i].plot(np.arange(100), b[i, 1, :], label='Wang\'s Bound', color=colors[1])
        axs[i].plot(np.arange(100), b[i, 2, :], label='Intermediate Bound', color=colors[3])
        axs[i].plot(np.arange(100), b[i, 3, :], label='Ours Bound', color=colors[2])
        axs[i].set_xlabel(r'epoch')
        axs[i].set_ylim(bottom=2e-2)
        axs[i].set_xlim(left=0, right=100)
        axs[i].grid(linewidth=0.5, axis='both', alpha=0.3)

    axs[0].set_title('MNIST')
    axs[1].set_title('CIFAR10')
    # legend = axs[1].legend(loc='lower right', bbox_to_anchor=(1, 0.0), framealpha=0.5)
    # for line in legend.get_lines():
    #     line.set_linewidth(0.8)

    fig.savefig('figures/ppt_1.png', format='png', dpi=600, bbox_inches='tight')
    fig.show()


def plot_ppt_2():
    fig, axs = plt.subplots(2, 4, figsize=(10, 4))

    d = np.array([26506, 47642])
    b = np.zeros((2, 2, 4, 100))
    v = np.zeros((2, 2, 2, 10000))
    for i in range(2):
        for k in range(2):
            data = read_csv('%s_%s_test.csv' % ('mnist' if k == 0 else 'cifar', 'sgd' if i == 1 else 'sgld'))
            for j in range(100):
                b[i, k, 0, j] = float(data[j][2]) - float(data[j][1])
                b[i, k, 1, j] = (float(data[j][6]) / np.sqrt(2 if k == 0 else 0.5) + float(data[j][3]) * (2 if k == 0 else 0.5) if i == 1 else float(data[j][6]))
                b[i, k, 2, j] = float(data[j][5])
                b[i, k, 3, j] = (float(data[j][7]) / np.sqrt(2 if k == 0 else 0.2) + float(data[j][3]) * (2 if k == 0 else 0.2) if i == 1 else float(data[j][7]))
            data = read_csv('%s_%s_train.csv' % ('mnist' if k == 0 else 'cifar', 'sgd' if i == 1 else 'sgld'))
            for j in range(10000):
                v[i, k, 0, j] = float(data[j][1])
                v[i, k, 1, j] = float(data[j][6])
            v[i, k, 1, :] = gaussian_filter(v[i, k, 1, :], sigma=3)
    v[:, :, 0, :] = d[None, :, None] / 2 * np.log(1 + v[:, :, 0, :] / d[None, :, None] * 10)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i in range(2):
        for k in range(2):
            axs[i, k * 2].set_yscale('log')
            axs[i, k * 2].plot(np.arange(100), b[i, k, 0, :], label='Generalization Error', color=colors[0])
            axs[i, k * 2].plot(np.arange(100), b[i, k, 1, :], label='Previous Bound', color=colors[1])
            axs[i, k * 2].plot(np.arange(100), b[i, k, 2, :], label='Computable Bound', color=colors[3])
            axs[i, k * 2].plot(np.arange(100), b[i, k, 3, :], label='Ours Bound', color=colors[2])
            # legend = axs[i, k * 2].legend(loc='lower right', bbox_to_anchor=(1, 0.0))
            # for line in legend.get_lines():
            #     line.set_linewidth(0.8)
            axs[i, k * 2].set_ylim(bottom=2e-2)
            axs[i, k * 2].set_xlim(left=0, right=100)
            axs[i, k * 2].grid(linewidth=0.5, axis='both', alpha=0.3)

            axs[i, k * 2 + 1].set_yscale('log')
            axs[i, k * 2 + 1].plot(np.arange(10000) / 100, v[i, k, 0, :], label=r'$V_t$')
            axs[i, k * 2 + 1].plot(np.arange(10000) / 100, v[i, k, 1, :], label=r'$\mathbb{V}_t$')
            # axs[i, k * 2 + 1].set_ylim(bottom=0)
            # axs[i, k * 2 + 1].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
            axs[i, k * 2 + 1].get_xaxis().get_offset_text().set_position((0.5, 0))
            axs[i, k * 2 + 1].set_xlim(left=0, right=100)
            axs[i, k * 2 + 1].grid(linewidth=0.5, axis='both', alpha=0.3)
            # axs[i, k * 2 + 1].xaxis.set_major_formatter(lambda x, pos: r'$%d \times 10^3$' % (x / 1000))

        axs[1, i * 2].set_xlabel(r'epoch')
        axs[1, i * 2 + 1].set_xlabel(r'epoch')

    legend = axs[0, 1].legend(loc='lower right', bbox_to_anchor=(1, 0.4))
    for line in legend.get_lines():
        line.set_linewidth(0.8)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', ncol=4, markerscale=1.5, bbox_to_anchor=(0.5, 0))
    for line in legend.get_lines():
        line.set_linewidth(1)

    axs[0, 0].set_ylabel(r'SGLD')
    axs[1, 0].set_ylabel(r'SGD')
    axs[0, 0].set_title(r'MNIST')
    axs[0, 2].set_title(r'CIFAR10')

    fig.savefig('figures/ppt_2.png', format='png', dpi=600, bbox_inches='tight')
    fig.show()


def plot_ppt_3():
    fig, axs = plt.subplots(2, 4, figsize=(10, 4))

    epochs = 1000
    steps = epochs * 10
    d = 14 * 14 * 128 + 128 + 128 * 10 + 10
    V = np.zeros((4, steps))
    C = np.zeros((4, steps))
    L = np.zeros((4, 2, epochs))
    A = np.zeros((4, 2, epochs))
    R, H = np.zeros(4), np.zeros(4)
    for i in range(4):
        r = (0, 2, 4, 6)[i]
        data = pickle.load(open('SGD_mnist_%d_128.pkl' % r, 'rb'))
        # r = (8, 16, 32, 64)[i]
        # data = pickle.load(open('SGD_mnist_0_%d.pkl' % r, 'rb'))
        V[i, :] = np.array([t['V'] for t in data['train']])
        C[i, :] = np.array([t['C'] for t in data['train']])
        for j in range(2):
            L[i, j, :] = np.array([t['L'][j] for t in data['test']])
            A[i, j, :] = np.array([t['acc'][j] for t in data['test']])
        R[i] = data['R'] * 50
        H[i] = data['H']
    V = d / 2 * np.log(1 + V / d * 0.02 ** 2 / 5e-6)
    HB = H[:, None] * np.arange(10000)[None, :] * 10 / 2
    VB = np.sqrt(2 * R[:, None] ** 2 * np.cumsum(V * 10, axis=1) / 5000) + HB * 5e-6
    CB = np.sqrt(2 * R[:, None] ** 2 * np.cumsum(C * 10 * 10, axis=1) / 5000) + HB * 1e-6

    for i in range(4):
        axs[0, 0].plot(np.arange(epochs), A[i, 0, :], label=r'$\rho$=%.1f' % (i * 0.2))
        axs[0, 1].plot(np.arange(epochs), L[i, 1, :] - L[i, 0, :])
        axs[1, 0].plot(np.arange(epochs), VB[i, ::10])
        axs[1, 1].plot(np.arange(epochs), CB[i, ::10])

    axs[0, 0].set_ylabel(r'Training Accuracy')
    axs[0, 1].set_ylabel(r'Generalization Error')
    axs[1, 0].set_ylabel(r'Previous Bound')
    axs[1, 1].set_ylabel(r'Ours Bound')
    axs[0, 0].set_ylim(bottom=0.3)
    axs[0, 0].set_yticks([0.4, 0.6, 0.8, 1.0])
    axs[0, 0].set_xlim(left=0, right=1000)
    axs[0, 0].grid(linewidth=0.5, axis='both', alpha=0.3)
    axs[0, 1].set_ylim(bottom=0)
    axs[0, 1].set_xlim(left=0, right=1000)
    axs[0, 1].grid(linewidth=0.5, axis='both', alpha=0.3)
    axs[1, 0].set_ylim(bottom=0)
    axs[1, 0].set_xlim(left=0, right=1000)
    axs[1, 0].grid(linewidth=0.5, axis='both', alpha=0.3)
    axs[1, 1].set_ylim(bottom=0)
    axs[1, 1].set_xlim(left=0, right=1000)
    axs[1, 1].grid(linewidth=0.5, axis='both', alpha=0.3)
    legend = axs[0, 0].legend(loc='lower right', bbox_to_anchor=(1, 0.0))
    for line in legend.get_lines():
        line.set_linewidth(0.8)
    axs[1, 0].set_xlabel(r'epoch')
    axs[1, 1].set_xlabel(r'epoch')

    epochs = 400
    steps = epochs * 10
    d = 3 * 32 * 9 + 32 + 32 * 32 * 9 + 32 + 32 * 48 * 9 + 48 + 48 * 48 * 9 + 48 + 48 * 48 + 48 + 48 * 10 + 10
    V = np.zeros((4, steps))
    C = np.zeros((4, steps))
    L = np.zeros((4, 2, epochs))
    A = np.zeros((4, 2, epochs))
    R, H = np.zeros(4), np.zeros(4)
    for i in range(4):
        r = (0, 2, 4, 6)[i]
        data = pickle.load(open('SGD_cifar_randlabel_%d.pkl' % r, 'rb'))
        V[i, :] = np.array([t['V'] for t in data['train']])
        C[i, :] = np.array([t['C'] for t in data['train']])
        for j in range(2):
            L[i, j, :] = np.array([t['L'][j] for t in data['test']])
            A[i, j, :] = np.array([t['acc'][j] for t in data['test']])
        R[i] = data['R'] * 50
        H[i] = data['H']
    R[0] = 1.4
    V = d / 2 * np.log(1 + V / d * 0.02 ** 2 / 4e-7)
    HB = H[:, None] * np.arange(4000)[None, :] * 10 / 2
    VB = np.sqrt(2 * R[:, None] ** 2 * np.cumsum(V * 10, axis=1) / 5000) + HB * 4e-7
    CB = np.sqrt(2 * R[:, None] ** 2 * np.cumsum(C * 10 * 1e-6 / 1.5e-7, axis=1) / 5000) + HB * 1.5e-7

    for i in range(4):
        axs[0, 2].plot(np.arange(epochs), A[i, 0, :], label=r'$\rho$=%.1f' % (i * 0.2))
        axs[0, 3].plot(np.arange(epochs), L[i, 1, :] - L[i, 0, :])
        axs[1, 2].plot(np.arange(epochs), VB[i, ::10])
        axs[1, 3].plot(np.arange(epochs), CB[i, ::10])

    axs[0, 2].set_ylabel(r'Training Accuracy', labelpad=1)
    axs[0, 3].set_ylabel(r'Generalization Error', labelpad=1)
    axs[1, 2].set_ylabel(r'Previous Bound', labelpad=1)
    axs[1, 3].set_ylabel(r'Ours Bound')
    axs[0, 2].set_ylim(bottom=0.1)
    axs[0, 2].set_xlim(left=0, right=400)
    axs[0, 2].grid(linewidth=0.5, axis='both', alpha=0.3)
    axs[0, 3].set_ylim(bottom=0)
    axs[0, 3].set_xlim(left=0, right=400)
    axs[0, 3].grid(linewidth=0.5, axis='both', alpha=0.3)
    axs[1, 2].set_ylim(bottom=0)
    axs[1, 2].set_xlim(left=0, right=400)
    axs[1, 2].grid(linewidth=0.5, axis='both', alpha=0.3)
    axs[1, 3].set_ylim(bottom=0)
    axs[1, 3].set_yticks([0, 100, 200, 300])
    axs[1, 3].set_xlim(left=0, right=400)
    axs[1, 3].grid(linewidth=0.5, axis='both', alpha=0.3)
    legend = axs[0, 2].legend(loc='lower right', bbox_to_anchor=(1, 0.0))
    for line in legend.get_lines():
        line.set_linewidth(0.8)
    axs[1, 2].set_xlabel(r'epoch')
    axs[1, 3].set_xlabel(r'epoch')

    axs[0, 0].set_title(r'MNIST')
    axs[0, 2].set_title(r'CIFAR10')

    axs[1, 0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axs[1, 1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axs[1, 2].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axs[1, 3].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.25)
    fig.savefig('figures/ppt_3.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def main():
    mpl.style.use('bmh')

    mpl.rcParams['lines.linewidth'] = 0.5
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['xtick.color'] = 'gray'
    mpl.rcParams['xtick.labelcolor'] = 'black'
    mpl.rcParams['ytick.color'] = 'gray'
    mpl.rcParams['ytick.labelcolor'] = 'black'
    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['ytick.direction'] = 'out'
    mpl.rcParams['xtick.major.size'] = 2
    mpl.rcParams['ytick.major.size'] = 2
    mpl.rcParams['ytick.major.pad'] = 1
    mpl.rcParams['ytick.minor.size'] = 1
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    plot_tight()
    plot_regress()
    # plot_IWZ()
    plot_realdata()
    # plot_label_mnist()
    # plot_width_mnist()
    # plot_label_cifar()

    # plot_ppt_1()
    # plot_ppt_2()
    # plot_ppt_3()

if __name__ == '__main__':
    main()
