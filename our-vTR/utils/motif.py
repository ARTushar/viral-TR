import os
from torch import Tensor, distributions
import torch
import torch.nn.functional as F
import pandas as pd
import logomaker as lm
import matplotlib.pyplot as plt

li = 'ACGT'

DEBUG = False

def debug_print(debug_code, *args):
    if DEBUG:
        print(debug_code, *args)

def calculate_information_content(prob: Tensor) -> Tensor:
    total_ic = 2
    uncertainty = torch.sum(
        prob * torch.nan_to_num(torch.log2(prob)), 0, keepdim=True)
    ic = total_ic + uncertainty
    return prob * ic

def calculate_shannon_ic(prob: Tensor, distribution: list) -> Tensor:
    dis = [distribution for _ in range(prob.shape[1])]
    debug_print('dis: ', dis)
    bg = torch.tensor(dis).T
    debug_print('bg_shape: ', bg.shape)
    return prob * torch.nan_to_num(torch.log2(prob / bg))


def make_motif(dir: str, kernels: Tensor, distribution: list, ic_type: int = 1) -> None:
    for i, kernel in enumerate(kernels):
        prob = F.softmax(kernel, dim=0)
        debug_print('prob: ', prob)
        if ic_type:
            ic = calculate_shannon_ic(prob, distribution)
        else:
            ic = calculate_information_content(prob)
        debug_print('IC: ', ic)
        npa = ic.detach().cpu().numpy().T
        df = pd.DataFrame(npa, columns=['A', 'C', 'G', 'T'])
        logo = lm.Logo(df)
        # if DEBUG:
        #     plt.show()
        plt.savefig(os.path.join(dir, 'logo' + str(i+1) + '.png'), dpi=50)
        plt.close()


if __name__ == '__main__':
    DEBUG = True

    if not os.path.isdir('logos'):
        os.mkdir('logos')
    test = torch.tensor([[
        [1, 0.67, 0, .83, .83, .66],
        [0, 0, .33, 0, 0, 0],
        [0, 0, .5, 0, 0, 0],
        [0, .33, .17, .17, .17, .33]
    ]])
    make_motif('logos/', test, [.3, .2, .2, .3])
    # make_motif('logos/', torch.randn(3, 4, 12), [.3, .2, .2, .3])
