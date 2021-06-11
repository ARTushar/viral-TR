import os
import subprocess
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


def make_motif(dir: str, probs: Tensor, distribution: list, ic_type: int = 0) -> None:
    for i, prob in enumerate(probs):
        meme_file = os.path.join(dir, 'meme' + str(i+1) + '.txt')
        with open(meme_file, 'w') as f:
            f.write('MEME version 5\n\nALPHABET= ACGT\n\n')
            f.write('strands: + -\n\n')
            f.write('Background letter frequencies\n')
            f.write('A {:.3f} C {:.3f} G {:.3f} T {:.3f}\n\n'.format(*distribution))
            f.write(f'MOTIF {i+1}\n')
            f.write(f'letter-probability matrix: alength= {prob.shape[0]} w= {prob.shape[1]}\n')
            for line in prob.T:
                f.write(' {:.6f}  {:.6f}  {:.6f}  {:.6f}\n'.format(*list(map(float, line))))

        # subprocess.run(f'meme2images {meme_file} {dir} -png'.split())

        if ic_type:
            ic = calculate_shannon_ic(prob, distribution)
        else:
            ic = calculate_information_content(prob)
        debug_print('IC: ', ic)
        npa = ic.detach().cpu().numpy().T
        df = pd.DataFrame(npa, columns=['A', 'C', 'G', 'T'])
        logo = lm.Logo(df)
        if DEBUG:
            plt.show()
        plt.savefig(os.path.join(dir, 'logo_' + str(i+1) + '.png'), dpi=50)
        plt.close()

        print('.', end='')
    print()


if __name__ == '__main__':
    # if not os.path.isdir('logos'):
    #     os.mkdir('logos')
    test = torch.tensor([[
        [.7, .3, .3, .3, .3],
        [.1, .2, .2, .2, .2],
        [.1, .2, .2, .2, .2],
        [.1, .3, .3, .3, .3]
    ]])
    make_motif('logos/', test, [.7, .1, .1, .1])
    # make_motif('logos/', torch.randn(3, 4, 12), [.3, .2, .2, .3])
