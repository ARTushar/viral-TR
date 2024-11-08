import os
import subprocess
from torch import Tensor
import torch
import torch.nn.functional as F
import pandas as pd
import logomaker as lm
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar

li = 'ACGT'

DEBUG = False

def debug_print(debug_code, args):
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


def draw_logo(directory: str, ic: Tensor, suf: str) -> None:
    debug_print('IC: ', ic)
    npa = ic.detach().cpu().numpy().T
    df = pd.DataFrame(npa, columns=['A', 'C', 'G', 'T'])
    logo = lm.Logo(df)
    logo.ax.set_ylim((0, 2))
    plt.savefig(os.path.join(directory, 'logo_' + suf + '.png'), dpi=50)
    plt.close()


def make_motif(directory: str, probs: Tensor, distribution: list, ic_type: int = 0) -> None:
    bar = IncrementalBar('Building Motifs', max=len(probs))

    meme_file = os.path.join(directory, 'motif' + '.meme')
    with open(meme_file, 'w') as f:
        f.write('MEME version 5\n\nALPHABET= ACGT\n\n')
        f.write('strands: + -\n\n')
        f.write('Background letter frequencies\n')
        f.write('A {:.3f} C {:.3f} G {:.3f} T {:.3f}\n'.format(*distribution))
        for i, prob in enumerate(probs):
            f.write(f'\nMOTIF {i+1}\n')
            f.write(f'letter-probability matrix: alength= {prob.shape[0]} w= {prob.shape[1]}\n')
            for line in prob.T:
                f.write(' {:.6f}  {:.6f}  {:.6f}  {:.6f}\n'.format(*list(map(float, line))))

            # subprocess.run(f'meme2images {meme_file} {directory} -png'.split())

            # if ic_type:
            #     ic = calculate_shannon_ic(prob, distribution)
            # else:
            #     ic = calculate_information_content(prob)

            draw_logo(directory, calculate_information_content(prob), str(i+1))
            # draw_logo(directory, calculate_shannon_ic(prob, distribution), 'bg' + str(i+1))

            bar.next()
        bar.finish()


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
