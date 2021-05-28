import os
from torch import Tensor
import torch
import torch.nn.functional as F
import pandas as pd
import logomaker as lm
import matplotlib.pyplot as plt

li = 'ACGT'

def make_motif(dir: str, kernels: Tensor) -> None:
    for i, kernel in enumerate(kernels):
        prob = F.softmax(kernel, dim=0)

        npa = prob.detach().cpu().numpy().T
        df = pd.DataFrame(npa, columns=['A', 'C', 'G', 'T'])
        logo = lm.Logo(df)
        plt.savefig(os.path.join(dir, 'logo' + str(i+1) + '.png'), dpi=50)
        plt.close()


if __name__ == '__main__':
    make_motif('logos/', torch.randn(3, 4, 12))