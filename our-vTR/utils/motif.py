from torch import Tensor
import torch

li = 'ACGT'

def make_motif(kernel: Tensor) -> str:
    mxidx = torch.argmax(kernel, dim=0)
    return ''.join(li[idx] for idx in mxidx)


if __name__ == '__main__':
    motif = make_motif(torch.randn(4, 7))
    print(motif)