from typing import Iterator, Tuple

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter


class CustomConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        alpha: float,
        beta: float,
        distribution: Tuple[float, ...],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode
        )
        self.alpha = alpha
        self.beta = beta
        self.run_value = 1
        t_distr = torch.tensor([distribution]).T
        self.distr_log = torch.log(t_distr.repeat(1, self.kernel_size[0]))

        self.exclude_idx = None


    def forward(self, input: Tensor) -> Tensor:
        use_weight = self.weight
        if self.run_value > 2:
            w_to_cpu = self.weight.type_as(self.distr_log)
            use_weight = torch.stack([self.calculate(w) for w in w_to_cpu])

        if self.exclude_idx is not None:
            use_weight[self.exclude_idx] = 0

        self.run_value += 1
        return self._conv_forward(input, use_weight.type_as(input), self.bias)

    def calculate(self, x: Tensor) -> Tensor:
        alpha_x = self.alpha * x
        ax_max, _ = torch.max(alpha_x, dim=0, keepdim=True)
        ax_sub_axmx = alpha_x - ax_max
        exp_sum = torch.sum(torch.exp(ax_sub_axmx), dim=0, keepdim=True)
        es_log = torch.log(exp_sum)
        return self.beta * (ax_sub_axmx - es_log - self.distr_log)

    def probability(self, x: Tensor) -> Tensor:
        alpha_x = self.alpha * x
        ax_max, _ = torch.max(alpha_x, dim=0, keepdim=True)
        ax_sub_axmx = alpha_x - ax_max
        exps = torch.exp(ax_sub_axmx)
        exp_sum = torch.sum(exps, dim=0, keepdim=True)
        return exps / exp_sum
        # es_log = torch.log(exp_sum)
        # return self.beta * (ax_sub_axmx - es_log - self.distr_log)

    def get_probabilities(self) -> Tensor:
        return torch.stack([self.probability(w) for w in self.weight])

    def nullify(self, idx: int = None) -> None:
        self.exclude_idx = idx