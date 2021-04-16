from typing import Tuple

import torch
from torch import Tensor, nn


class CustomConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        alpha: float,
        beta: float,
        distribution: Tuple[float, float, float, float],
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


    def forward(self, input: Tensor) -> Tensor:
        use_weight = self.weight
        if self.run_value > 2:
            U = 'cpu'
            wu = self.weight.type_as(self.distr_log) if U == 'cpu' else self.weight
            dl = self.distr_log.type_as(self.weight) if U == 'gpu' else self.distr_log
            use_weight = torch.stack([self.calculate(w, dl) for w in wu])

        self.run_value += 1
        return self._conv_forward(input, use_weight.type_as(input), self.bias)

    def calculate(self, x: Tensor, distr_log: Tensor) -> Tensor:
        alpha_x = self.alpha * x
        ax_max, _ = torch.max(alpha_x, dim=0, keepdim=True)
        ax_sub_axmx = alpha_x - ax_max
        exp_sum = torch.sum(torch.exp(ax_sub_axmx), dim=0, keepdim=True)
        es_log = torch.log(exp_sum)
        return self.beta * (ax_sub_axmx - es_log - distr_log)



