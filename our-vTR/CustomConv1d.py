import torch
from torch import nn
from torch import Tensor


class CustomConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
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
        self.run_value = 1
        self.distribution = torch.tensor([[0.25, 0.25, 0.25, 0.25]]).T
        self.distr_log = torch.log(self.distribution.repeat(1, kernel_size))

    def forward(self, input: Tensor) -> Tensor:
        print("self.run value is", self.run_value)

        use_weight = self.weight
        if self.run_value > 2:
            use_weight = torch.stack([self.calculate(w) for w in self.weight])

        self.run_value += 1
        return self._conv_forward(input, use_weight, self.bias)

    def calculate(self, x):
        alpha = 1000
        alpha_x = alpha * x
        ax_max, _ = torch.max(alpha_x, dim=0, keepdim=True)
        ax_sub_axmx = alpha_x - ax_max
        exp_sum = torch.sum(torch.exp(ax_sub_axmx), dim=0, keepdim=True)
        es_log = torch.log(exp_sum)
        return (ax_sub_axmx - es_log - self.distr_log) / alpha
