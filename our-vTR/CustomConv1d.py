import torch
from torch import nn
from torch import Tensor
from pytorch_lightning import LightningModule


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
        # self.register_buffer("t_distr", torch.tensor([distr]).T)

    def forward(self, input: Tensor) -> Tensor:
        distr = [0.25, 0.25, 0.25, 0.25]
        t_distr = torch.tensor([distr]).T.type_as(input)
        distr_log = torch.log(t_distr.repeat(1, 12))

        use_weight = self.weight
        if self.run_value > 2:
            use_weight = torch.stack([CustomConv1d.calculate(w, distr_log) for w in self.weight])

        self.run_value += 1
        return self._conv_forward(input, use_weight, self.bias)

    def calculate(x, distr_log):
        alpha = 1000
        alpha_x = alpha * x
        ax_max, _ = torch.max(alpha_x, dim=0, keepdim=True)
        ax_sub_axmx = alpha_x - ax_max
        exp_sum = torch.sum(torch.exp(ax_sub_axmx), dim=0, keepdim=True)
        es_log = torch.log(exp_sum)
        return (ax_sub_axmx - es_log - distr_log) / alpha
