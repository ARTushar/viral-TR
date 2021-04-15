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
        padding_mode: str = 'zeros'  # TODO: refine this type
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

    def forward(self, input: Tensor) -> Tensor:
        print("self.run value is", self.run_value)
        if self.run_value > 2:
            x_tf = self.weight
            x_tf = torch.transpose(x_tf, dim0=0, dim1=2) # swap 0 - 2
            x_tf = torch.transpose(x_tf, dim0=1, dim1=2) # swap 1 - 2

            alpha = 1000
            beta = 1 / alpha
            bkg = torch.Tensor([0.25, 0.25, 0.25, 0.25])

            def calculate(x):
                alpha_x = alpha * x_tf
                ax_reduced = tf.math.reduce_max(alpha_x, axis=1)
                axr_expanded = tf.expand_dims(ax_reduced, axis=1)
                ax_sub_axre = tf.subtract(alpha_x, axr_expanded)
                softmaxed = tf.math.reduce_sum(tf.math.exp(ax_sub_axre), axis=1)
                sm_log_expanded = tf.expand_dims(tf.math.log(softmaxed), axis=1)
                axsaxre_sub_smle = tf.subtract(ax_sub_axre, sm_log_expanded)

                bkg_streched = tf.tile(bkg_tf, [ tf.shape(x)[0] ])
                bkg_stacked = tf.reshape(bkg_streched, [ tf.shape(x)[0], tf.shape(bkg_tf)[0] ])
                bkgs_log = tf.math.log(bkg_stacked)

                return tf.math.scalar_mul(beta, tf.subtract(axsaxre_sub_smle, bkgs_log))

            return self._conv_forward(input, self.weight, self.bias)

