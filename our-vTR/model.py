from typing import Dict, Tuple, List, Union

import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torchmetrics import Accuracy, F1, MetricCollection, Precision, Recall, AUROC
from torchmetrics.functional import auroc

from CustomConv1d import CustomConv1d

# from torchviz import make_dot

PRINT = False

class SimpleModel(pl.LightningModule):

    def __init__(
        self,
        convolution_type: str,
        kernel_size: int,
        kernel_count: int,
        alpha: float,
        beta: float,
        pool_type: str,
        distribution: Tuple[float, ...],
        linear_layer_shapes: List[int],
        l1_lambda: float,
        l2_lambda: float,
        dropout_p: Union[float, None],
        lr: float,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()

        # print(convolution_type, kernel_size, kernel_count, alpha, beta, pool_type,
        #     distribution, linear_layer_shapes, l1_lambda, l2_lambda, dropout_p, lr)

        self.pool_type = pool_type
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.lr = lr

        if convolution_type == 'custom':
            print('using CUSTOM convolution')
            self.conv1d = CustomConv1d(
                kernel_size=kernel_size,
                in_channels=len(distribution),
                out_channels=kernel_count,
                alpha=alpha,
                beta=beta,
                distribution=distribution
            )
        elif convolution_type == 'regular':
            print('using REGULAR convolution')
            self.conv1d = nn.Conv1d(
                kernel_size=kernel_size,
                in_channels=len(distribution),
                out_channels=kernel_count
            )
        else:
            print('********** unknown convolution type **********')
            exit(1)

        use_shapes = [kernel_count, *linear_layer_shapes]
        # linear_layer_shapes.insert(0, kernel_count)

        linears = []
        # loop over consecutive pairs in list
        for in_size, out_size in zip(use_shapes, use_shapes[1:]):
            linears.append(nn.Linear(in_features=in_size, out_features=out_size))
            linears.append(nn.ReLU())
            if dropout_p is not None:
                linears.append(nn.Dropout(p=dropout_p))
        
        # final layer before prediction, no relu after this, only softmax
        linears.append(nn.Linear(in_features=use_shapes[-1], out_features=2))
        self.linears = nn.Sequential(*linears)

        # TODO: Check macro vs micro average

        self.train_metrics = MetricCollection([
            Accuracy(),
            # Precision(num_classes=2),
            # Recall(num_classes=2),
            # AUROC(num_classes=2),
        ], prefix='train')
        self.valid_metrics = MetricCollection([
            Accuracy(),
            # F1(num_classes=2)
        ], prefix='val')
        self.test_metrics = MetricCollection([
            Accuracy(),
            # F1(num_classes=2, average='macro'),
            # AUROC(num_classes=2) # macro
        ], prefix='test')

    def forward(self, x_fw: Tensor, x_rv: Tensor) -> Tensor:
        seq_length = x_fw.shape[2]

        conv_fw = self.conv1d(x_fw)
        conv_rv = self.conv1d(x_rv)
        merged = torch.cat((conv_fw, conv_rv), dim=2)
        pooled = None
        if self.pool_type == 'max':
            pooled = F.max_pool1d(merged, kernel_size=2*(seq_length-self.conv1d.kernel_size[0]+1))
        elif self.pool_type == 'avg':
            pooled = F.avg_pool1d(merged, kernel_size=2*(seq_length-self.conv1d.kernel_size[0]+1))
        flat = pooled.flatten(1, -1)
        line = self.linears(flat)
        probs = F.softmax(line, dim=1)

        if PRINT:
            print(conv_fw.shape, '-> forward conv')
            print(conv_rv.shape, '-> reverse conv')
            print(merged.shape, '-> concat')
            print(pooled.shape, f'-> {self.pool_type} pool')
            print(flat.shape, '-> flatten')
            for layer in self.linears:
                print(layer)
            print(line.shape, '-> line')
            print(probs.shape, '-> softmax')

        return probs

    def training_step(self, train_batch: Tensor, batch_idx: int) -> Tensor:
        # define the training loop

        X_fw, X_rv, y = train_batch
        logits = self.forward(X_fw, X_rv)
        loss = self.cross_entropy_loss(logits, y)

        metrics = self.train_metrics(logits, y.type(torch.int))

        self.log('trainLoss', loss, on_epoch=True, on_step=False, prog_bar=False)
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def validation_step(self, val_batch: Tensor, batch_idx: int) -> None:
        X_fw, X_rv, y = val_batch
        logits = self.forward(X_fw, X_rv)
        loss = self.cross_entropy_loss(logits, y)

        metrics = self.valid_metrics(logits, y.type(torch.int))

        self.log('valLoss', loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, test_batch: Tensor, batch_idx: int) -> None:
        X_fw, X_rv, y = test_batch
        logits = self.forward(X_fw, X_rv)
        loss = self.cross_entropy_loss(logits, y)

        metrics = self.test_metrics(logits, y.type(torch.int))

        self.log('testLoss', loss)
        self.log('testAUROC', auroc(logits, y.type(torch.int), num_classes=2))
        self.log_dict(metrics)

    def configure_optimizers(self) -> Optimizer:
        parameters = self.parameters()
        trainable_parameters = filter(lambda p: p.requires_grad, parameters)
        optimizer = torch.optim.RMSprop(
            trainable_parameters, lr=self.lr, weight_decay=self.l2_lambda)
        return optimizer

    def cross_entropy_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        bce_loss = F.binary_cross_entropy(logits, labels)
        reg_loss = self.l1_lambda * sum(
            sum(x.abs().sum() for x in linear.parameters()) for linear in self.linears
        )
        # reg_loss = self.l1_lambda * sum(x.abs().sum() for x in self.linears[2].parameters())
        return bce_loss + reg_loss

    def get_probabilities(self) -> Tensor:
        if hasattr(self.conv1d, 'get_probabilities'):
            return self.conv1d.get_probabilities()
        return F.softmax(self.conv1d.weight, dim=1)

    def nullify(self, idx: int) -> None:
        self.conv1d.nullify(idx)


def main():
    model = SimpleModel(
        convolution_type='custom',
        kernel_size=14,
        kernel_count=16,
        alpha=1000,
        beta=1/1000,
        distribution=(0.3, 0.2, 0.2, 0.3),
        linear_layer_shapes=[32],
        pool_type='max',
        l1_lambda=1e-3,
        l2_lambda=0,
        dropout_p=None,
        lr=1e-3
    )
    ret = model(torch.ones(64, 4, 156), torch.ones(64, 4, 156))

    # dot = make_dot(ret.mean(), params=dict(model.named_parameters()))
    # dot.format = 'png'
    # dot.render()


if __name__ == '__main__':
    main()
