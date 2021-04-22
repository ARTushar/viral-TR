from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torchmetrics import Accuracy, F1, MetricCollection, Precision, Recall
from torchmetrics.functional import auroc

from CustomConv1d import CustomConv1d

PRINT = False

class SimpleModel(pl.LightningModule):
    def __init__(
        self,
        seq_length: int,
        kernel_size: int,
        alpha: float,
        beta: float,
        distribution: Tuple[float, float, float, float],
        l1_lambda: float,
        l2_lambda: float,
        lr: float,
    ) -> None:
        super().__init__()
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.lr = lr

        # self.conv1d = nn.Conv1d(kernel_size=12, in_channels=4, out_channels=512)
        self.conv1d = CustomConv1d(
            kernel_size=kernel_size,
            in_channels=4,
            out_channels=512,
            alpha=alpha,
            beta=beta,
            distribution=distribution
        )
        self.max_pool1d = nn.MaxPool1d(kernel_size=2*(seq_length-kernel_size+1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=512, out_features=32)
        self.linear2 = nn.Linear(in_features=32, out_features=2)
        self.softmax = nn.Softmax(dim=1)

        # TO DO: Check macro vs micro average

        metrics = MetricCollection([
            Accuracy(),
            # Precision(num_classes=2),
            # Recall(num_classes=2),
            # AUROC(num_classes=2),
        ])
        self.train_metrics = metrics.clone()
        self.valid_metrics = metrics.clone()
        self.test_metrics = metrics.clone()

    def forward(self, x_fw: Tensor, x_rv: Tensor) -> Tensor:
        conv_fw = self.conv1d(x_fw)
        conv_rv = self.conv1d(x_rv)
        merged = torch.cat((conv_fw, conv_rv), dim=2)
        max_pooled = self.max_pool1d(merged)
        flat = self.flatten(max_pooled)
        line1 = self.linear1(flat)
        relu1 = F.relu(line1)
        line2 = self.linear2(relu1)
        probs = self.softmax(line2)
        if PRINT:
            print(conv_fw.shape, '-> forward conv')
            print(conv_rv.shape, '-> reverse conv')
            print(merged.shape, '-> concat')
            print(max_pooled.shape, '-> max pool')
            print(flat.shape, '-> flatten')
            print(line1.shape, '-> linear 1')
            print(relu1.shape, '-> relu 1')
            print(line2.shape, '-> linear 2')
            print(probs.shape, '-> softmax')
        return probs

    def training_step(self, train_batch: Tensor, batch_idx: int) -> Tensor:
        # define the training loop

        X_fw, X_rv, y = train_batch
        logits = self.forward(X_fw, X_rv)
        loss = self.cross_entropy_loss(logits, y)

        metrics = self.train_metrics(logits, y.type(torch.int))

        self.log('train_loss', loss, prog_bar=True)
        self.log_dict(metrics)

        return loss

    def validation_step(self, val_batch: Tensor, batch_idx: int) -> None:
        X_fw, X_rv, y = val_batch
        logits = self.forward(X_fw, X_rv)
        loss = self.cross_entropy_loss(logits, y)

        metrics = self.valid_metrics(logits, y.type(torch.int))

        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(metrics)

    def test_step(self, test_batch: Tensor, batch_idx: int) -> None:
        X_fw, X_rv, y = test_batch
        logits = self.forward(X_fw, X_rv)
        loss = self.cross_entropy_loss(logits, y)

        metrics = self.test_metrics(logits, y.type(torch.int))

        self.log('test_loss', loss)
        self.log('test_auroc', auroc(logits, y.type(torch.int), num_classes=2))
        self.log_dict(metrics)


    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_lambda)
        return optimizer

    def cross_entropy_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        bce_loss = F.binary_cross_entropy(logits, labels)
        reg_loss = self.l1_lambda * sum(x.abs().sum() for x in self.linear1.parameters())
        return bce_loss + reg_loss


if __name__ == '__main__':
    PRINT = True
    seq_length = 156
    model = SimpleModel(
        seq_length=seq_length,
        kernel_size=12,
        alpha=1000,
        beta=1/1000,
        distribution=(0.3, 0.2, 0.2, 0.3),
        l1_lambda=1e-3,
        l2_lambda=0,
        lr=1e-3
    )
    ret = model(torch.ones(64, 4, seq_length), torch.ones(64, 4, seq_length))
