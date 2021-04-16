from torchmetrics import Precision, Recall, Accuracy, F1, MetricCollection
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import auroc
from CustomConv1d import CustomConv1d

PRINT = False

class SimpleModel(pl.LightningModule):
    def __init__(self) -> None:
        super(SimpleModel, self).__init__()
        # self.conv1d = nn.Conv1d(kernel_size=12, in_channels=4, out_channels=512)
        self.conv1d = CustomConv1d(kernel_size=12, in_channels=4, out_channels=512)
        self.max_pool1d = nn.MaxPool1d(kernel_size=290)
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
    
    def training_step(self, train_batch, batch_idx):
        # define the training loop

        X_fw, X_rv, y = train_batch
        logits = self.forward(X_fw, X_rv)
        loss = self.cross_entropy_loss(logits, y)

        self.train_metrics(logits, y.type(torch.int))

        self.log('train_loss', loss)
        self.log_dict(self.train_metrics, prog_bar=False)

        return loss

    def validation_step(self, val_batch, batch_idx):
        X_fw, X_rv, y = val_batch
        logits = self.forward(X_fw, X_rv)
        loss = self.cross_entropy_loss(logits, y)

        self.valid_metrics(logits, y.type(torch.int))

        self.log('val_loss', loss)
        self.log_dict(self.valid_metrics, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        X_fw, X_rv, y = test_batch
        logits = self.forward(X_fw, X_rv)
        loss = self.cross_entropy_loss(logits, y)
        
        self.test_metrics(logits, y.type(torch.int))

        self.log('test_loss', loss)
        self.log('test_auroc', auroc(logits, y.type(torch.int), num_classes=2))
        self.log_dict(self.test_metrics)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def cross_entropy_loss(self, logits, labels):
        bce_loss = F.binary_cross_entropy(logits, labels)
        all_linear1_params = torch.cat(
            [x.view(-1) for x in self.linear1.parameters()])
        reg_loss = 0.0001 * torch.norm(all_linear1_params, 1)
        return bce_loss + reg_loss


if __name__ == '__main__':
    PRINT = True
    model = SimpleModel()  # to(device)
    ret = model(torch.ones(64, 4, 156), torch.ones(64, 4, 156))
