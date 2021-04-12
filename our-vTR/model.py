# Model: "model"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to                     
# ==================================================================================================
# forward (InputLayer)            [(None, 156, 4)]     0                                            
# __________________________________________________________________________________________________
# reverse (InputLayer)            [(None, 156, 4)]     0                                            
# __________________________________________________________________________________________________
# convolution_layer (ConvolutionL (None, 145, 512)     25088       forward[0][0]                    
#                                                                  reverse[0][0]                    
# __________________________________________________________________________________________________
# concatenate (Concatenate)       (None, 290, 512)     0           convolution_layer[0][0]          
#                                                                  convolution_layer[1][0]          
# __________________________________________________________________________________________________
# max_pooling1d (MaxPooling1D)    (None, 1, 512)       0           concatenate[0][0]                
# __________________________________________________________________________________________________
# flatten (Flatten)               (None, 512)          0           max_pooling1d[0][0]              
# __________________________________________________________________________________________________
# dense (Dense)                   (None, 32)           16416       flatten[0][0]                    
# __________________________________________________________________________________________________
# dense_1 (Dense)                 (None, 2)            66          dense[0][0]                      
# ==================================================================================================
# Total params: 41,570
# Trainable params: 41,570
# Non-trainable params: 0
# __________________________________________________________________________________________________

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy


class SimpleModel(pl.LightningModule):
    def __init__(self) -> None:
        super(SimpleModel, self).__init__()
        self.conv1d = nn.Conv1d(kernel_size=12, in_channels=4, out_channels=512)
        self.max_pool1d = nn.MaxPool1d(kernel_size=290)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=512, out_features=32)
        self.linear2 = nn.Linear(in_features=32, out_features=2)
        self.softmax = nn.Softmax(dim=1)

        metric = Accuracy()
        self.train_metric = metric.clone()
        self.valid_metric = metric.clone()
        self.test_metric = metric.clone()

    def forward(self, x_fw: Tensor, x_rv: Tensor) -> Tensor:
        conv_fw = self.conv1d(x_fw)
        # print(conv_fw.shape, '-> forward conv')
        conv_rv = self.conv1d(x_rv)
        # print(conv_rv.shape, '-> reverse conv')
        merged = torch.cat((conv_fw, conv_rv), dim=2)
        # print(merged.shape, '-> concat')
        max_pooled = self.max_pool1d(merged)
        # print(max_pooled.shape, '-> max pool')
        flat = self.flatten(max_pooled)
        # print(flat.shape, '-> flatten')
        line1 = self.linear1(flat)
        # print(line1.shape, '-> linear 1')
        relu1 = F.relu(line1)
        # print(relu1.shape, '-> relu 1')
        line2 = self.linear2(relu1)
        # print(line2.shape, '-> linear 2')
        probs = self.softmax(line2)
        # print(probs.shape, '-> softmax')
        return probs
    
    def training_step(self, train_batch, batch_idx):
        # define the training loop

        X_fw, X_rv, y = train_batch
        logits = self.forward(X_fw, X_rv)
        loss = self.cross_entropy_loss(logits, y)
        acc = 100*self.train_metric(logits, y.type(torch.int))

        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        X_fw, X_rv, y = val_batch
        logits = self.forward(X_fw, X_rv)
        loss = self.cross_entropy_loss(logits, y)
        acc = 100*self.valid_metric(logits, y.type(torch.int))
        
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        X_fw, X_rv, y = test_batch
        logits = self.forward(X_fw, X_rv)
        loss = self.cross_entropy_loss(logits, y)
        acc = 100*self.test_metric(logits, y.type(torch.int))

        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def cross_entropy_loss(self, logits, labels):
        return F.binary_cross_entropy(logits, labels)
    


# model = SimpleModel() # to(device)
# ret = model(torch.ones(64, 4, 156), torch.ones(64, 4, 156))