import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from typing import Callable
from model import SimpleModel

def loss_batch(model: nn.Module,
              loss_func: Callable, xb_fw: Tensor, xb_rv: Tensor, yb: Tensor, opt=None) -> tuple:
    loss = loss_func(model(xb_fw, xb_rv), yb)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), len(xb_fw)


def fit(
        epochs: int,
        model: SimpleModel,
        loss_func: Callable,
        optimizer: Optimizer,
        train_dl: DataLoader,
        valid_dl: DataLoader 
    ) -> None:
    for epoch in range(epochs):
        model.train()
        for fw, rv, yb in train_dl:
            loss_batch(model, loss_func, fw, rv, yb, optimizer)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, fw, rv, yb) for fw, rv, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)