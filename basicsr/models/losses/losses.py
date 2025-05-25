# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import pywt

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):
    """PSNR loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        toY (bool): Whether to convert RGB to Y channel. Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class DWTLoss(nn.Module):
    """DWT-based loss. Computes L1 loss on wavelet-decomposed images.

    Args:
        loss_weight (float): Weight of the loss.
        wavelet (str): Type of wavelet to use (e.g., 'haar', 'db1').
        reduction (str): Reduction method.
    """

    def __init__(self, loss_weight=1.0, wavelet='haar', reduction='mean'):
        super(DWTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.wavelet = wavelet
        self.reduction = reduction

    def dwt_decompose(self, img):
        # Input: (N, C, H, W), Output: same shape, with concatenated DWT channels
        coeffs = []
        for c in range(img.shape[1]):
            # Apply DWT per channel
            cfs = [pywt.dwt2(img[b, c].cpu().numpy(), self.wavelet) for b in range(img.shape[0])]
            LL, (LH, HL, HH) = zip(*cfs)
            coeffs.append([
                torch.tensor(np.stack(LL)).unsqueeze(1), # pylint: disable=no-member
                torch.tensor(np.stack(LH)).unsqueeze(1), # pylint: disable=no-member
                torch.tensor(np.stack(HL)).unsqueeze(1), # pylint: disable=no-member
                torch.tensor(np.stack(HH)).unsqueeze(1), # pylint: disable=no-member
            ])
        # Stack across channels
        components = [torch.cat([coeffs[c][i] for c in range(img.shape[1])], dim=1).to(img.device) for i in range(4)] # pylint: disable=no-member
        return components  # [LL, LH, HL, HH], each (N, C, H/2, W/2)

    def forward(self, pred, target):
        pred_coeffs = self.dwt_decompose(pred)
        target_coeffs = self.dwt_decompose(target)

        total_loss = 0.0
        for p, t in zip(pred_coeffs, target_coeffs):
            total_loss += F.l1_loss(p, t, reduction=self.reduction)

        return self.loss_weight * total_loss

