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
from pytorch_wavelets import DWTForward
from basicsr.models.losses.loss_util import weighted_loss
from basicsr.models.archs.vgg_arch import VGGFeatureExtractor

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
    def __init__(self, loss_weight=1.0,
                 reduction='mean', 
                 wave='haar', 
                 level=1, 
                 loss_fn=nn.L1Loss()):
        super().__init__()
        self.dwt = DWTForward(J=level, wave=wave, mode='zero')  # zero padding, can change to 'symmetric'
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        yl_pred, yh_pred = self.dwt(pred)
        yl_target, yh_target = self.dwt(target)

        # Loss on low frequency (approximation)
        loss_ll = self.loss_fn(yl_pred, yl_target)  # shape [B, C, H, W]

        # Loss on high-frequency (details)
        loss_h = 0
        for yh_p, yh_t in zip(yh_pred, yh_target):
            for p, t in zip(yh_p, yh_t):
                loss_h += self.loss_fn(p, t)

        # Combine losses
        loss = loss_ll + loss_h  # still shape [B, C, H, W]

        # Reduce over all elements and batch
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return self.loss_weight * loss

class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 loss_weight=1.0,
                 layer_weights=None,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.loss_weight = loss_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        if layer_weights is None:
            layer_weights = {'conv2_2': 1.0, 'conv3_4': 1.0, 'conv4_4': 1.0}
        self.layer_weights = layer_weights

        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)
        
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Scalar perceptual + style loss.
        """
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        percep_loss = 0.0
        style_loss = 0.0

        if self.perceptual_weight > 0:
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    diff = x_features[k] - gt_features[k]
                    loss = diff.view(diff.size(0), -1).norm(p='fro', dim=1).mean()
                else:
                    loss = self.criterion(x_features[k], gt_features[k])
                percep_loss += self.layer_weights[k] * loss
            percep_loss *= self.perceptual_weight

        if self.style_weight > 0:
            for k in x_features.keys():
                gram_x = self._gram_mat(x_features[k])
                gram_gt = self._gram_mat(gt_features[k])
                if self.criterion_type == 'fro':
                    diff = gram_x - gram_gt
                    loss = diff.view(diff.size(0), -1).norm(p='fro', dim=1).mean()
                else:
                    loss = self.criterion(gram_x, gram_gt)
                style_loss += self.layer_weights[k] * loss
            style_loss *= self.style_weight

        # Return combined scalar (or 0.0 if unused)
        total_loss = self.loss_weight * (percep_loss + style_loss)

        return total_loss

def _gram_mat(self, x):
    """
    Calculate the Gram matrix of input feature maps.

    The Gram matrix is used to measure the correlations between the different
    feature channels. It is commonly used in style loss computations.

    Args:
        x (torch.Tensor): Feature maps with shape (n, c, h, w), where
            n = batch size,
            c = number of channels,
            h = height,
            w = width.

    Returns:
        torch.Tensor: Gram matrix of shape (n, c, c), normalized by the
            number of elements (c * h * w).
    """
    n, c, h, w = x.size()
    # Reshape to (n, c, h*w) to treat each feature map as a vector
    features = x.view(n, c, h * w)
    # Transpose features to (n, h*w, c) for batch matrix multiplication
    features_t = features.transpose(1, 2)
    # Compute batch matrix multiplication of features and its transpose
    gram = features.bmm(features_t) / (c * h * w)  # Normalize by total elements
    return gram