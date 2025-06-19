import numpy as np
import torch
import torch.nn.functional as F
from basicsr.metrics.metric_util import reorder_image, to_y_channel


def calculate_local_mse_cv(img1,
                           img2,
                           crop_border,
                           input_order='HWC',
                           test_y_channel=False,
                           pool_size=32):
    """Calculate coefficient of variation (CV) of local MSEs between two images.

    Args:
        img1 (ndarray/tensor): Image with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Image with range [0, 255]/[0, 1].
        crop_border (int): Border to crop before calculation.
        input_order (str): 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Whether to convert to Y channel. Default: False.
        pool_size (int): Pool size for local MSE. Default: 32.

    Returns:
        float: CV (std/mean) of local MSEs.
    """
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported: HWC, CHW.')

    # If tensor -> numpy
    if isinstance(img1, torch.Tensor):
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0) if input_order == 'CHW' else img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0) if input_order == 'CHW' else img2.detach().cpu().numpy()

    # Reorder to HWC, float64
    img1 = reorder_image(img1, input_order=input_order).astype(np.float64)
    img2 = reorder_image(img2, input_order=input_order).astype(np.float64)

    # Crop border
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Y channel
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
        img1 = img1[..., None]
        img2 = img2[..., None]

    if img1.ndim == 2:
        img1 = img1[..., None]
        img2 = img2[..., None]

    # Convert to tensor [1,C,H,W]
    img1_t = torch.from_numpy(img1).permute(2,0,1).unsqueeze(0).float()
    img2_t = torch.from_numpy(img2).permute(2,0,1).unsqueeze(0).float()

    # Normalize if necessary
    if img1_t.max() > 1:
        img1_t = img1_t / 255.0
        img2_t = img2_t / 255.0

    # MSE map
    mse_map = (img1_t - img2_t) ** 2
    mse_map = mse_map.mean(1, keepdim=True)  # [1,1,H,W]

    # Pooling
    local_mse = F.avg_pool2d(mse_map, kernel_size=pool_size, stride=pool_size)
    local_mse_flat = local_mse.flatten()

    mean_mse = local_mse_flat.mean().item()
    std_mse = local_mse_flat.std().item()

    if mean_mse == 0:
        return 0.0

    return std_mse / mean_mse