from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor
import lpips
import numpy as np

from basicsr.metrics.metric_util import reorder_image, to_y_channel

import torch

def calculate_lpips(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):

    """Calculate LPIPS.
    Ref: https://github.com/xinntao/BasicSR/pull/367
    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: LPIPS result.
    """
    assert img.shape == img2.shape, (f'Image shapes are differnet: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)
        # Convert to 3 channels for LPIPS (LPIPS expects 3 channels)
        img = np.repeat(img[..., None], 3, axis=2)
        img2 = np.repeat(img2[..., None], 3, axis=2)

    # start calculating LPIPS metrics
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn_vgg = lpips.LPIPS(net='vgg', verbose=False).to(DEVICE)  # RGB, normalized to [-1,1]

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    # Normalize to [0,1] and convert to float32
    img = (img / 255.).astype(np.float32)
    img2 = (img2 / 255.).astype(np.float32)

    img_gt, img_restored = img2tensor([img2, img], bgr2rgb=True, float32=True)

    normalize(img_gt, mean, std, inplace=True)
    normalize(img_restored, mean, std, inplace=True)

    img_gt = img_gt.to(DEVICE)
    img_restored = img_restored.to(DEVICE)

    loss_fn_vgg.eval()
    with torch.no_grad():
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0), img_gt.unsqueeze(0))

    return lpips_val.detach().cpu().numpy().mean()