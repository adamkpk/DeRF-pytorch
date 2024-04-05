import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import lpips


def compute_rmse(prediction, target):
    return np.sqrt(((prediction - target) ** 2).mean())


def compute_psnr(prediction, target):
    mse = np.mean((prediction - target) ** 2)

    if mse == 0:
        return float('inf')

    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(prediction, target):
    pred_gray = np.array(Image.fromarray(prediction).convert('L'))
    targ_gray = np.array(Image.fromarray(target).convert('L'))
    return structural_similarity(pred_gray, targ_gray)


def compute_lpips(prediction, target):
    loss_fn = lpips.LPIPS(net='alex')
    return loss_fn.forward(prediction, target)
