import numpy as np
from typing import Tuple
from skimage.util import view_as_windows


def compute_robert_cross(image: np.ndarray, pad_mode: str='edge') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Robert's cross which are gradients along the diagonals, +45 and -45 degrees.
    Please refer to https://en.wikipedia.org/wiki/Roberts_cross.
    """
    image_padded = np.pad(image, pad_width=[(1, 1), (1, 1)], mode=pad_mode)  # (H+2, W+2)

    ## We define the system of coordinate as (x, y) in [0, W-1]x[0, H-1].
    ## Grad m45 (minus 45 degrees) is defined in a 2x2 quad as
    ##  |-I(x, y)         -     |
    ##  |    -       I(x+1, y+1)|
    ## Grad p45 (plus 45 degrees) is defined in a 2x2 quad as
    ##  |    -        I(x+1, y)|
    #   | -I(x, y+1)       -   |
    grad_p45 = np.roll(image_padded, shift=(0, -1), axis=(0, 1)) - np.roll(image_padded, shift=(-1, 0), axis=(0, 1))
    grad_m45 = np.roll(image_padded, shift=(-1, -1), axis=(0, 1)) - image_padded

    return grad_p45, grad_m45


def compute_structure_tensor(image: np.ndarray, pad_mode: str='edge') -> np.ndarray:
    """
    Compute the structure tensor from a grayscale image. We use the Robert's cross to
    compute the gradients for maximal edge detection. Please refer to
    https://bartwronski.com/2021/02/28/computing-gradients-on-grids-forward-central-and-diagonal-differences/.

    Inputs:
        image, (H, W) np.ndarray: the input image.

    Outputs:
        tensor, (H, W, 2, 2) np.ndarray: the 2x2 structure tensors per pixel.
    """
    ## Get image shape and build tensor.
    h, w = image.shape
    structure_tensor = np.zeros((h, w, 2, 2), image.dtype)  # (H, W, 2, 2)

    ## Compute the diagonal gradients (Robert cross.)
    grad_p45, grad_m45 = compute_robert_cross(image=image, pad_mode=pad_mode)

    ## Compute the structure coefficients in the (-45, 45) system of coordinates.
    windows_p45 = view_as_windows(arr_in=grad_p45, window_shape=(3, 3), step=(1, 1))  # (H, W, 3, 3)
    windows_m45 = view_as_windows(arr_in=grad_m45, window_shape=(3, 3), step=(1, 1))  # (H, W, 3, 3)

    sxx = np.sum(windows_m45 * windows_m45, axis=(2, 3))  # (H, W)
    syy = np.sum(windows_p45 * windows_p45, axis=(2, 3))  # (H, W)
    sxy = np.sum(windows_p45 * windows_m45, axis=(2, 3))  # (H, W)

    # ## Rotate the structure tensor coefficients by + 45 degrees to align with the (0, 90) system of coordinates.
    structure_tensor[..., 0, 0] = 0.5 * (sxx + syy) - sxy  # (H, W)
    structure_tensor[..., 0, 1] = sxx - syy  # (H, W)
    structure_tensor[..., 1, 0] = sxx - syy  # (H, W)
    structure_tensor[..., 1, 1] = 0.5 * (sxx + syy) + sxy  # (H, W)

    return structure_tensor
