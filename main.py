import numpy as np
from skimage import data, img_as_float32
import matplotlib.pyplot as plt

from structure_tensor import compute_structure_tensor


import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt

def flow_to_color(flow):
    """
    Convert a flow field to a color representation with saturation weighted by the flow magnitude.
    
    Parameters:
        flow (np.ndarray): Array of shape (H, W, 2) with 2D flow vectors.
    
    Returns:
        np.ndarray: An (H, W, 3) RGB image with values in [0, 1] encoding the flow direction and magnitude.
    """
    # Separate horizontal (u) and vertical (v) components.
    u = flow[..., 0]
    v = flow[..., 1]
    
    # Compute the angle of each vector (range: [-pi, pi]).
    angle = np.arctan2(v, u)
    
    # Map angle from [-pi, pi] to [0, 1] for the hue channel.
    hue = (angle + np.pi) / (2 * np.pi)
    
    # Compute the magnitude of the flow vectors.
    mag = np.sqrt(u**2 + v**2)
    
    # Normalize the magnitude to the range [0, 1] for saturation.
    # Adding a small epsilon to avoid division by zero.
    max_mag = np.max(mag)
    saturation = mag / (max_mag + 1e-5)
    
    # Set value (brightness) to 1.
    value = np.ones_like(hue)
    
    # Stack the HSV channels.
    hsv = np.stack((hue, saturation, value), axis=-1)
    
    # Convert HSV to RGB.
    rgb = hsv_to_rgb(hsv)
    return rgb

def hsv_to_rgb(hsv):
    """
    Convert an HSV image to RGB.
    
    Parameters:
        hsv (np.ndarray): Array of shape (..., 3) with h, s, v in [0, 1].
        
    Returns:
        np.ndarray: Array of same shape in RGB format.
    """
    h = hsv[..., 0] * 6.0  # scale hue to [0,6]
    s = hsv[..., 1]
    v = hsv[..., 2]
    
    i = np.floor(h).astype(int)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    
    i = i % 6  # wrap around
    
    # Allocate arrays for the RGB channels.
    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)
    
    # Use masks for each of the 6 sectors.
    mask = (i == 0)
    r[mask] = v[mask]
    g[mask] = t[mask]
    b[mask] = p[mask]
    
    mask = (i == 1)
    r[mask] = q[mask]
    g[mask] = v[mask]
    b[mask] = p[mask]
    
    mask = (i == 2)
    r[mask] = p[mask]
    g[mask] = v[mask]
    b[mask] = t[mask]
    
    mask = (i == 3)
    r[mask] = p[mask]
    g[mask] = q[mask]
    b[mask] = v[mask]
    
    mask = (i == 4)
    r[mask] = t[mask]
    g[mask] = p[mask]
    b[mask] = v[mask]
    
    mask = (i == 5)
    r[mask] = v[mask]
    g[mask] = p[mask]
    b[mask] = q[mask]
    
    rgb = np.stack([r, g, b], axis=-1)
    return rgb


def main():
    ## Get a grayscale image.
    # img = img_as_float32(data.camera())  # (H, W)
    # img = img_as_float32(data.astronaut()).mean(axis=-1)  # (H, W)
    img = img_as_float32(data.cat()).mean(axis=-1)  # (H, W)
    # img = np.zeros((256, 256)); img[:, 128] = 1  # just a straight line
    # img = np.zeros((256, 256)); img[128, :] = 1  # just a straight line

    ## Compute the structure tensor.
    S = compute_structure_tensor(img)  # (H, W, 2, 2)

    ## Compute the eigen decomposition of the structure tensor.
    eigvals, eigvecs = np.linalg.eig(S)  # (H, W, 2), (H, W, 2, 2)

    grad_mag = np.sqrt(eigvals[...,0] ** 2 + eigvals[...,1] ** 2)  # (H, W)
    grad_dir = eigvecs[..., 0]  # (H, W, 2)

    motion = grad_mag[..., None] * grad_dir

    color_flow = flow_to_color(motion)

    plt.figure()
    plt.imshow(color_flow)
    plt.show()



if __name__ == "__main__":
    main()
