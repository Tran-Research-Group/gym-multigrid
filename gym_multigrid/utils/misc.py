from matplotlib import animation
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import random


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def save_frames_as_gif(frames, path="./", filename="collect-", ep=0):
    filename = filename + str(ep) + ".gif"
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer="imagemagick", fps=60)
    plt.close()

def cartesian_product(self, arr1, arr2):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    
    # Ensure arr1 and arr2 are 2D
    if arr1.ndim == 1:
        arr1 = arr1[:, np.newaxis]  # Convert 1D to 2D (n, 1)
    if arr2.ndim == 1:
        arr2 = arr2[:, np.newaxis]  # Convert 1D to 2D (p, 1)

    n, m = arr1.shape
    p, r = arr2.shape
    
    # Create a meshgrid for Cartesian product
    arr1_expanded = arr1[:, np.newaxis, :]  # Shape (n, 1, m)
    arr2_expanded = arr2[np.newaxis, :, :]  # Shape (1, p, r)
    
    # Compute Cartesian product
    cart_product = np.concatenate([np.tile(arr1_expanded, (1, p, 1)), np.tile(arr2_expanded, (n, 1, 1))], axis=-1)
    
    # Reshape to the desired output shape (n*p, m + r)
    result = cart_product.reshape(n * p, m + r)
    
    return result
