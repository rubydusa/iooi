import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

@dataclass
class Image:
    name: str
    img: npt.NDArray

# get average color (BGRA), alpha used as weight
# result is BGR without alpha
def img_to_color(img: npt.NDArray) -> npt.NDArray:
    weights = img[:, :, 3]
    weights = np.expand_dims(weights, axis=2)
    weights = np.repeat(weights, repeats=4, axis=2)

    # [:3] is in order to remove alpha channel
    avg_color = np.round(np.average(img, weights=weights, axis=(0, 1)), decimals=0)[:3]  # type: ignore
    return avg_color.astype(np.uint8)

# all BGR colors
def all_colors() -> npt.NDArray:
    colors = np.array([int_to_rgb(i) for i in range(2**24)])
    return colors.astype(np.uint8)

# euclidian distance (not rooted)
def color_distance(color1, color2) -> npt.NDArray:
    return np.sum((color1 - color2) ** 2, axis=2)

def int_to_rgb(color: int) -> npt.NDArray:
    digits = []
    remaining = color
    
    for _ in range(3):
        digits.append(remaining % 256)
        remaining //= 256

    return np.array(digits)

# serialization and deserialization of image maps do not necessarily preserve order of images
# check if every number in arr1 can be mapped to every number in arr2
def correspondence(arr1: npt.NDArray, arr2: npt.NDArray) -> bool:
    if arr1.shape != arr2.shape:
        return False

    paired = np.stack([arr1, arr2], axis=-1)
    return np.unique(arr1).size == np.unique(paired.reshape(-1, 2), axis=0).size

