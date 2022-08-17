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
    avg_color = np.round(np.average(img, weights=weights, axis=(0, 1)), decimals=0)[:3]  
    return avg_color.astype(np.uint8)

def format_lab(lab_img: npt.NDArray) -> npt.NDArray:
    # cv2 weird lab colors: 0 <= L <= 255, 42 <= A <= 226, 20 <= B <= 223
    # correct lab colors: 0 <= L <= 100, -128 <= A <= 127, -128 <= B <= 127

    output = lab_img.astype(np.float64)
    output = output - [0, 42, 20]
    output = output * [100/255, 255/184, 255/203]
    output = output - [0, 128, 128]

    return output
    

# all BGR colors
def all_colors() -> npt.NDArray:
    colors = np.array([int_to_bgr(i) for i in range(2**24)])
    return colors.astype(np.uint8)

# euclidian distance (not rooted)
def color_distance(color1, color2) -> npt.NDArray:
    return np.sum((color1 - color2) ** 2, axis=-1)

def int_to_bgr(color: int) -> npt.NDArray:
    digits = []
    remaining = color
    
    for _ in range(3):
        digits.append(remaining % 256)
        remaining //= 256

    return np.array(digits[::-1])

