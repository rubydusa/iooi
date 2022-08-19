from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image as PILImage


@dataclass
class Image:
    name: str
    img: npt.NDArray


# =================== #
# Image Map Utilities #
# =================== #


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
    output = output * [100 / 255, 255 / 184, 255 / 203]
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


# ============== #
# Main Utilities #
# ============== #


# assumes base and top are both BGR(A)
def overlay_imgs(
    base: npt.NDArray,
    top: npt.NDArray,
    dest: tuple[int, int] = (0, 0),
    source: tuple[int, int] = (0, 0),
) -> npt.NDArray:
    output = PILImage.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGBA))
    output.alpha_composite(
        PILImage.fromarray(cv2.cvtColor(top, cv2.COLOR_BGR2RGBA)),
        dest=dest,
        source=source,
    )

    return cv2.cvtColor(np.asarray(output), cv2.COLOR_RGBA2BGRA)


# offset image right-down and use fill values of edge over resized
# take average over shape / resize grid
# returned array is same shape as img
def offseted_averaged(img: npt.NDArray, offset: int, resize: int) -> npt.NDArray:
    resized_shape = np.array(img.shape[:2][::-1]) * resize
    resized_img = cv2.resize(img, dsize=resized_shape, interpolation=cv2.INTER_NEAREST)
    reverse_offset = resize - offset

    padded_img = np.pad(
        resized_img,
        ((offset, reverse_offset), (offset, reverse_offset), (0, 0)),
        mode="edge",
    )
    h, w, c = padded_img.shape
    chunked_img = padded_img.reshape((h // resize, resize, w // resize, resize, c))
    averaged_img = np.average(chunked_img, axis=(1, 3))
    averaged_img = averaged_img[:-1, :-1, :]
    averaged_img = np.around(averaged_img)
    averaged_img = np.clip(averaged_img, 0, 255)
    averaged_img = averaged_img.astype(np.uint8)

    return averaged_img


# offset image right-down and cut to fit original resolution
# uses transperent pixels for BGRA/RGBA and black for BGR/RGB
def offseted_cut(img: npt.NDArray, offset_x: int, offset_y: int) -> npt.NDArray:
    offseted_img = np.pad(
        img, ((offset_y, 0), (offset_x, 0), (0, 0)), mode="constant", constant_values=0
    )
    x, y = offseted_img.shape[:2][::-1]
    cut_img = offseted_img[: y - offset_y, : x - offset_x, :]

    return cut_img


def resize_to(
    src: npt.NDArray, dest: npt.NDArray, inter=cv2.INTER_NEAREST
) -> npt.NDArray:
    return cv2.resize(src, dest.shape[:2][::-1], inter)
