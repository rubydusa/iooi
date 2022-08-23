import math
from enum import Enum
from functools import reduce
from typing import Any, Callable

import cv2
import numpy as np
import numpy.typing as npt

import common
from image_map import ImageMap


class Mode(Enum):
    SINGLE_LAYER = 1
    MULTI_LAYERED = 2


# apply image map over src, and paste upon source
# assumes all images in img_map are the same size
def single_layer(img: npt.NDArray, img_map: ImageMap) -> npt.NDArray:
    output = transform_img(img, img_map)
    output = common.overlay_imgs(common.resize_to(img, output), output)

    return output


# apply image map over src, fill empty space by shifting image colors
# assumes all images in img_map are the same size
# assumes steps < unit_x
def multi_layered(img: npt.NDArray, img_map: ImageMap, *args) -> npt.NDArray:
    steps = multi_layered_parse_args(*args)

    img_borderd = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    unit_x, unit_y = img_map.imgs[0].img.shape[:2][::-1]

    layers = []

    for i in range(steps):
        precent = i / steps
        offset_x, offset_y = math.floor(unit_x * precent), math.floor(unit_y * precent)

        offseted_averaged_img = common.offseted_averaged(img_borderd, offset_x, unit_x)
        transformed_img = transform_img(offseted_averaged_img, img_map)
        transformed_img = common.offseted_cut(transformed_img, offset_x, offset_y)

        layers.append(transformed_img)

    # prioritize layer with least offset
    layers = layers[::-1]

    output = reduce(lambda base, top: common.overlay_imgs(base, top), layers)
    output = output[unit_y:-unit_y, unit_x:-unit_x, :]
    output = common.overlay_imgs(common.resize_to(img, output), output)

    return output


def multi_layered_parse_args(*args) -> int:
    return int(args[0])


# src is bgr
def transform_img(img: npt.NDArray, img_map: ImageMap) -> npt.NDArray:
    indices = np.apply_along_axis(lambda x: img_map.colors[tuple(x)], 2, img)
    output = np.array([i.img for i in img_map.imgs])[indices]
    output = np.concatenate(output, axis=1)
    output = np.concatenate(output, axis=1)

    return output


# every filter must be [npt.NDArray, ImageMap, *str] -> npt.NDArray
def dispatcher() -> dict[Mode, Callable[..., npt.NDArray]]:
    return {Mode.SINGLE_LAYER: single_layer, Mode.MULTI_LAYERED: multi_layered}
