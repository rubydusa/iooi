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


# return dominant color using k-mean clustering
# output color always has 3 channels
def img_to_color(img: npt.NDArray) -> npt.NDArray:
    channels = img.shape[2]
    pixels = img.reshape(-1, channels).astype(np.float32)

    weights = np.ones((pixels.shape[0]), dtype=np.float32)
    # use alpha as weight
    if channels == 4:
        pixels, weights = pixels[:, :-1], pixels[:, -1]

    result = kmean_weighted(pixels, weights, KMeanWeightedArgs(3))
    dominant = result.centroids[0][:3].astype(np.uint8)
    return dominant


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
    return cv2.resize(src, dest.shape[:2][::-1], interpolation=inter)


# ========== #
# Algorithms #
# ========== #


@dataclass(frozen=True)
class KMeanWeightedArgs:
    k: int
    repeats: int = 200


@dataclass(frozen=True)
class KMeanWeightedResult:
    centroids: npt.NDArray[np.float32]
    cluster_sizes: npt.NDArray[np.uint32]
    labels: npt.NDArray[np.uint32]


# a is 2 dimensional and weights is 1 dimensional
# mean calculation is weighted
def kmean_weighted(
    a: npt.NDArray[np.float32],
    weights: npt.NDArray[np.float32],
    args: KMeanWeightedArgs,
) -> KMeanWeightedResult:
    values = np.column_stack((a, weights))

    # random centroids
    centroids = values[np.random.choice(np.arange(values.shape[0]), args.k)]
    cluster_sizes = np.zeros((args.k), dtype=np.uint32)
    labels = np.array([])

    for _ in range(args.repeats):
        # disregard weight in distance
        distances = np.array(
            [color_distance(centroid[:-1], values[:, :-1]) for centroid in centroids]
        )
        labels = np.argmin(distances, axis=0)

        # -1 is cluster group and -2 is weight arg
        values_indexed = np.column_stack((values, labels))
        values_indexed = values_indexed[np.argsort(values_indexed[:, -1])]

        clusters = np.split(
            values_indexed[:, :-1],
            np.unique(values_indexed[:, -1], return_index=True)[1][1:],
        )

        centroids = np.array(
            [
                np.average(cluster, weights=cluster[:, -1], axis=0)
                if np.sum(cluster[:, -1]) > 0
                else np.average(cluster, axis=0)
                for cluster in clusters
            ]
        )
        cluster_sizes = np.array([cluster.shape[0] for cluster in clusters])

    # sort from biggest to smallest
    order = np.argsort(cluster_sizes)[::-1]

    centroids = centroids[order]
    cluster_sizes = cluster_sizes[order]
    labels = labels.astype(np.uint32)

    return KMeanWeightedResult(centroids, cluster_sizes, labels)
