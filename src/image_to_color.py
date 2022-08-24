from dataclasses import dataclass
from enum import Enum
from typing import Callable

import cv2
import numpy as np
import numpy.typing as npt

import common


class Mode(Enum):
    KMEAN = 1
    AVERAGE = 2


# ===== #
# Modes #
# ===== #

# return dominant color in image using weighted kmean
def kmean(img: npt.NDArray) -> npt.NDArray:
    channels = img.shape[2]
    pixels = img.reshape(-1, channels).astype(np.float32)

    pixels, weights = pixels[:, :3], pixels[:, 3]

    result = kmean_weighted(pixels, weights, KMeanWeightedArgs(3))
    dominant = result.centroids[0][:3].astype(np.uint8)
    return dominant


# average color, alpha used as weight
def average(img: npt.NDArray) -> npt.NDArray:
    weights = img[:, :, 3]
    weights = np.expand_dims(weights, axis=2)
    weights = np.repeat(weights, repeats=4, axis=2)

    avg_color = np.round(np.average(img, weights=weights, axis=(0, 1)), decimals=0)[:3]
    return avg_color.astype(np.uint8)


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
            [
                common.color_distance(centroid[:-1], values[:, :-1])
                for centroid in centroids
            ]
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


# ========== #
# Dispatcher #
# ========== #


def dispatcher() -> dict[Mode, Callable[[npt.NDArray], npt.NDArray]]:
    return {Mode.KMEAN: kmean, Mode.AVERAGE: average}
