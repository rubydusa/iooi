from pathlib import Path
import argparse
import json
import numpy as np
import numpy.typing as npt
import cv2

import common
from image_map import ImageMap

# src is rgb
def transform_img(src: npt.NDArray, img_map: ImageMap) -> npt.NDArray:
    indices = np.apply_along_axis(lambda x: img_map.colors[tuple(x)], 2, src)
    output = np.array([i.img for i in img_map.imgs])[indices]
    output = np.concatenate(output, axis=1)
    output = np.concatenate(output, axis=1)

    return output

def run(src: Path, out: Path, imgs: Path, pallet: Path):
    src_img = cv2.imread(str(src))

    with open(pallet) as pallet_f:
        img_map = json.load(pallet_f)
    
    img_map = ImageMap.deserialize(img_map)
    img_map.load(imgs)

    output = transform_img(src_img, img_map)
    cv2.imwrite(str(out), output)

def main():
    parser = argparse.ArgumentParser("Create Images Out Of Images (IOOI)")
    parser.add_argument("src", type=Path, help="Path to source image")
    parser.add_argument("out", type=Path, help="Output path")
    parser.add_argument("-i", "--images", nargs="?", default=Path("./imgs"), type=Path, help="Prepared images directory (default: ./imgs)")
    parser.add_argument("-p", "--pallet", nargs="?", default=Path("./pallet.json"), type=Path, help="Pallet JSON file (default: ./pallet.json)")
    args = parser.parse_args()

    run(args.src, args.out, args.images, args.pallet)

if __name__ == "__main__":
    main()
