import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from filters import Mode, dispatcher
from image_map import ImageMap


# load image, lookup dispatcher and apply filter with image map, write to out
def run(
    src: Path, out: Path, imgs: Path, pallet: Path, dispatcher: dict, mode: Mode, *args
):
    src_img = cv2.imread(str(src))

    with open(pallet) as pallet_f:
        img_map = json.load(pallet_f)

    img_map = ImageMap.deserialize(img_map)
    img_map.load(imgs)

    output = dispatcher[mode](src_img, img_map, *args)

    cv2.imwrite(str(out), output)


def main():
    parser = argparse.ArgumentParser("Create Images Out Of Images (IOOI)")
    parser.add_argument("src", type=Path, help="Path to source image")
    parser.add_argument("out", type=Path, help="Output path")
    parser.add_argument(
        "-i",
        "--images",
        nargs="?",
        default=Path("./imgs"),
        type=Path,
        help="Prepared images directory (default: ./imgs)",
    )
    parser.add_argument(
        "-p",
        "--pallet",
        nargs="?",
        default=Path("./pallet.json"),
        type=Path,
        help="Pallet JSON file (default: ./pallet.json)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        nargs="?",
        default=Mode.SINGLE_LAYER.name,
        type=str,
        choices=[mode.name for mode in list(Mode)],
        help="Filter to apply (default: SINGLE_LAYER)",
    )
    parser.add_argument(
        "-a",
        "--args",
        nargs="*",
        help="Additional positional arguments, depends on Mode",
    )
    args = parser.parse_args()

    filter_args = args.args if args.args else []
    run(
        args.src,
        args.out,
        args.images,
        args.pallet,
        dispatcher(),
        Mode[args.mode],
        *filter_args
    )


if __name__ == "__main__":
    main()
