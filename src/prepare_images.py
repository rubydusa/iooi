import argparse
import json
from pathlib import Path
from typing import Optional

import cv2

import common
from image_map import ImageMap


def run(
    src: Path,
    dest: Path,
    width: int,
    height: int,
):
    for img_path in src.iterdir():
        img_name = img_path.name
        img_dest = dest / img_name
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(img_dest), img)


def main():
    parser = argparse.ArgumentParser("Prepare images to be used by iooi")
    parser.add_argument("src", type=Path, help="Path to source folder")
    parser.add_argument(
        "dest", type=Path, help="Path to destination folder (folder must exist already)"
    )
    parser.add_argument("width", type=int, help="Target width of images in pixels")
    parser.add_argument("height", type=int, help="Target height of images in pixels")

    args = parser.parse_args()
    run(
        args.src,
        args.dest,
        args.width,
        args.height,
    )


if __name__ == "__main__":
    main()
