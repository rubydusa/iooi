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
    out: Optional[Path] = None,
    override: bool = False,
):
    imgs = []

    for img_path in src.iterdir():
        img_name = img_path.name
        img_dest = dest / img_name
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        imgs.append(common.Image(img_name, img))

        if not override:
            if img_dest.exists():
                continue

        cv2.imwrite(str(img_dest), img)

    if out:
        img_map = ImageMap(imgs)
        serializied = img_map.serialize()
        serializied_json = json.dumps(serializied, indent=4)

        with open(out, "w") as out_f:
            out_f.write(serializied_json)


def main():
    parser = argparse.ArgumentParser("Prepare images to be used by iooi")
    parser.add_argument("src", type=Path, help="Path to source folder")
    parser.add_argument("dest", type=Path, help="Path to destination folder")
    parser.add_argument("width", type=int, help="Width of images in pixels")
    parser.add_argument("height", type=int, help="Height of images in pixels")
    parser.add_argument(
        "-o",
        "--out",
        nargs="?",
        default=None,
        type=Path,
        help="Where to save the ImageMap (default: None) will not generate ImageMap if argument not present",
    )
    parser.add_argument(
        "--override",
        nargs="?",
        default=False,
        type=bool,
        help="Whether to override existing files in destination folder (default: False)",
    )

    args = parser.parse_args()
    run(
        args.src,
        args.dest,
        args.width,
        args.height,
        out=args.out,
        override=args.override,
    )


if __name__ == "__main__":
    main()
