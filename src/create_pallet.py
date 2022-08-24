import argparse
import json
from pathlib import Path

import cv2

import common
from image_map import ImageMap
from image_to_color import Mode


def run(src: Path, out: Path, mode: Mode):
    imgs = []

    for img_path in src.iterdir():
        img_name = img_path.name
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        imgs.append(common.Image(img_name, img))

    img_map = ImageMap(imgs, mode)
    serializied = img_map.serialize()
    serializied_json = json.dumps(serializied, indent=4)

    with open(out, "w") as out_f:
        out_f.write(serializied_json)


def main():
    parser = argparse.ArgumentParser("Create pallet out of folder of images")
    parser.add_argument("src", type=Path, help="Path to source folder")
    parser.add_argument("out", type=Path, help="Where to save ImageMap")
    parser.add_argument(
        "-m",
        "--mode",
        nargs="?",
        default=Mode.KMEAN.name,
        type=str,
        choices=[mode.name for mode in list(Mode)],
        help="Algorithm to evaluate images as a color (default: KMEAN)",
    )
    args = parser.parse_args()

    run(args.src, args.out, Mode[args.mode])


if __name__ == "__main__":
    main()
