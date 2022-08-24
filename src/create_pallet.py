import argparse
import json
from pathlib import Path

import cv2

import common
from image_map import ImageMap


def run(src: Path, out: Path):
    imgs = []

    for img_path in src.iterdir():
        img_name = img_path.name
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        imgs.append(common.Image(img_name, img))

    img_map = ImageMap(imgs)
    serializied = img_map.serialize()
    serializied_json = json.dumps(serializied, indent=4)

    with open(out, "w") as out_f:
        out_f.write(serializied_json)


def main():
    parser = argparse.ArgumentParser("Create pallet out of folder of images")
    parser.add_argument("src", type=Path, help="Path to source folder")
    parser.add_argument("out", type=Path, help="Where to save ImageMap")
    args = parser.parse_args()

    run(args.src, args.out)


if __name__ == "__main__":
    main()
