import json
import cv2
import argparse
from typing import Optional
from pathlib import Path

import image_map
import common

def run(src: Path, dest: Path, width: int, height: int, out: Optional[Path] = None, override: bool = False):
    imgs = []
    for _, img_path in zip(range(10), src.iterdir()):
        img_name = img_path.name
        img_dest = dest / img_name
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGBA) 
        img = cv2.resize(img, (width, height), interpolation= cv2.INTER_NEAREST)
        imgs.append(common.Image(img_name, img))

        if not override:
            if img_dest.exists():
                continue

        cv2.imwrite(str(img_dest), img)

   
    if out:
        img_map = image_map.ImageMap(imgs)
        serializied = img_map.serialize()
        serializied_json = json.dumps(serializied, indent=4)

        with open(out, "w") as out_f:
            out_f.write(serializied_json)

def main():
    parser = argparse.ArgumentParser("Prepare images to be used by iooi")
    parser.add_argument("src", nargs=1, type=Path, help="Path of source folder")
    parser.add_argument("dest", nargs=1, type=Path, help="Path of destination folder")
    parser.add_argument("width", nargs=1, type=int, help="Width of images in pixels")
    parser.add_argument("height", nargs=1, type=int, help="Height of images in pixels")
    parser.add_argument("--out", nargs="?", default=None, type=Path, help="Where to save the ImageMap (default: None) will not generate ImageMap if argument not present")
    parser.add_argument("--override", nargs="?", default=False, type=bool, help="Whether to override existing files in destination folder (default: False)")

    args = parser.parse_args()
    run(args.src[0], args.dest[0], args.width[0], args.height[0], out=args.out, override=args.override)

if __name__ == "__main__":
    main()
