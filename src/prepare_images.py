import cv2
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser("Prepare images to be used by iooi")
    parser.add_argument("from", nargs=1, type=Path, required=True, help="Path to source folder")
    parser.add_argument("to", nargs=1, type=Path, required=True, help="Path of destination folder")
    parser.add_argument("width", nargs=1, type=int, required=True, help="Width of images in pixels")
    parser.add_argument("height", nargs=1, type=int, required=True, help="Height of images in pixels")

    args = parser.parse_args()
    for img_path in args.from:  # type: ignore
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # type: ignore
        img = cv2.resize(img, (args.width, args.height), interpolation= cv2.INTER_NEAREST)  # type: ignore
        dest = args.to / img_path.name
        cv2.imwrite(dest, img)  # type: ignore



if __name__ == "__main__":
    main()
