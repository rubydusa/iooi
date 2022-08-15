from pathlib import Path
import argparse
import numpy as np
import numpy.typing as npt

import common
import image_map

def run():
    pass

def main():
    imgs = []
    for _, img in zip(range(2), Path("src/imgs").iterdir()):
        img = common.Image(img.name, cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB))
        imgs.append(img)

    img_map = image_map.ImageMap(imgs)

if __name__ == "__main__":
    main()
