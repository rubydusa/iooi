import cv2
from pathlib import Path
from typing import Optional
import numpy as np
import numpy.typing as npt

import common

# map of all rgb colors to a set of images
class ImageMap:
    _imgs: list[common.Image]
    _colors: npt.NDArray  # 1D array size 2**24 where each value is the index of the corresponding hex value

    def __initialize(self, imgs: list[common.Image]):
        imgs_contents = [i.img for i in imgs]
        
        imgs_amount = len(imgs_contents)
        imgs_colors = np.array([common.img_to_color(i) for i in imgs_contents])

        imgs_colors_lab_img = imgs_colors.reshape((1, imgs_amount, 3))
        imgs_colors_lab_img = cv2.cvtColor(imgs_colors_lab_img, cv2.COLOR_RGB2LAB)  
        imgs_colors_lab_img = common.float_lab(imgs_colors_lab_img)
        imgs_colors_lab_img = imgs_colors_lab_img.reshape((imgs_amount, 3))

        colors = common.all_colors()
        colors_lab_img = colors.reshape((1, 2**24, 3))
        colors_lab_img = cv2.cvtColor(colors_lab_img, cv2.COLOR_RGB2LAB)  
        colors_lab_img = common.float_lab(colors_lab_img)
        colors_lab_img = colors_lab_img.reshape((2**24, 3))
        distances = np.array([common.color_distance(imgs_colors_lab_img[i], colors_lab_img) for i in range(imgs_amount)])
        
        colors = np.argmin(distances, axis=0)
        colors = colors.reshape((256, 256, 256))

        self._imgs = imgs
        self._colors = colors

    # imgs is rgba images
    def __init__(self, imgs: Optional[list[common.Image]]=None):
        if imgs is None:
            return
        else:
            self.__initialize(imgs)

    def serialize(self) -> dict:
        result = dict()
        result["imgs"] = [i.name for i in self.imgs]
        result["colors"] = self.colors.reshape((2**24)).tolist()

        return result

    @staticmethod
    def deserialize(data: dict) -> "ImageMap":
        new = ImageMap()
        new.__set_imgs([common.Image(name, np.array([])) for name in data["imgs"]])
        new.__set_colors(np.array(data["colors"]).reshape((256, 256, 256)))

        return new

    # use after deserialize to load images contents
    def load(self, src: Optional[Path]=None):
        new_imgs = []
        src = src if src is not None else Path(".")

        for img in self.imgs:
            str_path = str(src / img.name)
            print(str_path)
            content = cv2.cvtColor(cv2.imread(str_path), cv2.COLOR_BGR2RGBA)
            new_imgs.append(common.Image(img.name, content))

        self.__set_imgs(new_imgs)
     
    @property
    def imgs(self) -> list[common.Image]:
        return self._imgs

    @property
    def colors(self) -> npt.NDArray:
        return self._colors

    def __set_imgs(self, imgs: list[common.Image]):
        self._imgs = imgs

    def __set_colors(self, colors: npt.NDArray):
        self._colors = colors

