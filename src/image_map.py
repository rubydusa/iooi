from typing import Optional
import numpy as np
import numpy.typing as npt
import cv2

import common

# map of all bgr colors to a set of images
class ImageMap:
    _imgs: list[common.Image]
    _colors: npt.NDArray

    # imgs is bgra images
    def __init__(self, imgs: Optional[list[common.Image]]):
        if imgs is None:
            return ImageMap(None)

        imgs_contents = [i.img for i in imgs]
        
        imgs_amount = len(imgs_contents)
        imgs_colors = np.array([common.img_to_color(i) for i in imgs_contents])
        imgs_colors_lab_img = imgs_colors.reshape((1, imgs_amount, 3))
        imgs_colors_lab_img = cv2.cvtColor(imgs_colors_lab_img, cv2.COLOR_BGR2LAB)  # type: ignore
        imgs_colors_lab_img = imgs_colors_lab_img.reshape((imgs_amount, 3))

        colors = common.all_colors()
        colors_lab_img = colors.reshape((4096, 4096, 3))
        colors_lab_img = cv2.cvtColor(colors_lab_img, cv2.COLOR_BGR2LAB)  # type: ignore
        distances = np.array([common.color_distance(imgs_colors_lab_img[i], colors_lab_img) for i in range(imgs_amount)])
        
        colors = np.argmin(distances, axis=0)
        colors = colors.reshape((256, 256, 256))

        self._imgs = imgs
        self._colors = colors

    def __getitem__(self, item: tuple[int]) -> common.Image:
        return self._imgs[self._colors[item]]

    def serialize(self) -> list[str]:
        return [self._imgs[color_index].name for color_index in self._colors.reshape((2**24))]

    @staticmethod
    def deserialize(data: list[str]) -> "ImageMap":
        indices = dict()
        next_index = 0

        imgs = []
        colors = []
        for img_name in data:
            if img_name not in indices:
                imgs.append(common.Image(img_name, np.array([])))
                indices[img_name] = next_index
                next_index += 1

            index = indices[img_name]
            colors.append(index)

        colors = np.array(colors)

        image_map = ImageMap(None)
        image_map.__set_imgs(imgs)
        image_map.__set_colors(colors)

        return image_map

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


def main():
    pass

if __name__ == "__main__":
    main()
