import image_map
import common

import numpy as np
import cv2

strawberries = common.Image("strawberries", cv2.cvtColor(cv2.imread("strawberries.jpg"), cv2.COLOR_BGR2BGRA))  # type: ignore
ocean = common.Image("ocean", cv2.cvtColor(cv2.imread("ocean.jpg"), cv2.COLOR_BGR2BGRA))  # type: ignore

imgs = [strawberries, ocean]

