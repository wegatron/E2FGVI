import cv2
import numpy as np
from paddleocr import PaddleOCR

def seg(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 200])
    upper = np.array([150, 15, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask

class MaskDetect:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")  # need to run only once to download and load model into memory
        self.kernel = np.ones((8, 8), np.uint8)         

    def mask(self, images):
        final_masks = np.empty((len(images), images[0].shape[0], images[0].shape[1], images[0].shape[2]), dtype=bool)
        index = 0
        for image in images:
            result = self.ocr.ocr(image, cls=True)
            text_area_mask = np.zeros_like(image)
            for idx in range(len(result)):
                res = result[idx]
                boxes = np.array([line[0] for line in res]).astype(np.int32)
                text_area_mask = cv2.fillPoly(text_area_mask, boxes, color=(255,255,255))
            text_area_mask = cv2.dilate(text_area_mask, self.kernel)
            mask = seg(image)
            final_masks[index, ...] = (text_area_mask == 255) & (mask == 255)
            index = index + 1
        return final_masks