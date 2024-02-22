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
    def __init__(self, inpaint_board = False):
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False, use_gpu=True)  # need to run only once to download and load model into memory
        if inpaint_board:
            self.kernel = np.ones((30, 30), np.uint8)
            self.text_area_expand = 180
        else:
            print('!!!!!!!!!!!!!!!!!!!!ZZZZ')
            self.kernel = np.ones((13, 13), np.uint8)
            self.text_area_expand = 8            

    def mask(self, images):
        final_masks = np.empty((len(images), images[0].shape[0], images[0].shape[1], images[0].shape[2]), dtype=bool)
        index = 0
        for image in images:
            result = self.ocr.ocr(image, cls=True)
            text_area_mask = np.zeros_like(image)
            for idx in range(len(result)):
                res = result[idx]
                if res == None or len(res) == 0:
                    continue
                boxes = np.array([line[0] for line in res]).astype(np.int32)
                boxes[:, 0, :] = boxes[:, 0, :] - self.text_area_expand
                
                boxes[:, 1, 0] = boxes[:, 1, 0] + self.text_area_expand
                boxes[:, 1, 1] = boxes[:, 1, 1] - self.text_area_expand
                
                boxes[:, 2, :] = boxes[:, 2, :] + self.text_area_expand

                boxes[:, 3, 0] = boxes[:, 3, 0] - self.text_area_expand
                boxes[:, 3, 1] = boxes[:, 3, 1] + self.text_area_expand
                for l in range(boxes.shape[0]): # opencv have a bug!!!! fill multiple polygons
                    text_area_mask = cv2.fillPoly(text_area_mask, [boxes[l]], color=(255,255,255))
            #text_area_mask = cv2.dilate(text_area_mask, self.kernel)
            mask = seg(image)
            mask = cv2.dilate(mask, self.kernel)
            final_masks[index, ...] = (text_area_mask == 255) & (mask == 255)
            #final_masks[index, ...] = mask == 255
            # cv2.imwrite('masked.png', ((mask==255).astype('float') * images[0]).astype('uint8'))
            # cv2.imwrite('masked.png', (final_masks[0, ...].astype('float') * images[0]).astype('uint8'))
            index = index + 1
        return final_masks