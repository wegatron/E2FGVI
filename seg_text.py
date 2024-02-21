import cv2
import numpy as np
from paddleocr import PaddleOCR
import sys

def seg(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 200])
    upper = np.array([150, 15, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def detect(image, bbox):
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(use_angle_cls=True, lang="en")  # need to run only once to download and load model into memory
    result = ocr.ocr(image, cls=True)
    text_area_mask = np.zeros_like(image)
    for idx in range(len(result)):
        res = result[idx]
        boxes = np.array([line[0] for line in res]).astype(np.int32)        
        bbox[0] = min(bbox[0], boxes[:,:, 0].min())
        bbox[1] = min(bbox[1], boxes[:,:, 1].min())
        bbox[2] = max(bbox[2], boxes[:,:, 0].max())
        bbox[3] = max(bbox[3], boxes[:,:, 1].max())
        # print('---')
        # print(boxes)
        # print('---')
        text_area_mask = cv2.fillPoly(text_area_mask, boxes, color=(255,255,255))
    kernel = np.ones((8, 8), np.uint8) 
    final_mask = cv2.dilate(final_mask, kernel)    
    return text_area_mask

def text_mask(image, bbox):
    text_area_mask = detect(image, bbox)
    mask = seg(image)
    final_mask = (text_area_mask == 255) & (mask == 255)
    final_mask = final_mask.astype(np.uint8) * 255
    return final_mask

if __name__ == '__main__':
    cap = cv2.VideoCapture(sys.argv[1])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
    index = 0
    bbox = np.array([1000000, 1000000, 0, 0], dtype=np.int32)
    while True:
        # Read a frame from the video
        ret, frame = cap.read()        
        # Check if the frame was successfully read
        if not ret:
            # If the frame couldn't be read, it means we reached the end of the video
            print("End of video.")
            break
        final_mask = text_mask(frame, bbox)
        cv2.imwrite(f'{sys.argv[2]}mask{index}.png', final_mask)
        index = index + 1
        if index == int(sys.argv[3]):
            break
    #expand bbox
    bbox[0] = max(0, bbox[0]-40)
    bbox[1] = max(0, bbox[1]-40)
    bbox[2] = min(bbox[2], width)
    bbox[3] = min(bbox[3], height)
    #np.savetxt(f'{sys.argv[2][:-1]}_bbox.txt', bbox)
#img = cv2.imread('/home/wegatron/win-data/data/subtile_poj/input_images/in001.png')
#final_mask = text_mask(img)
