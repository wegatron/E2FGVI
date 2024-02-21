from text_det import MaskDetect
import sys
import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture(sys.argv[1])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
    index = 0
    mask_det = MaskDetect()
    while True:
        # Read a frame from the video
        ret, frame = cap.read()        
        # Check if the frame was successfully read
        if not ret:
            # If the frame couldn't be read, it means we reached the end of the video
            print("End of video.")
            break
        final_masks = mask_det.mask([frame])
        cv2.imwrite(f'{sys.argv[2]}mask{index}.png', final_masks[0].astype('uint8')*255)
        index = index + 1
        if index == int(sys.argv[3]):
            break
    #expand bbox
    # bbox[0] = max(0, bbox[0]-40)
    # bbox[1] = max(0, bbox[1]-40)
    # bbox[2] = min(bbox[2], width)
    # bbox[3] = min(bbox[3], height)
    #np.savetxt(f'{sys.argv[2][:-1]}_bbox.txt', bbox)
#img = cv2.imread('/home/wegatron/win-data/data/subtile_poj/input_images/in001.png')
#final_mask = text_mask(img)
