import copy
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

img_size = (112,32)
#class_name = {0:'bad',1:'good'}

def predict_np(img):
    np_img = copy.deepcopy(img)
    result = ocr.ocr(np_img, cls=True)
  
    txts = []
    for idx in range(len(result)): 
        res = result[idx]
        if res is not None:
            txts = [line[1][0] for line in res]
    if len(txts) > 0:
        return "".join(txts)
    else:
        return "None"