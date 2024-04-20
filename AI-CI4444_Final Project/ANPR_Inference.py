import os
import sys
import copy
import glob
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import torch
import numpy as np
import pandas as pd

from textRecognition import predict_np
from ultralytics import YOLO


class ANPR_Inference:
    def __init__(self, model_path="anpr_truck_07092023.pt", model_confidence=0.5, iou_threshold=0.5, boxes=True, classes_to_predict=None):
        self.boxes = boxes
        self.model_path = model_path
        self.iou_threshold = iou_threshold
        self.model_confidence = model_confidence
        self.classes_to_predict = classes_to_predict
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            self.names = self.model.names
            self.colors_list = self.create_colors(self.names)
        else:
            print("Model not found")
            sys.exit()

    def create_colors(self, names):
        return [tuple(np.random.randint(low=10, high=256, size=3, dtype=np.dtype(int))) for i in range(len(names))]
        

    @staticmethod
    def draw_rectangle(image, class_name, x1, y1, x2, y2, color):
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=tuple(color), thickness=2)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x1)+(int(x2) - int(x1)), int(y1) - 20), color=(255,255,255), thickness=-1)
        cv2.putText(image, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return image

    
    def run(self, frame, filename):
        output = []
        number_plate_output = []
        original_image = copy.deepcopy(frame)
        count_of_vehicles = 0
        results = self.model.predict(frame, conf=self.model_confidence, iou=self.iou_threshold, boxes=self.boxes, verbose=False)
        # print(dir(results))
        pred = results[0].boxes.data
        # print(type(pred), )
        if pred.shape[1] > 6:
            for i, det in enumerate(pred.cpu()):
                x1, y1, x2, y2 = int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy())
                area = (x2-x1)*(y2-y1)
                
                if len(det) and self.model.names[int(det[6].numpy())] in self.classes_to_predict and int(area) > 3000 :
                    class_name = self.model.names[int(det[6].numpy())] +"_ID:"+ str(int(det[5].numpy()))
                    original_image = self.draw_rectangle(original_image, class_name, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), self.colors_list[int(det[6].numpy())])
                    output.append((filename, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), round(float(det[4].numpy()), 2), self.model.names[int(det[6].numpy())]))


                if self.model.names[int(det[6].numpy())] == "np":
                    np_image = original_image[int(det[1].numpy()): int(det[3].numpy()), int(det[0].numpy()):int(det[2].numpy())]
                    np_image_cls, output_str = predict_np(np_image)
                    class_name = self.model.names[int(det[6].numpy())] +"_ID:"+ str(int(det[5].numpy())) +"_"+ output_str
                    original_image = self.draw_rectangle(original_image, class_name, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), self.colors_list[int(det[6].numpy())])
                    number_plate_output.append((filename, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), round(float(det[4].numpy()), 2), self.model.names[int(det[6].numpy())], output_str))
            
            return output, number_plate_output, original_image
        else:
            for i, det in enumerate(pred.cpu()):
                x1, y1, x2, y2 = int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy())
                area = (x2-x1)*(y2-y1)
                if len(det) and self.model.names[int(det[5].numpy())] in self.classes_to_predict:
                    count_of_vehicles+=1
                    class_name = self.model.names[int(det[5].numpy())]
                    original_image = self.draw_rectangle(original_image, class_name, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), self.colors_list[int(det[5].numpy())])
                    output.append((filename, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), round(float(det[4].numpy()), 2), self.model.names[int(det[5].numpy())]))
            

                if self.model.names[int(det[5].numpy())] == "np":
                    np_image = original_image[int(det[1].numpy()): int(det[3].numpy()), int(det[0].numpy()):int(det[2].numpy())]
                    
                    output_str = predict_np(np_image)
                    class_name = self.model.names[int(det[5].numpy())] +"_"+ output_str
                    original_image = self.draw_rectangle(original_image, class_name, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), self.colors_list[int(det[5].numpy())])
                    number_plate_output.append((filename, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), round(float(det[4].numpy()), 2), self.model.names[int(det[5].numpy())], output_str))
            # print(f"COUNT OF VEHICLES:{count_of_vehicles}")
            # original_image = cv2.putText(original_image, f"COUNT OF VEHICLES:{count_of_vehicles}", (int(100), int(60)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            return output, number_plate_output, original_image





ANPR_Inference = ANPR_Inference(classes_to_predict=["auto", "bus", "car", "motorcycle", "hmv", "truck", "tractor"])


#the below code will execute to test out the images that are present in test_data
'''images_path = glob.glob("test_data/*.jpg")
for image_name in images_path:
    frame = cv2.imread(image_name)
    print(image_name)
    if frame is not None:
        anpr_output, number_plate_output, output_image =ANPR_Inference.run(frame, None)
        print(f" anpr_output:{anpr_output},number_plate_output:{number_plate_output}")
        cv2.imshow("image", cv2.resize(output_image, (640, 640)))
        # out.write(output_image)
        key = cv2.waitKey()
        if key == 27:
            break'''

#The below code is to test out code against user given input image

#image_url = 'https://media.geeksforgeeks.org/wp-content/uploads/20200326001003/blurred.jpg'
#image_url='https://media.interaksyon.com/wp-content/uploads/2023/02/Toyota-Avanza.jpg'
#image_url='https://images.squarespace-cdn.com/content/v1/5c981f3d0fb4450001fdde5d/1563727260863-E9JQC4UVO8IYCE6P19BO/cars+1.jpg?format=2500w'
image_url='https://stage-drupal.car.co.uk/s3fs-public/styles/original_size/public/2019-09/whose-number-plate-is-this.jpg?sayKdAGCZ0.9aBppcqXq0dpRU5LG.vZO&itok=7xYZbpK6'

response = requests.get(image_url)
#print("Status code:", response.status_code)

if response.status_code == 200:
    # Print the response content
    #print("Response content:", response.content)
    image_array = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if img is not None:
        anpr_output, number_plate_output, output_image = ANPR_Inference.run(img, None)
        print(f" anpr_output:{anpr_output},number_plate_output:{number_plate_output}")
        cv2.imshow("image", cv2.resize(output_image, (640, 640)))
        key = cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        print("Failed to decode the image. Please check the URL.")
else:
    print("Failed to fetch the image. Please check the URL.")
# # out.release()
# cap.release()
# cv2.destroyAllWindows()



