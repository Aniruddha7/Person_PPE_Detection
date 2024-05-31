from ultralytics import YOLO
import cv2
import numpy as np
import os
#from google.colab.patches import cv2_imshow
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO
import torch
import argparse


def inference(input_path, output_path, person_det, ppe_det):

  person_model= YOLO(model=person_det)
  ppe_model= YOLO(model= ppe_det)
  
  for filename in os.listdir(input_path):
    img = cv2.imread(os.path.join(input_path, filename))
    img2 = img.copy()
    #img2= cv2.imread(os.path.join(input_path, filename))
    #print(img)
    person_results = person_model.predict(img)
    ppe_results = ppe_model.predict(img2)  # return a list of Results objects
    results= [ppe_results, person_results]
    
    for result in person_results:
      annotator = Annotator(img)
      boxes = result.boxes   # Boxes object for bounding box outputs
  
      for bbox in boxes:
          box = bbox.xyxy[0]
          classes = bbox.cls
          confidence= float(bbox.conf)

          # Draw bounding box rectangle
          rect= cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 150, 100), 4)
          #cv2.putText(rect, str("{:.2f}".format(confidence)), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150,50,255), 2)

          # Put text for class label
          person_label= f"{person_model.names[int(classes)]}:{confidence:.2f}"
          cv2.putText(rect, person_label, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

      person_opt= output_path + 'person_detection/' 

      if not os.path.exists(person_opt):
          os.makedirs(person_opt)

      output_file= person_opt + os.path.splitext(filename)[0] + '.jpg'
      img_with_boxes = annotator.result()
      cv2.imwrite(output_file, img_with_boxes)
      
      for result in ppe_results:
        annotator = Annotator(img2)
        boxes = result.boxes   # Boxes object for bounding box outputs
  
        for bbox in boxes:
            box = bbox.xyxy[0]
            classes = bbox.cls
            confidence= float(bbox.conf)

            # Draw bounding box rectangle
            rect2= cv2.rectangle(img2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 150, 100), 4)
            #cv2.putText(rect, str("{:.2f}".format(confidence)), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150,50,255), 2)

            # Put text for class label
            ppe_label= f"{ppe_model.names[int(classes)]}:{confidence:.2f}"
            cv2.putText(rect2, ppe_label, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        ppe_opt= output_path + 'ppe_detection/' 

        if not os.path.exists(ppe_opt):
            os.makedirs(ppe_opt)

        output_file= ppe_opt + os.path.splitext(filename)[0] + '.jpg'
        img_with_boxes = annotator.result()
        cv2.imwrite(output_file, img_with_boxes)

def parse():
    parser= argparse.ArgumentParser(prog= 'Person_ppe_inference',
                                    description= 'Fucntion to predict person and ppe detection in the images and to save in directory' )
    parser.add_argument('input_path', nargs= "+", type= str )
    parser.add_argument('output_path',nargs= "+", type= str )
    parser.add_argument('person_det',nargs= "+", type= str )
    parser.add_argument('ppe_det',nargs= "+", type= str )
    args= parser.parse_args()
    return args
       
def main():
    args= parse()
    inference(args.input_path[0], args.output_path[0], args.person_det[0], args.ppe_det[0])

if __name__== '__main__':
    main()
 
  #cv2_imshow(img_with_boxes)
#inference('d:/PPE_detection/datasets/test/images/', 'd:/person_ppe_inference/yolov8/', 'd:/person_detection/datasets/weights/person_det_best.pt', 'd:/PPE_detection/datasets/weights/PPE_det_best.pt')

#CLI
#python inference.py 'D:/PPE_detection/datasets/test/images/' 'D:/person_ppe_inference/yolov8/' 'D:/person_detection/datasets/weights/person_det_best.pt/' 'D:/PPE_detection/datasets/weights/PPE_det_best.pt' 