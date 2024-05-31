# Person and PPE (Personal protective equipment) detection using Yolov8 object detection model
Object detection of single or multiple objects on Google colab. Since Colab provides free GPUs such as Nvidia Tesla T4, we can instantly use these resources for training and testing the deep learning models. Here are the steps for implementing this model for detecting person and PPE.

1. The custom dataset of person wearing PPE has 416 images with labels in PascalVOC format

2. The labeled data is converted into .txt files by saving class indices and anchor boxes according to dataset using ```PASCALVOC_to_yolo.py```file so that it can be trained using Yolov8.

3. Divide the dataset into train, valid, test format with the ratio 70%-20%-10% resepectively.

4. Zip the prepared data and upload it to Google Colab

5. Load the datset and train the Person and PPE models separately

6. Inference models using ```inference.py``` with ```person_det_best.pt``` and ```PPE_det_best.pt``` as the final step

### Step1:
Convert the xml files to yolo format by only selecting class name tag "person"" in ```PASCALVOC_to_yolo.py``` for person detection model. Repeat the same step for PPE detection model but omit```person``` class name tag and save the rest of the class names. Due to class imbalance, 'glasses', 'ear=protector' and 'safety-harness' classes are dropped during PPE model training. 

### Step2:
Both weights ```person_det_best.pt``` and ```PPE_det_best.pt``` are saved in weights folders ```person_detection/datasets/weights``` and ``PPE_detection/datasets/weights```. Both are trained for 70 and 80 epochs respectively.

### Step3:
Evaluate both models on images present in valid folder. 
The validation accuracy of person model is mAP=0.857 and 0.587 for mAP(50-95) with F1 score= 0.802
Validation accuracy of PPE model for all classes is mAP= 0.494 and 0.316 for mAP(50-95) with F1 score= 0.50

### Step4:
Preform inference on images in test folders using ```inference.py```
