# Deepfake Detection System

The idea of deepfake refers to images or videos that are not real and depict evnts that never occured. These deepfakes are created using deep neural network. 

The develpoment in the field of deepfakes are both astonishing and concerning. Nowadays, the danger of false news are generally recognized. More than 1 billion hours of video footage that are consumed in social media per day. So, spread of falsified video is concerning more than ever before. In inappropiate hands, these tools can be used to disseminate misinformation and can undermine public trust.   

This project brings together the most recent research on deepfake recognition in order to improve digital forensic and assist overcoming the shortcomings of the present approach. 


## Requirement

- Python 3.7.3

Some of the basic libraries that is been used in the project: 
- numpy 1.21.5 
- pandas 1.4.2
- tensorflow 2.9.2
- keras 2.9.0
- sklearn 1.0.2
- matplotlib 3.5.1
- seaborn 0.11.2
- flask 1.1.2
- werkzeug 2.0.3

## Data Source

Deepfake dataset was produced with different approach instead of creating deepfake from the beginning which would constrain the diversity and amount of the false data that is then fed to the MesoNet. It is chosen to extract face image from existing false clips. The deepfake dataset was developed by pulling over 175 existing films from well-known deepfake platforms. It is being clarified that particular frames containing faces from deepfake films are being extracted. Similar process is being used to extract real picture data from real film clip source like movies and television shows. It is also explained that the data is being stratified, so that the various angles of the face and levels of resolution were distributed equally throughout the real and deepfake dataset. Deepfake dataset is consisting 7104 images which belongs to 2 classes.
## Flow Chart

![Flow Chart](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/c71bcd3c-c122-43af-8f97-f6858111725f)

## Model Architecture

Meso-4 network is a convolutional neural network with 4 convolutional block. Each block consist of a convolutional layer with batch normalization and a max pooling. Meso-4 works on mesoscopic feature of an image. And the 4 Convulational layers are then connected to fully connected layer for prediction.

![Meso-4](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/b5feacb8-2313-4f73-936a-768574d3ef2f)

- Total parameters

![parameters](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/13311b56-3f01-43cc-8b8d-0768c3a13969)

## Confusion Matrix

Confusion matrix is a tabular visualisation of hte ground-truth labels versus model predictions. Each row of matrix represents the instances in a predicted class and each column represents the instances in an actual class. 

| True Positive (TP) | True Negative (TN) | False Positive (FP) | False Negative (FN) |
| :-------- | :-------- | :-------- | :-------- |
| 3745 | 2564 | 281 | 514 |

![Confusion Matrix](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/baeecd44-6d1a-4d52-89b6-3b795247fabf)

## Performance Metrics

Performance metrices are a part of every machine learning pipeline. They tell whether the model is making any progress or not.  Metrics are used to monitor and measure the performance of a model (during training and testing), and don't need to be differentiable. 

- precision

```
precision = True Positive / (True Positive + False Positive)

```

- Recall / Sensitivity / Hit-Rate

```
Recall = True Positive / (True Positive + False Negative)

```

- F1 Score

```
F1 Score = 2 X Precision X Recall / (Precision + Recall)

```
| Precision | Recall | F1 Score |
| :-------- | :-------- | :-------- |
| 0.93 | 0.87 | 0.90 |

## Area under Curve (AUC)

Better known as Area under Receiver operating characteristics curve (AUROC) is a graph between True Positive Rate also known as Recall & False Positive Rate also known as Fallout

![AUC](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/07e7ab3e-1ffa-4162-b08f-f0ff08a6e3a9)

## Accuracy

The simplest metric to use and implement and is defined as the number of correct predictions divided by the total number of predictions, multiplied by 100.

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

```

- Model Accuracy and Loss percentage

| Model | Train Accuracy | Validation Accuracy | Training loss | Validation loss | Accuracy Average |
| :-------- | :-------- | :-------- | :-------- | :-------- | :-------- |
| Meso 4 | 96.21 | 83.95 | 07.89 | 15.66 | 88.8 |

- Train Accuracy VS Validation Accuracy

![Accuracy](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/180bd55b-8613-4389-bf2e-13af7a451ff3)

- Train loss VS Validation loss
  
![Loss](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/9945e2d4-fff4-47a0-bfc4-88a733b96312)

## Web Application

![Home Page](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/5127f6a7-6dfa-4b77-9d25-85577ef30164)

![Upload](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/97f6a6d9-1cad-4818-ba0e-db54c861e1c2)

![Result](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/2a983e59-126b-47f9-a4ba-c025079f012e)

[Download Paper](https://github.com/DragnaRR/Deepfake-Detection-System/blob/main/paper.pdf)



 
