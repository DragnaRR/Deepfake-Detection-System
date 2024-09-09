# Deepfake Detection System

The idea of deepfake refers to images or videos that are not real and depict events that never occured. These deepfakes are created using deep neural network. 

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

Confusion matrix is a tabular visualisation of hte ground-truth labels versus model predictions. It's a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. Each row of matrix represents the instances in a predicted class and each column represents the instances in an actual class. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.

Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-

**True Positives (TP)** – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.

**True Negatives (TN)** – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.

**False Positives (FP)** – False Positives occur when we predict an observation belongs to a    certain class but the observation actually does not belong to that class. This type of error is called **Type I error.**

**False Negatives (FN)** – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called **Type II error.**

| True Positive (TP) | True Negative (TN) | False Positive (FP) | False Negative (FN) |
| :-------- | :-------- | :-------- | :-------- |
| 3745 | 2564 | 281 | 514 |

![Confusion Matrix](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/baeecd44-6d1a-4d52-89b6-3b795247fabf)

The confusion matrix shows `3745 + 2564 = 6309 correct predictions` and `281 + 514 = 795 incorrect predictions`.


In this case, we have


- `True Positives` (Actual Positive:1 and Predict Positive:1) - 3745


- `True Negatives` (Actual Negative:0 and Predict Negative:0) - 2564


- `False Positives` (Actual Negative:0 but Predict Positive:1) - 281 `(Type I error)`


- `False Negatives` (Actual Positive:1 but Predict Negative:0) - 514 `(Type II error)`

## Performance Metrics

Performance metrices are a part of every machine learning pipeline. They tell whether the model is making any progress or not.  Metrics are used to monitor and measure the performance of a model (during training and testing), and don't need to be differentiable. 

- precision

Precision can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true and false positives (TP + FP).

So, Precision identifies the proportion of correctly predicted positive outcome. It is more concerned with the positive class than the negative class.

Mathematically, precision can be defined as the ratio of TP to (TP + FP).

```
precision = True Positive / (True Positive + False Positive)

```

- Recall / True Positive Rate / Sensitivity / Hit-Rate

Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true positives and false negatives (TP + FN). Recall is also called Sensitivity.

Recall identifies the proportion of correctly predicted actual positives.

Mathematically, recall can be defined as the ratio of TP to (TP + FN).

```
Recall = True Positive / (True Positive + False Negative)

```

- F1 Score

f1-score is the weighted harmonic mean of precision and recall. The best possible f1-score would be 1.0 and the worst would be 0.0. f1-score is the harmonic mean of precision and recall. So, f1-score is always lower than accuracy measures as they embed precision and recall into their computation. The weighted average of f1-score should be used to compare classifier models, not global accuracy.

```
F1 Score = 2 X Precision X Recall / (Precision + Recall)

```
| Precision | Recall | F1 Score |
| :-------- | :-------- | :-------- |
| 0.93 | 0.87 | 0.90 |

## Area under Curve (AUC)

Better known as Area under Receiver operating characteristics curve (AUROC) is a graph between True Positive Rate also known as Recall & False Positive Rate also known as Fallout
ROC AUC is a single number summary of classifier performance. The higher the value, the better the classifier. A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5.
Another tool to measure the classification model performance visually is ROC Curve. ROC Curve stands for Receiver Operating Characteristic Curve. An ROC Curve is a plot which shows the performance of a classification model at various classification threshold levels.

The ROC Curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold levels.

True Positive Rate (TPR) is also called Recall. It is defined as the ratio of TP to (TP + FN).

False Positive Rate (FPR) is defined as the ratio of FP to (FP + TN).

In the ROC Curve, we will focus on the TPR (True Positive Rate) and FPR (False Positive Rate) of a single point. This will give us the general performance of the ROC curve which consists of the TPR and FPR at various threshold levels. So, an ROC Curve plots TPR vs FPR at different classification threshold levels. If we lower the threshold levels, it may result in more items being classified as positve. It will increase both True Positives (TP) and False Positives (FP).

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

[Download Paper](https://github.com/DragnaRR/Deepfake-Detection-System/files/12874285/paper.pdf)




 
