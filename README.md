# FaceRecognition
The Face Recognition project uses OpenCV in Python to identify faces in images or video using a Kaggle dataset (e.g., LFW)., it employs Haar cascades for face detection and LBPH for recognition.

#Workflow

1.) Collect data of various person
    - Asking multiple people to come in front of webcam , click 20-30 pictires fro each person
    - store the part of the image containing the face (Haarcascade to detect the face)

2.) Train a classifier to learn who is that person(classification)  
    - Load the traing data(.npy arrays)    ### this is acturally not required as we are using KNN approach , i.e unsupervised data
    - store the data and target values(labels)

3.) Predicting the name of the person
    - Read the video stream
    - Extract the face out of it
    - predict the label for that face
         - logistic reg
         - neural networks
         -- KNN(non parametric - look for similkarity in nearest neighbors) ---> we are using this method in our project---very simple
                  - give labels , like 0 for Chanakya , 1 for navya
