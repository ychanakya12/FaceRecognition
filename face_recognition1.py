import cv2 as cv
import numpy as np
import os

#Data 
dataset_path = "./dataa/"
faceData = [] # x value stored here
labels = [] # y value stored here
classId = 0
nameMap = {}

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):

        nameMap[classId] = f[:-4]
        #X values
        dataItem = np.load(dataset_path + f)
        m = dataItem.shape[0] #no of images
        faceData.append(dataItem) 

        # Y values
        target = classId * np.ones((m,))
        classId += 1
        labels.append(target)

Xt = np.concatenate(faceData , axis=0)
yt = np.concatenate(labels,axis=0).reshape((-1,1))


print(Xt.shape)
print(yt.shape)
print(nameMap)

# Algorithm 
def dist(p,q):
    return np.sqrt(np.sum((p-q)**2))

def knn(X,y,xt, k =5):  # k is the hyper parameter , results will change acc to this k value
    m = X.shape[0]
    dlist = []

    for i in range(m):
        d = dist(X[i],xt)
        dlist.append((d,y[i][0]))

    dlist = sorted(dlist)
    dlist = dlist[:k]
    dlist = np.array(dlist)
    labels = dlist[:,1] # till here logic for classifcation and regresiion remains same

    labels , counts = np.unique(labels, return_counts = True) # count of each category and the category
    idx = counts.argmax()
    pred = labels[idx]

    return int(pred)

# predictions

#creating a camera object
cam = cv.VideoCapture(0)

model = cv.CascadeClassifier("haar_face.xml")

offset = 20
while True :
    isTrue , img = cam.read()
    if not isTrue:
        print("reading the cam failed")
    faces = model.detectMultiScale(img , scaleFactor=1.3 ,minNeighbors=5)

    for f in faces:
        x,y,w,h = f
        print(f)
        #crop and save the largest face
        cropped_face = img[y-offset : y+h+offset , x-offset  : x+w+offset]
        # we have to resize any pic to standard value
        cropped_face = cv.resize(cropped_face,(100,100))

        #predict the name using knn
        classPredicted = knn(Xt,yt,cropped_face.flatten()) # to change it to x, y f0rm 
        #name
        namePredicted = nameMap[classPredicted]
        print(namePredicted)

        #display namd and box
        cv.putText(img ,namePredicted,(x,y-10),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv.LINE_AA)
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv.imshow("Prediction window",img)

    key = cv.waitKey(1)
    if key == ord('d'):
        break
cam.release()
cv.destroyAllWindows()