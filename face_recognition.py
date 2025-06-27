import cv2 as cv
import numpy as np
#creating a camera object
cam = cv.VideoCapture(0)

#Ask the name of the candidate
filename = input("please enter your name for verification of your face")
dataset_path = "./dataa/"
offset = 20 # extra pixels for cropped face
faceData = [] # list to save face data
skip = 0


model = cv.CascadeClassifier("haar_face.xml")

while True :
    isTrue , img = cam.read()
    if not isTrue:
        print("reading the cam failed")
    #store the gray image
    grayImg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(img , scaleFactor=1.3 ,minNeighbors=5)

    #out of all the faces in a cam, pick the largest face bounded by the rectangle
    faces = sorted(faces,key = lambda f :f[2] * f[3])  # f[2] and f[3] is w and h and product gives the area
    if len(faces)>0: # bcz if there are no faces then program may crash
        f = faces[-1] #last face is the largest face with moer area
        
        x,y,w,h = f
        cv.rectangle(img ,(x,y),(x+w ,y+h),(0,255,0),thickness=2)

        #crop and save the largest face
        cropped_face = img[y-offset : y+h+offset , x-offset  : x+w+offset]
        # we have to resize any pic to standard value
        cropped_face = cv.resize(cropped_face,(100,100))
        skip+=1
        if skip %10 ==0 : # for every 10 frames save the face data(image)
            faceData.append(cropped_face)
            print("saved so far" + ' '+ str(len(faceData)) )


    cv.imshow("image window ",img)
    #cv.imshow("cropped face",cropped_face)
    key = cv.waitKey(1)
    if key == ord('d'):
        break

# write the faceData om the disk
faceData = np.asarray(faceData)
m = faceData.shape[0] # no of pixels
faceData = faceData.reshape((m,-1)) # (20 , 100 *100 * 3) second terms-1 represents the product of those values
print(faceData.shape)

# save on the disk as numpy array
filepath = dataset_path + filename + ".npy"
np.save(filepath,faceData)
print("data saved succesfully"+filepath)



#release the camera, and destroy the window
cam.release()
cv.destroyAllWindows()

