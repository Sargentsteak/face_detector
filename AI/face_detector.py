import cv2
from random import randrange

#Load some pre-trained data on face frontals from opencv( haar cascade algorithm)
trained_face_data= cv2.CascadeClassifier('C:\AI\haarcascade_frontalface_default.xml')
#choose an image 
img = cv2.imread('rdj.jpeg')

# #Must covert to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# #Detect faces
# #face classifier detects all the faces with multiscale thing
# #Like if the image gets smaller or bigger it doesnt matter it'll autoscale it
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

#Draw rectangles around the faces
for (x,y,w,h) in face_coordinates:
 cv2.rectangle(img, (x,y) , (x+w,y+h) , (0,255,0) , 1 )
 print(face_coordinates)



cv2.imshow('Face detector', img)
cv2.waitKey()

print("Code completed")

