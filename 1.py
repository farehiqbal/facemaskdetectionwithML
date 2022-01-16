import cv2

#method to read image in cv2
img = cv2.imread("E:\Onedrive\OneDrive - Higher Education Commission\Desktop\FaceMask Detection\pic.jpeg") 
print(img.shape)  #3rd enry is rgb channel

print(img[0]) # return the values of rgb of first row of the image


#importing module to visulize the image
# visualzing image using plt gives colorless image
import matplotlib.pyplot as plt
print(plt.imshow(img))


#we can also visualize the image using cv3
'''
while True:
    cv2.imshow("image",img)  #opening it gives an error so we will use loop
    if cv2.waitKey(2)==27:   #27 is ASCII key of ESC key, WAITKEY() waits for a key
        break                # infinately untill it is pressed, 2 is the delay in ms here
cv2.destroyAllWindows
'''

'''
Voila jons algorithm

The algorithm has four stages:

Haar Feature Selection
Creating an Integral Image
Adaboost Training
Cascading Classifiers
'''

# Value = Σ (pixels in black area) - Σ (pixels in white area)


haar_data = cv2.CascadeClassifier("data.xml")        #Algorithm data
print(haar_data.detectMultiScale(img))               # Method which detects haar features


while True:
    faces = haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 4)
    cv2.imshow("image",img)  
    if cv2.waitKey(2)==27:   
        break                
cv2.destroyAllWindows


capture =  cv2.VideoCapture(0)
data= []
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50,50))
            print(len(data))
            if len (data) < 400:
                data.append(face)

        cv2.imshow("image",img)  
        if cv2.waitKey(2)==27 or len(data) >=200:
            break

capture.release()
cv2.destroyAllWindows()

import numpy as np

np.save("without_mask.npy", data)

#np.save("with_mask.npy", data)

