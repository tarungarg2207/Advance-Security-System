import cv2
import os
import numpy as np
import faceRecognition as fr

import xlrd

#This module takes images  stored in diskand performs face recognition
test_img=cv2.imread('TestImages/a.jpg')#test_img path
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)


#Comment belows lines when running this program second time.Since it saves training.yml file in directory
#faces,faceID=fr.labels_for_training_data('/home/rahul/FaceRecognition-master/trainingImages')
#face_recognizer=fr.train_classifier(faces,faceID)
#face_recognizer.save('trainingData.yml')
#face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer = cv2.face_LBPHFaceRecognizer().create()

#face_recognizer = cv2.createLBPHFaceRecognizer()

#Uncomment below line for subsequent runs
face_recognizer.read('trainingData.yml')#use this to load training data for subsequent runs

wb = xlrd.open_workbook("db.xls")
xl_sheet = wb.sheet_by_index(0)
name = dict()
for i in range(xl_sheet.nrows):
    name[int(xl_sheet.cell(i, 0).value)] = xl_sheet.cell(i, 1).value

#name={0:"a",1:"b",2:"c",3:"d",4:"e",5:"f",6:"g",7:"h"}#creating dictionary containing names for each label

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(300,700))
cv2.imshow("face dtecetion tutorial",resized_img)
cv2.waitKey(0)#Waits indefinitely until a key is pressed
cv2.destroyAllWindows()
