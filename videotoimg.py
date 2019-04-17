import cv2
import os
from xlrd import open_workbook
import xlwt
from xlutils.copy import copy


cap=cv2.VideoCapture(0)

count = 0

while True:
    usr_id = input("Enter your user id")
    des = os.path.join('trainingImages',usr_id)
    if not os.path.exists(des):
        break
    print('User Id already Exist')
    
usr_name = input("Enter your username.")

rb = open_workbook("db.xls")
rsheet = rb.sheet_by_index(0)
wb = copy(rb)
sheet = wb.get_sheet(0)
sheet.write(rsheet.nrows,0,usr_id)
sheet.write(rsheet.nrows,1,usr_name)
wb.save("db.xls")

os.makedirs(des)

while True:
    ret,test_img=cap.read()
    frame = 'frame'+str(count)+".jpg"
    if not ret :
        continue
    cv2.imwrite(os.path.join(des,frame),test_img)     # save frame as JPG file
    count += 1
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('face detection Tutorial ',resized_img)
    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break


cap.release()
cv2.destroyAllWindows
