import pytesseract
import cv2
import re
import mysql.connector
import glob
pytesseract.pytesseract.tesseract_cmd = 'E:/Tesseract-OCR/tesseract.exe'
images = []
for image in glob.glob("C:/Users/Aditya/Desktop/Id_Images/res/*.jpg"):
    images.append(cv2.imread(image,1))
db = mysql.connector.connect(host="localhost",user="aditya",password="root1234",database="company")
for i in images:
    test = pytesseract.image_to_string(i)
    x = re.search("19B[B-E][B-E][0-9]+",test)
    x1 = re.search("HOSTELLER",test)
    cursor = db.cursor()
    cursor.execute("SELECT * FROM student_details")
    res = cursor.fetchall()
    for record in res:
        if record[1]==x.group()  and record[4]==x1.group():
            print("Valid Record")
