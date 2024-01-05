import face_recognition
import os
import cv2 as cv
import numpy as np
import pandas as pd
import time
from datetime import datetime

class Register:
    def __init__(self):
        self.encodingArr = np.load("utilsFiles\\encodingArr.npy")
        self.labelArr = np.load("utilsFiles\\labelArr.npy")
        self.StdData = pd.read_csv("utilsFiles\StdDetails.csv")
        self.columns = self.StdData.columns
        

    def checkPresence(self, inputArr):
        
        for i, arr in enumerate(self.encodingArr):
            
            result = face_recognition.compare_faces(arr, inputArr)
            
            if result[0]:
               
                return (True, i+1)
        

        return (False, -1)

    def addPerson(self, inputArr, Name):
        exist, index = self.checkPresence(inputArr)
       
        if not exist:
            self.encodingArr = np.append(self.encodingArr, inputArr, axis=0)
            self.labelArr = np.append(self.labelArr, [self.labelArr.shape[0]], axis=0)
            
            data = pd.DataFrame({self.columns[0]: [Name], self.columns[1]: [len(self.labelArr)]})
            self.StdData = pd.concat([self.StdData, data], ignore_index=True)  # Use pd.concat instead of append
            self.saveToData()
            print("Registration Completed")

        else:
            name = self.StdData[self.StdData['Index'] == index].values[0][0]
            print(f"{name} Already Exist in the Database")
      
    def saveToData(self):
        self.StdData.to_csv("utilsFiles\StdDetails.csv", index=False)
        np.save("utilsFiles\\encodingArr.npy", self.encodingArr)
        np.save("utilsFiles\\labelArr.npy", self.labelArr)
        print("Saved Sucessfully")



class Recognize(Register):
    def __init__(self) :
        super().__init__()

    def start_face_recognition(self):
    
        
        name="unknown"
        data=pd.read_csv("utilsFiles/StdDetails.csv")
        video_capture = cv.VideoCapture(0)

        face_locations = []
        face_encodings = []
        face_names = []
        
        while True:


            ret, frame = video_capture.read()
            
            start=time.time()
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame)

            face_names = []


            for face_encoding in face_encodings:
                
                face_encoding=face_encoding.reshape(1,-1)
                find,i=super().checkPresence(face_encoding)
                if find:
                    
                    name=data[data['Index']==i].values[0][0]
                    print(f"Person {name} detected")
                    face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):

                # Draw a box around the face
                #print("Creating Rectangles")
                cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                font = cv.FONT_HERSHEY_DUPLEX
                #print(name)
                print(f"Name {name} and type {type(name)}")
                cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            end=time.time()
            cv.imshow('Video', frame)

            time1=end-start
            print(f"Total Time{time1}")
            if cv.waitKey(1) & 0xFF == ord('q'):
                break


        video_capture.release()
        cv.destroyAllWindows()

    def capture_good_quality_frame(self):
        cap = cv.VideoCapture(0)

        # Set camera parameters for better quality
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)  
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080) 
        cap.set(cv.CAP_PROP_FPS, 30)  

    
        for _ in range(30):
            _, _ = cap.read()

        while cap.isOpened():
            find, frame = cap.read()

            

            cv.imshow("Frame", frame)

            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    
        cap.release()
        cv.destroyAllWindows()

        return frame
    

class Start(Recognize):
    def __init__(self):
        super().__init__()

    def launch(self):
        while True:

            x = int(input("Press 1 to register\nPress 2 to start Verification\nPress -1 to exit"))
            if x == 1:
                name = input("Enter your name")
                verified = 0
                while not verified:
                    frame = super(Start, self).capture_good_quality_frame()
                    cv.imshow(name, frame)
                    cv.waitKey(0)
                    cv.destroyAllWindows()
                    encoding = face_recognition.face_encodings(frame)
                    if encoding:

                        encoding=encoding[0]
                        verified = int(input("Enter 1 to verify\nElse enter 0"))
                
                encoding = encoding.reshape(1, -1)
                reg = Register()
                reg.addPerson(encoding, name)

            if x == 2:
                super(Start, self).start_face_recognition()
            if x == -1:
                break

st = Start()
st.launch()