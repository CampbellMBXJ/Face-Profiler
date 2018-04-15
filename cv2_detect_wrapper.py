"""Wrapper class for a layer of abbstraction
while working with face detection in opencv
"""

__version__ = '0.2.2'
__author__ = "Campbell Mercer-Butcher"

import cv2
import sys
import os
import pathlib
import time

class Detector():
    """Wrapper Class for face detection"""
    def __init__(self, cascade):
        self._face_cascade = cv2.CascadeClassifier(cascade)
        self._video_capture = cv2.VideoCapture(0)
    
    def video_read(self):
        """Captures current frame in colour and greyscale"""
        self.frame = self._video_capture.read()[1]
        self.frame_clean = self.frame.copy()
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        
    def input_frame(self, frame):
        self.frame = frame
        self.frame_clean = self.frame.copy()
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
    
    def detect(self, scale_factor, min_neighbors, min_size):
        """Detect instances in current frame"""
        self.faces = self._face_cascade.detectMultiScale(
            self.gray,
            scaleFactor=scale_factor,
            minNeighbors=5,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
    def draw_and_display(self):
        """Draw rectangles and display video"""
        for (x, y, w, h) in self.faces:
            cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  
        cv2.imshow('Video', self.frame)        
        
    def take_input(self, records_path):
        """Detects user input and returns True when q is pressed"""
        input_ = cv2.waitKey(1)
        if input_ == 32:
            self._save_faces(records_path)
        elif input_ == ord('q'):
            self._end()
            return True
        return False
    
    def _save_faces(self, records_path):
        """Crops image to faces and saves them"""
        dirs = os.listdir(records_path)
        profiles = [0]
        for dir_name in dirs:
            if not dir_name.startswith("P"):
                continue;
            profiles.append(int(dir_name[1]))
        num_profs = max(profiles) + 1
        curr_time = int(round(time.time()))
        for i in range(len(self.faces)):
            x, y, w, h = [v for v in self.faces[i]]
            sub_face = self.frame_clean[y-20:y+h+20, x-20:x+w+20]
            pathlib.Path("records/P{0}".format(num_profs+i)).mkdir(exist_ok=True)
            cv2.imwrite("records/P{0}/{1}.jpg".format(num_profs+i,curr_time), sub_face)        
    
    def _end(self):
        """Terminate video stream and close windows"""
        self._video_capture.release()
        cv2.destroyAllWindows()