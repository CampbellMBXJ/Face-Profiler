"""Wrapper class for a layer of abbstraction
while working with face detection in opencv
"""

__version__ = '0.4.2'
__author__ = "Campbell Mercer-Butcher"

import cv2
import os
import pathlib
import time

CAPTURE_KEYCODE = 32 # spacebar
QUIT_KEYCODE = 113 # "q"

class Detector():
    """Wrapper Class for face detection"""
    def __init__(self, cascade, webcam=1):
        self._webcam = webcam
        self._face_cascade = cv2.CascadeClassifier(cascade)
        self._video_capture = cv2.VideoCapture(0)
    
    def video_read(self):
        """Captures current frame in colour and greyscale"""
        self.frame = self._video_capture.read()[self._webcam]
        try:
            self.frame_clean = self.frame.copy()
        except AttributeError:
            raise NoWebcamError(self._webcam)
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
        if input_ == CAPTURE_KEYCODE:
            return 2
        elif input_ == QUIT_KEYCODE:
            self._end()
            return 0
        return 1
    
    def save_faces(self, records_path, labels, next_profile):
        """Crops image to faces and saves them"""
        try:
            dirs = os.listdir(records_path)
        except FileNotFoundError:
            raise MissingDirectoryError(records_path)
        
        curr_time = int(round(time.time()))
        for i in range(len(self.faces)):
            x, y, w, h = [v for v in self.faces[i]]
            sub_face = self.frame_clean[y-20:y+h+20, x-20:x+w+20]
            if not labels[i]:
                pathlib.Path("{0}/P{1}".format(records_path,next_profile)).mkdir(exist_ok=True)
                cv2.imwrite("{0}/P{1}/{2}.jpg".format(records_path,next_profile,curr_time), sub_face)
                labels[i] = next_profile
                next_profile+=1
            else:
                cv2.imwrite("{0}/P{1}/{2}.jpg".format(records_path,labels[i],curr_time), sub_face)
        return labels
    
    def _end(self):
        """Terminate video stream and close windows"""
        self._video_capture.release()
        cv2.destroyAllWindows()
        

class DetectorError(Exception):
    """Base exception for errors raised by Detector"""
    def __init__(self, msg=None):
        if msg is None:
            msg = "An error occured in Detector"
        super().__init__(msg)
        
class NoWebcamError(DetectorError):
    """Webcam Cant be detected"""
    def __init__(self, webcam):
        msg = "Webcam {0} not detected".format(webcam)
        super().__init__(msg)

class MissingDirectoryError(DetectorError):
    """Supplied directory not found"""
    def __init__(self, directory):
        msg = "{0} does not exist".format(directory)
        super().__init__(msg)