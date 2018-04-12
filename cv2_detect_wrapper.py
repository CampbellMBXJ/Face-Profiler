"""Wrapper class for a layer of abbstraction
while working with face detection
"""

__version__ = '0.2'
__author__ = "Campbell Mercer-Butcher"

import cv2
import sys

class Detector():
    """Wrapper Class for face detection"""
    def __init__(self, cascade):
        self._face_cascade = cv2.CascadeClassifier(cascade)
        self._video_capture = cv2.VideoCapture(0)
    
    def video_read(self):
        """Captures current frame in colour and greyscale"""
        self.frame = self._video_capture.read()[1]
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
        
    def take_input(self):
        """Detects user input and returns True when q is pressed"""
        input_ = cv2.waitKey(1)
        if input_ == 32:
            pass
        elif input_ == ord('q'):
            self.end()
            return True
        return False
    
    def end(self):
        """Terminate video stream and close windows"""
        self._video_capture.release()
        cv2.destroyAllWindows()