"""Wrapper class for face recognition
while working with opencv
"""

__version__ = '0.0.1'
__author__ = "Campbell Mercer-Butcher"

import cv2
import numpy

class Recogniser():
    """Wrapper class from face recognition"""
    def __init__(self):
        self.face_recogniser = cv2.face.LBPHFaceRecognizer_create()
        
    def train(self, faces, labels):
        """Trains to recognise faces based on inputed faces and labels"""
        self.face_recogniser.train(faces, numpy.array(labels))
        
    def predict(self, face, frame, location):
        """Creates prediction on suplied face"""
        label, confidence = self.face_recogniser.predict(face)
        if confidence>80:
            label = None
        else:
            self._draw_text(frame, label, location)
        return label, frame
    
    def _draw_text(self, frame, label, location):
        """Draws text in location (x,y)"""
        cv2.putText(frame, str(label), location, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)