"""Wrapper class for face recognition
while working with opencv
"""
__author__ = "Campbell Mercer-Butcher"

import cv2
import numpy

MIN_CONFIDENCE = 80

class Recogniser():
    """Wrapper class from face recognition"""
    def __init__(self):
        self.face_recogniser = cv2.face.LBPHFaceRecognizer_create()
        self._trained = False
        
    def train(self, faces, labels):
        """Trains to recognise faces based on inputed faces and labels"""
        self.face_recogniser.train(faces, numpy.array(labels))
        self._trained = True
        
    def predict(self, face, frame, location):
        """Creates prediction on suplied face"""
        if not self._trained:
            return None, frame
        
        label, confidence = self.face_recogniser.predict(face)
        
        if confidence > MIN_CONFIDENCE:
            label = None
        else:
            self._draw_text(frame, label, location)
        return label, frame
    
    def _draw_text(self, frame, label, location):
        """Draws text in location (x,y)"""
        cv2.putText(frame, str(label), location, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
