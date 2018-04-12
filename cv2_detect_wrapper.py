import cv2
import sys

class detector():
    """Wrapper Class for face detection"""
    def __init__(self, cascade):
        self._MIN_SIZE = (30,30)
        self._face_cascade = cv2.CascadeClassifier(cascade)
        self._video_capture = cv2.VideoCapture(0)
    
    def video_read(self):
        """Captures current frame in colour and greyscale"""
        self.frame = self._video_capture.read()[1]
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
    
    def detect(self, scale_factor, min_neighbors):
        """Detect instances in current frame"""
        self.faces = self._faceCascade.detectMultiScale(
            self.gray,
            scaleFactor=scale_factor,
            minNeighbors=5,
            minSize=self._MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
    def draw_and_display(self):
        """Draw rectangles and display video"""
        for (x, y, w, h) in self.faces:
            cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  
        cv2.imshow('Video', self.frame)
        
    def take_input(self):
        """Uses user input to determine next step"""
        input_ = cv2.waitKey(1)
        if input_ == 32:
            pass
        elif input_ == ord('q'):
            self.end()
    
    def end(self):
        """Terminate video stream and close windows"""
        self._video_capture.release()
        cv2.destroyAllWindows()