"""Detects faces in video stream and applies facial 
recognition to create profiles of detected faces"""
__author__ = "Campbell Mercer-Butcher"

from cv2_detect_wrapper import Detector
from face_recogniser import Recogniser
import os
import numpy
import cv2

CASCADE_PATH = "haarcascade_frontalface_default.xml"
SCALE_FACTOR = 1.2
MIN_NEIGHBORS = 5
MIN_SIZE = (30,30)
RECORDS_FOLDER_PATH = "records"
PROFILE_PREFIX = "P"
INP_QUIT = 0
INP_CAPTURE = 2


def main():
    detector = Detector(CASCADE_PATH)
    recogniser = Recogniser()
    
    faces, labels = gather_training_data(RECORDS_FOLDER_PATH, detector)
    
    if len(faces):
        recogniser.train(faces, labels)
    
    while True:
        detector.video_read()
        detector.detect(SCALE_FACTOR, MIN_NEIGHBORS, MIN_SIZE)
        labels, faces = predict(recogniser, detector)
        detector.draw_and_display()
        input_ = detector.take_input(RECORDS_FOLDER_PATH)
        if input_ == INP_QUIT:   
            break
        elif input_ == INP_CAPTURE:
            next_prof = next_profile()
            labels = detector.save_faces(RECORDS_FOLDER_PATH, labels, next_prof)
            recogniser.train(faces, labels)

def gather_training_data(folder_path, detector):
    """Gathers previously recorded faces and converts to data for training"""
    dirs = []
    try:
        dirs = os.listdir(folder_path)
    except FileNotFoundError:
        os.mkdir(folder_path)
    faces = []
    labels = []
    
    for dir_name in dirs:
        if not dir_name.startswith(PROFILE_PREFIX):
            continue;
        
        label = int(dir_name[1])
        subject_dir_path = folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            detector.input_frame(image)
            detector.detect(SCALE_FACTOR, MIN_NEIGHBORS, MIN_SIZE)
            (x,y,w,h) = detector.faces[0]
            face = detector.gray[y:y+w, x:x+h]
            if not face.any():
                continue;
            
            faces.append(face)
            labels.append(label)
    return faces, labels


def predict(recogniser, detector):
    """Loops through faces to apply recognition"""
    labels = []
    faces = []
    for (x,y,w,h) in detector.faces:
        face = detector.gray[y:y+w, x:x+h]
        frame = detector.frame
        label, detector.frame = recogniser.predict(face, frame, (x,y))
        faces.append(face)
        labels.append(label)   
    return labels, faces

def next_profile():
    """Finds label for next profile"""
    dirs = os.listdir(RECORDS_FOLDER_PATH)
    profiles = [0]
    for dir_name in dirs:
        if not dir_name.startswith(PROFILE_PREFIX):
            continue;
        profiles.append(int(dir_name[1]))
    return max(profiles) + 1    

main()

    
    


