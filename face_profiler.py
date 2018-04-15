from cv2_detect_wrapper import Detector
import os
import numpy

CASCADE_PATH = "haarcascade_frontalface_default.xml"
SCALE_FACTOR = 1.2
MIN_NEIGHBORS = 5
MIN_SIZE = (30,30)
RECORDS_FOLDER_PATH = "records"

detector = Detector(CASCADE_PATH)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

faces, labels = gather_training_data(RECORDS_FOLDER_PATH, detector)
face_recognizer.train(faces, np.array(labels))


while True:
    detector.video_read()
    detector.detect(SCALE_FACTOR, MIN_NEIGHBORS, MIN_SIZE)
    detector.draw_and_display()
    if detector.take_input(RECORDS_FOLDER_PATH):
        break
    
    
def gather_training_data(folder_path, detector):
    """Gathers previously recorded faces and converts to data for training"""
    dirs = os.listdir(folder_path)
    faces = []
    labels = []
    
    for dir_name in dirs:
        if not dir_name.startswith("P"):
            continue;
        
        label = int(dir_name[1])
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            detector.input_frame(image)
            detector.detect(SCALE_FACTOR, MIN_NEIGHBORS, MIN_SIZE)
            faces = detector.faces
            if len(faces) == 0:
                continue;
            
            face = faces[0]
            faces.append(face)
            labels.append(label)
    return faces, labels

#def predict(test_img):
    ##make a copy of the image as we don't want to chang original image
    #img = test_img.copy()
    ##detect face from the image
    #face, rect = detect_face(img)

    ##predict the image using our face recognizer 
    #label, confidence = face_recognizer.predict(face)
    ##get name of respective label returned by face recognizer
    #label_text = subjects[label]
    
    ##draw a rectangle around face detected
    #draw_rectangle(img, rect)
    ##draw name of predicted person
    #draw_text(img, label_text, rect[0], rect[1]-5)
    
    #return img