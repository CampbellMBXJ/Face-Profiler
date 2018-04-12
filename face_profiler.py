from cv2_detect_wrapper import Detector

CASCADE_PATH = "haarcascade_frontalface_default.xml"
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = (30,30)

detector = Detector(CASCADE_PATH)

while True:
    detector.video_read()
    detector.detect(SCALE_FACTOR, MIN_NEIGHBORS, MIN_SIZE)
    detector.draw_and_display()
    if detector.take_input():
        break