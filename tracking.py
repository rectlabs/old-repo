from elements.deep_sort import DEEPSORT
from elements.assets import detect_color
import torch
from model import Detector
import cv2 as cv
from video import Video

def df2List(df):
    list_of_dict = []
    n_rows = df.shape[0]
    
    for row in range(n_rows):
        content = df.iloc[row]
        list_of_dict.append({"label":content["name"], "bbox":[(content["xmin"], content["ymin"]), (content["xmax"], content["ymax"])],
                             "score":content["confidence"], "cls":content["class"]})
        
    return list_of_dict

config = "deep_sort_pytorch\\configs\\deep_sort.yaml"

deep_sort = DEEPSORT(config)
# detector = YOLO(model_path="best.pt", conf_thres=0.5, iou_thres=0.5)
detector = Detector("best.pt")

cap = cv.VideoCapture("test-video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Frame is broken! Exiting...")
        break
    
    detections = detector.detect(frame)
    detections = df2List(detections)
    
    if detections:
        deep_sort.detection_to_deepsort(detections, frame)
    else:
        deep_sort.deepsort.increment_ages()
        
    
    # print(type(outputs))
    # print(len(outputs))
    cv.imshow("frame", frame)
    
    if cv.waitKey(1) & ord('q') == 0xFF:
        break
    
cap.release()
cv.destroyAllWindows()

