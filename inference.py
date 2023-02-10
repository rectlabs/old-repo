import cv2 as cv
import torch
import pickle
from torchvision import transforms
from model import ConvAutoencoder

MODEL_PATH = "autoencoder.pth"

def getAllBoundingBoxes(detections_df):
    detections_map = {}
    nRows = detections_df.shape[0]
    for row in range(nRows):
        rowInfo = detections_df.iloc[row]
        if rowInfo["name"] in detections_map:
            detections_map[rowInfo["name"]].append([int(rowInfo["xmin"]), int(rowInfo["xmax"]), int(rowInfo["ymin"]), int(rowInfo["ymax"])])
        else:
            detections_map[rowInfo["name"]] = [[int(rowInfo["xmin"]), int(rowInfo["xmax"]), int(rowInfo["ymin"]), int(rowInfo["ymax"])]]
            
    return detections_map
    
def getPlayers(frame, bboxes):
    players = []
    for box in bboxes:
        xmin, xmax, ymin, ymax = box
        players.append(frame[ymin:ymax, xmin:xmax])
    return players

def drawBoundingBoxes(frame, teams, playersBboxes):
    for teamIdx, player in zip(teams, playersBboxes):
        xmin, xmax, ymin, ymax = player
        if teamIdx == 0:
            frame = cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
        elif teamIdx == 1:
            frame = cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
                
    return frame

def refereeBoundingBoxes(frame, refereeBoxes):
    for referee in refereeBoxes:
        xmin, xmax, ymin, ymax = referee
        frame = cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 1)
        
    return frame

def ballBoundingBox(frame, ballBox):
    for ball in ballBox:
        xmin, xmax, ymin, ymax = ball
        frame = cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        
    return frame

def goalBoundingBox(frame, goalBox):
    for goal in goalBox:
        xmin, xmax, ymin, ymax = goal
        frame = cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        
    return frame

def teamClassifier(players):
    model = ConvAutoencoder(3)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    with open("kmeans", "rb") as F:
        kmeans = pickle.load(F)
    teamIdx = []
    for player in players:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])
        player = transform(player)
        featureVector = model.encoder(player)
        featureVector = featureVector.detach().numpy()
        featureVector = featureVector.reshape((1, -1))
        
        label = kmeans.predict(featureVector)
        teamIdx.append(label)
    return teamIdx

path = "test-video.mp4"

detector = torch.hub.load("yolov5", "custom", path="best.pt", source="local")

cap = cv.VideoCapture(path)

saveAs = "demo.avi"
fps = int(cap.get(cv.CAP_PROP_FPS))
videoWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
videoHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

videoWriter = cv.VideoWriter(saveAs, fourcc=cv.VideoWriter_fourcc(*'MJPG'), fps=fps, frameSize=(videoWidth, videoHeight))
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Frame is broken! Exiting...")
        break
        
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    detection = detector(frame)
    results_df = detection.pandas().xyxy[0]
    
    bboxes = getAllBoundingBoxes(results_df)
    if bboxes != {}:
        players = getPlayers(frame, bboxes["Player"])
        teamIdx = teamClassifier(players)
        
    frame = drawBoundingBoxes(frame, teamIdx, bboxes["Player"])
    
    if "Referee" in bboxes:
        frame = refereeBoundingBoxes(frame, bboxes["Referee"])
        
    if "Ball" in bboxes:
        frame = ballBoundingBox(frame, bboxes["Ball"])
        
    if "Goalkeeper" in bboxes:
        frame = goalBoundingBox(frame, bboxes["Goalkeeper"])
    
    cv.imshow("inference", frame)
    
    videoWriter.write(frame)
    
    if cv.waitKey(25) == ord("q"):
        break
    
cap.release()
videoWriter.release()
cv.destroyAllWindows()
    
    
    