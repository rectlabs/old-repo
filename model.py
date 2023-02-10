import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas 
import cv2 as cv
from typing import Generator
from sklearn.cluster import KMeans

# Autoencoder class
class ConvAutoencoder(nn.Module):
    def __init__(self, channels):
        super(ConvAutoencoder, self).__init__()
        # Encoder layer
        self.conv1 = nn.Conv2d(channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Decoder layer
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = torch.nn.ConvTranspose2d(16, 3, 2, stride=2)
        
    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        return x
    
    def decoder(self, x):
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
        
        return x
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x
    
# Kmeans class
class Kmeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        
    def train(self, x):
        self.kmeans.fit(x)
        
    def predict(self, x):
        return self.kmeans.predict(x)
        
class Detector:
    def __init__(self, bestWeights, local=True):
        if local:
            self.detector = torch.hub.load("yolov5", "custom", path=bestWeights, source="local")
        else:
            self.detector = torch.hub.load("yolov5", "custom", path=bestWeights)
            
    def detect(self, frame: Generator) -> pandas.DataFrame:
       # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        detections = self.detector(frame)
        
        return detections.pandas().xyxy[0]
    
    


        