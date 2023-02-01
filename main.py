import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas
import pickle
from typing import Generator, Dict, List
import cv2 as cv
from torchvision import transforms
from video import Video
from data import PlayerDataset, getPlayers, createDataLoader
from model import ConvAutoencoder, Kmeans, Detector

def getBoundingBoxesMap(detections_df: pandas.DataFrame) -> Dict[str, List]:
    detections_map = {}
    nRows = detections_df.shape[0]
    for row in range(nRows):
        rowInfo = detections_df.iloc[row]
        if rowInfo["name"] in detections_map:
            detections_map[rowInfo["name"]].append([int(rowInfo["xmin"]), int(rowInfo["xmax"]), int(rowInfo["ymin"]), int(rowInfo["ymax"])])
        else:
            detections_map[rowInfo["name"]] = [[int(rowInfo["xmin"]), int(rowInfo["xmax"]), int(rowInfo["ymin"]), int(rowInfo["ymax"])]]
            
    return detections_map
       
def generateDataset(videoPath: str, nBurn: int, detectorWeights):
    # 1. Create Generator object
    videoObject = Video(path=videoPath)
    frameGenerator = videoObject.readFrame()
    frameIter = iter(frameGenerator)
    
    # 2. Create detector object
    detector = Detector(detectorWeights)
    
    # 3. Loop through the generator
    numOfFrames = 0
    croppedPlayers = []
    print(f"Generating Training from the first {nBurn} Frames...")
    for frame in frameIter:
        # 3i. Detect Players, Goalkeeper, ball and Referees
        df = detector.detect(frame)
        bboxesMap = getBoundingBoxesMap(df)
        if numOfFrames <= nBurn:
            if bboxesMap != {} and "Player" in bboxesMap:
                players = getPlayers(frame, bboxesMap["Player"])
                croppedPlayers += players
        
        if numOfFrames == nBurn:
            dataset = createDataLoader(croppedPlayers, batch_size=10)
            print(f"Dataset Generated!")
            return dataset
        
        numOfFrames += 1
        
def trainAutoencoder(model, epochs: int, dataLoader):
    # set loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training Autoencoder...")
    print("------------------------")
    for epoch in range(1, epochs+1):
        # monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for images in dataLoader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model.forward(images)
            # calculate the loss
            loss = criterion(outputs, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*images.size(0)
                
        # print avg training statistics 
        train_loss = train_loss/len(dataLoader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))
    
    print()
    print("Training Successful!")
        
    torch.save(model.state_dict(), os.getcwd() + "\\autoencoder.pth")
    print("Model successfully saved!")
    
    return model

def trainKmeans(dataset, autoencoder, kmeans):
    featureVectors = []
    for images in dataset:
        batchLength = len(images)
        featureVector = autoencoder.encoder(images).detach().numpy()
        featureVector = featureVector.reshape((batchLength, -1))
        featureVectors.append(featureVector)
        
    featureVectors = tuple(featureVectors)
    featureVectors = np.concatenate(featureVectors, axis=0)
    
    kmeans.train(featureVectors)
    # labels = kmeans.predict(featureVectors)
    file = os.getcwd() + "\\kmeans"
    with open(file, "wb") as F:
        pickle.dump(kmeans, F)
        
    print("KMeans successfully saved!")
    # return featureVectors, labels
        
def train(dataLoader, epochs):
    # 1. Initialize autoencoder and kmeans clustering. n_clusters=2 because we are separating players in two teams
    autoencoder = ConvAutoencoder(channels=3)
    kmeans = Kmeans(n_clusters=2)
    
    autoencoder = trainAutoencoder(autoencoder, epochs=epochs, dataLoader=dataLoader)
    trainKmeans(dataLoader, autoencoder, kmeans)

    
def main():
    dataset = generateDataset(videoPath="Record-2.mp4", nBurn=100, detectorWeights="best.pt")

    train(dataset, 100)

if __name__ == "__main__":
    main()
