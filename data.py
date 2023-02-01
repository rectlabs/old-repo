from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from typing import List

class PlayerDataset(Dataset):
    def __init__(self, X, transform):
        self.X = X
        self.n_samples = len(X)
        self.transform = transform
        
    def __getitem__(self, idx):
        return self.transform(self.X[idx])
    
    def __len__(self):
        return self.n_samples
    
def getPlayers(frame: np.ndarray, bboxes: List[List]) -> List[np.ndarray]:
    players = []
    for box in bboxes:
        xmin, xmax, ymin, ymax = box
        players.append(frame[ymin:ymax, xmin:xmax])
    return players

def createDataLoader(data: List[np.ndarray], batch_size: int):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])
    dataset = PlayerDataset(data, transform=transform)
    dataset = DataLoader(dataset, batch_size=batch_size)
    
    return dataset
