import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeNetCFD(Dataset):
    """Utils class to load the Dataset
    Helper functions to get the len and index the files
    Normalizes the Mesh values

    Returns:
        Returns a Dict of points (mesh) and Pressure Values as tensors"""
    def __init__(self, file, data_dir = "Processed_Data", normalize=True):
        with open(file, "r") as f:
            self.ids = f.read().strip().split(',')
        self.data_dir = data_dir
        self.normalize = normalize
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        sample_id = self.ids[index].zfill(3)
        path = os.path.join(self.data_dir, f"{sample_id}.npz")
        data = np.load(path)
        
        points = data["points"]
        pressure = data["pressure"]
        
        if self.normalize:
            mean = np.mean(points, axis=0)
            points = points - mean
            scale = np.max(np.linalg.norm(points, axis=1))
            points = points / scale
            
        return {
            "points": torch.from_numpy(points).float(),
            "pressure": torch.from_numpy(pressure).float()
        }