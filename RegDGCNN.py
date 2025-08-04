import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils.knn import get_GraphFeatures


# RegDGCNN - Regression Dynamic Graph Convoluted Neural Networks
# RegDGCNN can be used for Geometrical 3D point cloud modeling and prediction
# RegDGCNN works using EdgeConvolutions, operating on graphs constructed using K Nearest Neighbors of each point
# Output generated for each point cloud

# Each point cloud -> [Batchsize, Channels, NoOfPoints]
class EdgeConv(nn.Module):
    """
    Develops an EdgeConvolution. 
    For every point x, design space is explored using KNN.
    For each point x_i, a difference vector x_j-x_i is computed, representing an edge
    So now each edge is represented by 2 * input_channel (x,y,z)
    """
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2,out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        # x: [B, C, N]
        x = get_GraphFeatures(x, k=self.k) # [B, 2C, N, k]
        x = self.conv(x) # [B, out_channel, N, k]
        x = x.max(dim=-1)[0] # [B, out_channel, N]
        return x
    
class RegDGCNN(nn.Module):
    """
    RegDGCNN forms layers of EdgeConvolutions.
    First layer looks at the point and local neighbours. Output -> Local Edge Features
    Second Layer looks at Edge Feature. Output -> Patterns arounds points
    Subsequent Layers abstracts informations and learns more not only its own shape, but shape of the full object

    Combining all layers -> from simple point to edge features to complex shape understanding
    
    Output of RegDGCNN -> Predicted Pressure at each point
    """
    
    def __init__(self, k=20):
        super().__init__()
        self.k = k
        # [B, 3, N] -> [B, 6, N] -> [B, 64, N]        
        self.ec1 = EdgeConv(3, 64, k)
        # [B, 64, N] -> [B, 128, N] -> [B, 64, N]  
        self.ec2 = EdgeConv(64, 64, k)
        # [B, 64, N] -> [B, 128, N] -> [B, 128, N]
        self.ec3 = EdgeConv(64, 128, k)
        # [B, 128, N] -> [B, 256, N] -> [B, 256, N]
        self.ec4 = EdgeConv(128, 256, k)
        
        # Dense Layer for Regression
        self.conv = nn.Sequential(
            # EdgeConv layers concated and passed through dense layers
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # [B, 1, N]
            nn.Conv1d(128, 1, 1)
        )
    
    def forward(self, x):
        # x: [B, N, C]
        x = x.permute(0, 2, 1) # [B, N, C] -> [B, C, N]
        
        x1 = self.ec1(x)
        x2 = self.ec2(x1)
        x3 = self.ec3(x2)
        x4 = self.ec4(x3)
        
        x_cat = torch.cat((x1, x2, x3, x4), dim=1) # [B, 512, N]
        output = self.conv(x_cat) # [B, 1, N]
        return output.squeeze(1) # [B, N]